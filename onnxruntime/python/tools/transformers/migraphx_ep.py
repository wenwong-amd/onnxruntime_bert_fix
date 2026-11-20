#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
# Shared helper to make the MIGraphX execution provider usable across ONNX Runtime
# versions that register it as a built-in EP as well as newer versions that ship it
# as a plugin EP library that must be registered and attached to SessionOptions
# via the OrtEpDevice/add_provider_for_devices() API before use.
import glob
import os

import onnxruntime

MIGRAPHX_EP = "MIGraphXExecutionProvider"

# PCI vendor id ONNX Runtime uses to identify AMD GPU allocators/data-transfers,
# needed to bind IO to the MIGraphX EP's device explicitly (see
# bind_migraphx_output() below).
try:
    MIGRAPHX_VENDOR_ID = onnxruntime.OrtDeviceVendorId.AMD
except AttributeError:
    MIGRAPHX_VENDOR_ID = 0x1002  # AMD PCI vendor ID fallback


def _find_migraphx_plugin_libs(registration_name):
    """Return (registration_name, library_path) pairs for the MIGraphX plugin EP library.

    Resolution order:
      1. Explicit override via the MIGRAPHX_EP_LIB environment variable.
      2. The separately-installable 'onnxruntime_ep_migraphx' package (built by
         onnxruntime-ep-amdgpu), which exposes get_ep_names()/get_library_paths()
         (or get_library_path() in older releases).
      3. Inside the onnxruntime package itself, for builds that ship the plugin
         EP library alongside the classic provider libraries.
    """
    override = os.environ.get("MIGRAPHX_EP_LIB")
    if override:
        return [(registration_name, override)]

    try:
        import onnxruntime_ep_migraphx  # noqa: PLC0415

        if hasattr(onnxruntime_ep_migraphx, "get_library_paths"):
            return list(
                zip(onnxruntime_ep_migraphx.get_ep_names(), onnxruntime_ep_migraphx.get_library_paths(), strict=True)
            )
        return [(registration_name, onnxruntime_ep_migraphx.get_library_path())]
    except (ImportError, FileNotFoundError, AttributeError):
        pass

    package_dir = os.path.dirname(os.path.abspath(onnxruntime.__file__))
    patterns = (
        "capi/libonnxruntime_providers_migraphx.*",
        "libonnxruntime_providers_migraphx.*",
    )
    for pattern in patterns:
        matches = glob.glob(os.path.join(package_dir, pattern))
        if matches:
            return [(registration_name, matches[0])]
    return []


def _find_migraphx_ep_device():
    """Return the MIGraphX OrtEpDevice provided by a registered plugin EP library, if any."""
    if not hasattr(onnxruntime, "get_ep_devices"):
        return None
    for ep_device in onnxruntime.get_ep_devices():
        if ep_device.ep_name == MIGRAPHX_EP:
            return ep_device
    return None


def is_builtin_migraphx_ep():
    # Newer onnxruntime also lists registered plugin EPs in get_available_providers(),
    # so only treat MIGraphX as built-in when no plugin OrtEpDevice backs it.
    return MIGRAPHX_EP in onnxruntime.get_available_providers() and _find_migraphx_ep_device() is None


def _register_migraphx_plugin(registration_name):
    ep_device = _find_migraphx_ep_device()
    if ep_device is not None:
        return ep_device

    if not hasattr(onnxruntime, "register_execution_provider_library"):
        raise RuntimeError(
            f"This onnxruntime ({onnxruntime.__version__}) has neither a "
            f"built-in MIGraphX EP nor the plugin registration API"
        )

    libs = _find_migraphx_plugin_libs(registration_name)
    if not libs:
        raise RuntimeError("MIGraphX plugin EP library not found; set MIGRAPHX_EP_LIB to its path")

    # migraphx must be imported before loading the plugin EP library to initialize the ROCm runtime.
    import migraphx  # noqa: F401, PLC0415

    for name, lib_path in libs:
        onnxruntime.register_execution_provider_library(name, lib_path)

    ep_device = _find_migraphx_ep_device()
    if ep_device is None:
        lib_paths = [lib_path for _, lib_path in libs]
        raise RuntimeError(
            f"Registered MIGraphX plugin EP from {lib_paths} but no matching "
            f"OrtEpDevice was found (see onnxruntime.get_ep_devices()). This "
            f"usually means ONNX Runtime's hardware device discovery didn't "
            f"find a usable GPU for the plugin EP to attach to -- check the "
            f"device discovery warnings logged at startup (e.g. a broken "
            f"/sys/class/drm/card*/device entry aborting GPU enumeration)."
        )
    return ep_device


def register_migraphx_ep(registration_name=MIGRAPHX_EP):
    """Ensure the MIGraphX execution provider is registered and available.

    Works on both classic builds (MIGraphX registered as a built-in EP, so this is
    a no-op) and newer plugin-EP builds where the EP library must be registered via
    ``register_execution_provider_library`` before it shows up in
    ``get_available_providers()``. Raises if MIGraphX cannot be made available so a
    failure is loud instead of silently falling back to another provider.

    Note: on plugin-EP builds where the classic providers list never reflects the
    registered library (because the EP requires an explicit OrtEpDevice attached to
    SessionOptions instead), this still registers the library but callers must use
    ``ensure_migraphx_ep(session_options, ...)`` to actually select the device.
    """
    if is_builtin_migraphx_ep():
        return

    if hasattr(onnxruntime, "register_execution_provider_library"):
        _register_migraphx_plugin(registration_name)
        return

    raise RuntimeError(
        f"This onnxruntime ({onnxruntime.__version__}) has neither a built-in "
        f"MIGraphX EP nor the plugin registration API"
    )


def ensure_migraphx_ep(session_options, provider_options=None, registration_name=MIGRAPHX_EP):
    """Ensure the MIGraphX execution provider is available and usable.

    Works on both classic builds (MIGraphX registered as a built-in EP, selected
    via the 'providers' list passed to InferenceSession) and newer plugin-EP
    builds (where the EP library must be registered and its OrtEpDevice attached
    directly to `session_options` via add_provider_for_devices() before creating
    the session).

    session_options: an onnxruntime.SessionOptions() instance that will be
    passed as `sess_options` to InferenceSession(). It may be mutated in-place
    (plugin EP path) to attach the MIGraphX device.

    provider_options: optional dict of MIGraphX EP options (e.g.
    {"migraphx_fp16_enable": "1"}). The plugin EP accepts its own unprefixed names
    and the classic EP's "migraphx_"-prefixed names listed in onnxruntime-ep-amdgpu's
    mgx_options.h kLegacyOptionAliases; other classic-only options (e.g.
    migraphx_max_dynamic_batch) are rejected with EP_FAIL at session creation.

    Returns the 'providers' list to pass as InferenceSession(providers=...).
    On the plugin EP path this is None, since the provider is already configured
    on `session_options`; pass it through as-is (InferenceSession treats
    providers=None as "use what's on session_options").
    """
    if is_builtin_migraphx_ep():
        # Classic built-in EP: select it via the standard providers list.
        if provider_options:
            return [(MIGRAPHX_EP, provider_options)]
        return [MIGRAPHX_EP]

    ep_device = _register_migraphx_plugin(registration_name)

    # Plugin EP: attach the discovered OrtEpDevice directly to session_options.
    session_options.add_provider_for_devices([ep_device], provider_options or {})
    return None


def bind_migraphx_output(io_binding, name, shape, element_type, device_id=0):
    """Bind an IOBinding output to the MIGraphX EP's GPU device.

    `IOBinding.bind_output(name, "cuda")` builds its OrtDevice via onnxruntime's
    legacy vendor-less constructor, which only fills in the AMD vendor id
    automatically when onnxruntime core itself was compiled with --use_migraphx.
    Plugin-EP builds don't set that, so the resulting OrtDevice has vendor_id=0
    and ORT can't find an allocator for it ("Failed to find allocator for device
    ... VendorId:0 ..."). Creating the output OrtValue explicitly with
    vendor_id=AMD and binding that avoids the issue on both classic and
    plugin-EP builds.

    Returns the bound onnxruntime.OrtValue; after running the session, read its
    contents back with e.g. `io_binding.copy_outputs_to_cpu()` or
    `output_ortvalue.numpy()`.
    """
    output_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        shape, element_type, "cuda", device_id, MIGRAPHX_VENDOR_ID
    )
    io_binding.bind_ortvalue_output(name, output_ortvalue)
    return output_ortvalue
