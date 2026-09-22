# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Mesh rendering on top of an input image, via pyrender."""

import os
import platform
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _egl_available() -> bool:
    """Return True if a *hardware* EGL device is available (GPU offscreen OpenGL).

    Probes libEGL via ctypes: enumerates devices with eglQueryDevicesEXT (the
    same enumeration pyrender relies on for headless GPU rendering), then queries
    each device's extension string and counts only real GPUs -- i.e. those
    exposing a DRM node (EGL_EXT_device_drm) and not flagged as software
    rasterizers (EGL_MESA_device_software, e.g. llvmpipe/swrast). No GL context
    is created, so this is cheap and side-effect free.
    """
    import ctypes
    from ctypes import util, c_void_p, c_int, c_char_p, POINTER, byref, CFUNCTYPE

    EGL_EXTENSIONS = 0x3055

    egl_lib = None
    candidates = ([util.find_library("EGL")] if util.find_library("EGL") else [])
    candidates += ["libEGL.so.1", "libEGL.so"]
    for name in candidates:
        try:
            egl_lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if egl_lib is None:
        return False

    try:
        egl_lib.eglGetProcAddress.restype = c_void_p
        egl_lib.eglGetProcAddress.argtypes = [c_char_p]

        addr = egl_lib.eglGetProcAddress(b"eglQueryDevicesEXT")
        if not addr:
            return False
        query_devices = CFUNCTYPE(
            ctypes.c_uint, c_int, POINTER(c_void_p), POINTER(c_int)
        )(addr)

        # First call (devices=NULL) returns how many devices exist.
        num = c_int(0)
        if not query_devices(0, None, byref(num)) or num.value <= 0:
            return False

        # Second call retrieves the device handles.
        devices = (c_void_p * num.value)()
        returned = c_int(0)
        if not query_devices(num.value, devices, byref(returned)):
            return False

        # Need per-device extension strings to tell hardware from software.
        addr = egl_lib.eglGetProcAddress(b"eglQueryDeviceStringEXT")
        if not addr:
            # Can't classify; assume the enumerated device(s) are usable.
            return returned.value > 0
        query_device_string = CFUNCTYPE(c_char_p, c_void_p, c_int)(addr)

        for i in range(returned.value):
            exts = query_device_string(devices[i], EGL_EXTENSIONS)
            exts = exts.decode("ascii", "ignore") if exts else ""
            if "EGL_MESA_device_software" in exts:
                continue  # software rasterizer (llvmpipe/swrast)
            if "EGL_EXT_device_drm" in exts:
                return True  # real GPU exposing a DRM node
        return False
    except Exception:
        return False


# Respect an explicit user choice; otherwise auto-select the GPU-backed EGL
# backend when a device is available, falling back to the CPU-based osmesa
# backend. This must happen before pyrender is imported.
if "PYOPENGL_PLATFORM" not in os.environ:
    if platform.system() != "Windows":
        os.environ["PYOPENGL_PLATFORM"] = "egl" if _egl_available() else "osmesa"
logger.info("Using OpenGL platform: %s", os.environ.get("PYOPENGL_PLATFORM", "native"))

_RENDERING_IMPORT_ERROR = None
try:
    import pyrender
    import trimesh
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
except ImportError:
    _RENDERING_IMPORT_ERROR = "pyrender and its OpenGL backend could not be loaded"
    logger.warning(
        "Rendering packages not installed. Rendering features are not available"
    )

OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]])

def render_meshes(img, l_mesh, l_face, cam_param, color=None):
    """
    Rendering multiple mesh and project then in the initial image.
    Args:
        - img: np.array [w,h,3]
        - l_mesh: np.array list of [v,3]
        - l_face: np.array list of [f,3]
        - cam_param: info about the camera intrinsics (focal, princpt) and (R,t) is possible
    Return:
        - img: np.array [w,h,3]
    """
    if _RENDERING_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Rendering is unavailable: {_RENDERING_IMPORT_ERROR}. "
            "Install the render dependencies and use a working OpenGL backend."
        )

    # scene
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))

    # mesh
    for i, mesh in enumerate(l_mesh):
        if color is None:
            _color = (np.random.choice(range(1, 225)) / 255,
                      np.random.choice(range(1, 225)) / 255,
                      np.random.choice(range(1, 225)) / 255)
        else:
            if isinstance(color, list):
                _color = color[i]
            elif isinstance(color, tuple):
                _color = color
            else:
                raise NotImplementedError
        mesh = trimesh.Trimesh(mesh, l_face[i],
                               process=True)  # process=True deletes some vertices which prevents per-vertex coloration

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.,
            roughnessFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(_color[0], _color[1], _color[2], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh, f"mesh_{i}")

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera_pose = np.eye(4)
    if 'R' in cam_param.keys():
        camera_pose[:3, :3] = cam_param['R']
    if 't' in cam_param.keys():
        camera_pose[:3, 3] = cam_param['t']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0],
                                       cy=princpt[1])

    # camera
    camera_pose = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ camera_pose
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                          viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=camera_pose)

    # render    
    renderflags = pyrender.RenderFlags.RGBA
    rgb, depth = renderer.render(scene, flags=renderflags)

    # update ccording to fg
    rgb = rgb[:, :, :3].astype(np.float32)
    fg = (depth > 0)[:, :, None].astype(np.float32)

    # Simple smoothing of the mask
    bg_blending_radius = 1
    bg_blending_kernel = 2.0 * torch.ones(
        (1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                 2 * bg_blending_radius + 1) ** 2
    bg_blending_bias = -torch.ones(1)
    fg = fg.reshape((fg.shape[0], fg.shape[1]))
    fg = torch.from_numpy(fg).unsqueeze(0)
    fg = torch.clamp_min(
        torch.nn.functional.conv2d(fg, weight=bg_blending_kernel, bias=bg_blending_bias,
                                   padding=bg_blending_radius) * fg, 0.0)
    fg = fg.permute(1, 2, 0).numpy()

    img = (fg * rgb + (1 - fg) * img).astype(np.uint8)

    renderer.delete()

    return img.astype(np.uint8)
