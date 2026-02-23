"""
shared/utils.py
---------------
Image I/O, ellipse binary I/O, ellipse mask creation, and drawing utilities.
Shared between oval_annotation/ and face_encrypt/.
"""

import struct

import cv2
import numpy as np
from PIL import Image

# ── Binary format ─────────────────────────────────────────────────────────────
# 5 x float32, big-endian → 20 bytes
# Fields: center_x, center_y, major_axis, minor_axis, rotation_angle
_ELLIPSE_FMT = "!5f"

ELLIPSE_KEYS = ("center_x", "center_y", "major_axis", "minor_axis", "rotation_angle")


# ── Ellipse binary I/O ────────────────────────────────────────────────────────

def save_ellipse(path: str, ellipse_params: dict) -> None:
    """Save ellipse parameters as a compact 20-byte binary file."""
    data = struct.pack(
        _ELLIPSE_FMT,
        float(ellipse_params["center_x"]),
        float(ellipse_params["center_y"]),
        float(ellipse_params["major_axis"]),
        float(ellipse_params["minor_axis"]),
        float(ellipse_params["rotation_angle"]),
    )
    with open(path, "wb") as f:
        f.write(data)


def load_ellipse(path: str) -> dict:
    """Load ellipse parameters from a 20-byte binary file."""
    with open(path, "rb") as f:
        vals = struct.unpack(_ELLIPSE_FMT, f.read(20))
    return dict(zip(ELLIPSE_KEYS, vals))


def annotation_path(annotation_dir: str, idx: int) -> str:
    """Return the .bin annotation path for image index idx."""
    return f"{annotation_dir}/{idx:05d}.bin"

def make_ellipse_mask(
    h: int, w: int,
    center_x: float, center_y: float,
    major_axis: float, minor_axis: float,
    rotation_deg: float,
) -> np.ndarray:
    """
    Return a float64 HxW binary mask (0.0 / 1.0) for the rotated ellipse.

    Args:
        h, w: image height and width in pixels.
        center_x, center_y: ellipse centre in pixel coordinates.
        major_axis, minor_axis: full axis lengths in pixels.
        rotation_deg: counter-clockwise rotation angle in degrees.
    """
    a = major_axis / 2.0
    b = minor_axis / 2.0
    y = np.arange(h)
    x = np.arange(w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    x0 = xx - center_x
    y0 = yy - center_y
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = x0 * cos_t + y0 * sin_t
    yr = -x0 * sin_t + y0 * cos_t
    mask = ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0
    return mask.astype(np.float64)

def draw_oval_on_image(
    image: np.ndarray,
    center: tuple,
    axes: tuple,
    angle: float,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a rotated ellipse onto a BGR uint8 image (OpenCV format).

    Args:
        image: HxWx3 uint8 BGR image.
        center: (cx, cy) ellipse centre in pixels.
        axes: (major, minor) full axis lengths in pixels.
        angle: rotation angle in degrees (OpenCV convention).
        color: BGR colour tuple.
        thickness: line thickness (-1 for filled).

    Returns:
        Copy of image with ellipse drawn.
    """
    result = image.copy()
    cv2.ellipse(
        result,
        (int(center[0]), int(center[1])),
        (int(axes[0] / 2), int(axes[1] / 2)),
        angle, 0, 360,
        color, thickness,
    )
    return result
