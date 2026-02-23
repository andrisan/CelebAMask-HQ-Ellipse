"""
oval_annotation/extractor.py
-----------------------------
Pure utility functions for fitting a rotated ellipse to a CelebAMask-HQ skin mask.
No I/O side-effects; worker logic lives in workers.py.
"""

import os

import cv2
import numpy as np


def build_face_mask(
    mask_folder: str,
    image_id: str,
    target_size: tuple = None,
) -> np.ndarray:
    """
    Load the skin-segmentation mask for one image.

    Args:
        mask_folder: Directory containing ``{image_id}_skin.png``.
        image_id:    Zero-padded 5-digit string, e.g. ``"00030"``.
        target_size: Optional ``(width, height)`` to resize the mask to.

    Returns:
        Grayscale uint8 mask array (HxW).

    Raises:
        FileNotFoundError: If the skin mask does not exist.
    """
    path = os.path.join(mask_folder, f"{image_id}_skin.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skin mask not found: {path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask: {path}")

    if target_size is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return mask


def fit_ellipse(mask: np.ndarray) -> tuple:
    """
    Fit a rotated ellipse to the largest contour in a binary mask.

    Args:
        mask: Grayscale uint8 mask (non-zero pixels = face region).

    Returns:
        ``(center_x, center_y, major_axis, minor_axis, rotation_angle)``
        where axes are full lengths and angle is in degrees.

    Raises:
        ValueError: If no usable contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask.")

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        raise ValueError("Not enough contour points to fit an ellipse (need ≥ 5).")

    ellipse = cv2.fitEllipse(largest)
    (cx, cy), (major, minor), angle = ellipse
    return float(cx), float(cy), float(major), float(minor), float(angle)
