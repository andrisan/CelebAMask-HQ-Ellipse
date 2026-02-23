"""
oval_annotation/workers.py
---------------------------
Top-level picklable worker functions consumed by shared.parallel.run_parallel.
Each worker takes a single argument tuple and returns (idx, result | None).
"""

import os
import sys

# Ensure the repo root is on sys.path so 'shared' is importable
# (required when workers are spawned as subprocesses).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np

from utils import save_ellipse, draw_oval_on_image, make_ellipse_mask, annotation_path
from extractor import build_face_mask, fit_ellipse


def extract_worker(item: tuple):
    """
    Extract and save the oval ellipse annotation for one image.

    Args:
        item: ``(idx, image_dir, mask_dir, output_dir,
                  images_per_folder, save_mask, save_overlay)``

    Returns:
        ``(idx, info_dict)`` on success, or ``(idx, None)`` on failure.
        ``info_dict`` contains ellipse parameters and pixel count.
    """
    idx, image_dir, mask_dir, output_dir, images_per_folder, save_mask, save_overlay = item

    image_id = f"{idx:05d}"
    mask_folder = os.path.join(mask_dir, str(idx // images_per_folder))
    img_path = os.path.join(image_dir, f"{idx}.jpg")

    # Load original image to get dimensions
    img = cv2.imread(img_path)
    if img is None:
        return idx, None

    img_h, img_w = img.shape[:2]

    try:
        mask = build_face_mask(mask_folder, image_id, target_size=(img_w, img_h))
        cx, cy, major, minor, angle = fit_ellipse(mask)
    except (FileNotFoundError, ValueError, IOError):
        return idx, None

    ellipse_params = {
        "center_x":      cx,
        "center_y":      cy,
        "major_axis":    major,
        "minor_axis":    minor,
        "rotation_angle": angle,
    }

    # Save 20-byte binary annotation
    bin_path = annotation_path(output_dir, idx)
    save_ellipse(bin_path, ellipse_params)

    # Count pixels inside ellipse
    ellipse_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask,
        (int(round(cx)), int(round(cy))),
        (int(round(major / 2)), int(round(minor / 2))),
        angle, 0, 360, 255, -1,
    )
    num_pixels = int(np.sum(ellipse_mask > 0))

    # Optional: save binary oval mask PNG
    if save_mask:
        mask_dir_out = os.path.join(output_dir, "oval_masks")
        os.makedirs(mask_dir_out, exist_ok=True)
        cv2.imwrite(os.path.join(mask_dir_out, f"{image_id}_oval_mask.png"), ellipse_mask)

    # Optional: save overlay image
    if save_overlay:
        overlay_dir = os.path.join(output_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        overlay = draw_oval_on_image(
            img,
            center=(cx, cy),
            axes=(major, minor),
            angle=angle,
            color=(0, 255, 0),
            thickness=3,
        )
        cv2.imwrite(os.path.join(overlay_dir, f"{image_id}_overlay.jpg"), overlay)

    info = {**ellipse_params, "num_pixels": num_pixels}
    return idx, info
