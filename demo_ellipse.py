"""
demo_ellipse.py
---------------
Demo of make_ellipse_mask and draw_oval_on_image using a real image and
its precomputed ellipse annotation from the output directory in config.yaml.

Usage
-----
    python demo_ellipse.py [--idx N] [--config CONFIG]

If the annotation for the requested index does not exist yet, the script
will tell you how to generate it first.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

from utils import load_ellipse, make_ellipse_mask, draw_oval_on_image, annotation_path


_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: make_ellipse_mask + draw_oval_on_image")
    parser.add_argument("--idx", type=int, default=0, metavar="N",
                        help="Image index to visualise (default: 0)")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, metavar="CONFIG",
                        help="Path to YAML config file (default: config.yaml)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    image_dir  = cfg["paths"]["image_dir"]
    output_dir = cfg["paths"]["output_dir"]
    idx        = args.idx

    # ── Check annotation ──────────────────────────────────────────────────────
    ann_path = annotation_path(output_dir, idx)
    if not os.path.exists(ann_path):
        print(f"Annotation not found: {ann_path}")
        print()
        print("Please run the extraction pipeline first:")
        print(f"  python run.py --config {args.config} extract --start {idx} --end {idx + 1}")
        sys.exit(1)

    # ── Load image ────────────────────────────────────────────────────────────
    img_path = os.path.join(image_dir, f"{idx}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    img_h, img_w = img.shape[:2]

    # ── Load ellipse annotation ───────────────────────────────────────────────
    e = load_ellipse(ann_path)
    cx    = e["center_x"]
    cy    = e["center_y"]
    major = e["major_axis"]
    minor = e["minor_axis"]
    angle = e["rotation_angle"]

    print(f"Image : {img_path}  ({img_w}x{img_h})")
    print(f"Annotation: cx={cx:.1f}, cy={cy:.1f}, "
          f"major={major:.1f}, minor={minor:.1f}, angle={angle:.1f}°")

    # ── make_ellipse_mask ─────────────────────────────────────────────────────
    mask = make_ellipse_mask(img_h, img_w, cx, cy, major, minor, angle)

    # Dim region outside the ellipse for a cleaner visualisation
    preview = img.copy().astype(float)
    preview[mask == 0] *= 0.3
    preview = preview.astype("uint8")

    # ── draw_oval_on_image ────────────────────────────────────────────────────
    result = draw_oval_on_image(
        preview,
        center=(cx, cy),
        axes=(major, minor),
        angle=angle,
        color=(0, 255, 0),
        thickness=3,
    )

    # ── Display ───────────────────────────────────────────────────────────────
    mask_display = cv2.cvtColor((mask * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([mask_display, result])

    cv2.imshow(f"[{idx}] mask | overlay", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
