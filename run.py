#!/usr/bin/env python3
"""
-----------------------
CLI entry point for the oval extraction pipeline.

Usage
-----
    python run.py [--config CONFIG] [--workers N] extract [options]

The YAML config file provides defaults for all parameters.
Any CLI flag overrides the corresponding YAML value.
"""

import argparse
import json
import os
import sys

import yaml

# Ensure repo root is on path so 'shared' is importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Ensure this folder's modules are importable (workers, extractor).
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from parallel import run_parallel
from workers import extract_worker


# ── Config helpers ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yaml")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_set(d: dict, keys: list, value):
    """Set a nested key only when value is not None."""
    if value is None:
        return
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ── Subcommand: extract ────────────────────────────────────────────────────────

def cmd_extract(cfg: dict) -> None:
    paths = cfg["paths"]
    proc = cfg["processing"]
    dataset = cfg["dataset"]

    image_dir = paths["image_dir"]
    mask_dir = paths["mask_dir"]
    output_dir = paths["output_dir"]
    images_per_folder = dataset["images_per_folder"]
    start = proc["start"]
    end = proc["end"]
    save_masks = proc["save_masks"]
    save_overlays = proc["save_overlays"]
    n_workers = cfg.get("workers")

    os.makedirs(output_dir, exist_ok=True)

    # Build work list
    work = []
    for idx in range(start, end):
        work.append((
            idx, image_dir, mask_dir, output_dir,
            images_per_folder, save_masks, save_overlays,
        ))

    print(f"Extracting ovals: images {start}–{end-1} "
          f"({len(work)} images, workers={n_workers or 'all CPUs'})")
    print(f"Output dir: {output_dir}")

    results, failed = run_parallel(
        work_items=work,
        worker_fn=extract_worker,
        n_workers=n_workers,
        desc="Extracting ovals",
    )

    # Summary
    print(f"\n{'='*55}")
    print(f"COMPLETE — {len(results)} succeeded, {len(failed)} failed")
    print(f"{'='*55}")

    if failed:
        failed_indices = [item[0] for item in failed]
        fail_log = os.path.join(output_dir, "failed.json")
        with open(fail_log, "w") as f:
            json.dump(failed_indices, f)
        print(f"Failed indices saved to: {fail_log}")
        if len(failed_indices) <= 10:
            print(f"Failed: {failed_indices}")
        else:
            print(f"First 10 failed: {failed_indices[:10]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Oval annotation extraction for CelebAMask-HQ",
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG,
        metavar="CONFIG",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of parallel workers (overrides config; default: all CPUs)",
    )

    sub = parser.add_subparsers(dest="subcommand", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Extract oval annotations from skin masks")
    p_extract.add_argument("--start", type=int, default=None, metavar="N",
                           help="First image index (inclusive)")
    p_extract.add_argument("--end", type=int, default=None, metavar="N",
                           help="Last image index (exclusive)")
    p_extract.add_argument("--save-masks", action="store_true", default=None,
                           help="Save binary oval mask PNGs")
    p_extract.add_argument("--save-overlays", action="store_true", default=None,
                           help="Save original images with oval drawn on them")
    p_extract.add_argument("--image-dir", default=None, metavar="DIR",
                           help="Override paths.image_dir")
    p_extract.add_argument("--mask-dir", default=None, metavar="DIR",
                           help="Override paths.mask_dir")
    p_extract.add_argument("--output-dir", default=None, metavar="DIR",
                           help="Override paths.output_dir")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply global overrides
    if args.workers is not None:
        cfg["workers"] = args.workers

    if args.subcommand == "extract":
        deep_set(cfg, ["processing", "start"], args.start)
        deep_set(cfg, ["processing", "end"], args.end)
        deep_set(cfg, ["paths", "image_dir"], args.image_dir)
        deep_set(cfg, ["paths", "mask_dir"], args.mask_dir)
        deep_set(cfg, ["paths", "output_dir"], args.output_dir)
        if args.save_masks:
            cfg["processing"]["save_masks"] = True
        if args.save_overlays:
            cfg["processing"]["save_overlays"] = True

        cmd_extract(cfg)


if __name__ == "__main__":
    main()
