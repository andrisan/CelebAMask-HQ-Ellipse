# CelebAMask-HQ-Ellipse

Compact ellipse-based facial ROI annotations derived from CelebAMask-HQ skin masks for JPEG-compatible face obfuscation research.

## Overview

This tool processes the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset and fits a rotated ellipse to the skin-segmentation mask of each face image. The result is a compact 20-byte binary annotation per image encoding the ellipse parameters, suitable for use in face obfuscation, ellipse-based face detection, or privacy-preserving image pipelines.

**Pipeline:**

1. Load the `{image_id}_skin.png` mask from CelebAMask-HQ mask annotations
2. Find the largest contour in the skin mask
3. Fit a rotated ellipse to that contour via OpenCV's `fitEllipse`
4. Save the 5 ellipse parameters as a packed binary file (5 x `float32` = 20 bytes)
5. Optionally save a binary oval mask PNG and/or an overlay image for visual inspection

## Pre-built Annotations

If you just need the annotations and don't want to run the extraction pipeline yourself, a pre-built archive is provided:

**`pre-built-annotation.zip`** — a flat ZIP containing one 20-byte `.bin` file for every image in CelebAMask-HQ (30,000 files, ~600 KB uncompressed):

```
pre-built-annotation.zip
├── 00000.bin
├── 00001.bin
├── …
└── 29999.bin
```

Extract it directly into the directory you want to use as your annotation source:

```bash
unzip pre-built-annotation.zip -d /your/output/dir/
```

Each file follows the binary format described in the [Output Format](#output-format) section below.

---

## Dataset

This tool requires a local copy of the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset with the following structure:

```
CelebAMask-HQ/
├── CelebA-HQ-img/          # 30,000 JPEG images (0.jpg … 29999.jpg)
└── CelebAMask-HQ-mask-anno/
    ├── 0/                   # Masks for images 0–1999
    │   ├── 00000_skin.png
    │   ├── 00001_skin.png
    │   └── …
    ├── 1/                   # Masks for images 2000–3999
    └── …                   # (15 folders total, 2000 images each)
```

## Requirements

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- NumPy
- PyYAML

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy the example config and edit it to match your local dataset paths:

```bash
cp copy.yaml.example config.yaml
```

**`config.yaml`:**

```yaml
workers: null              # null → os.cpu_count(); applies to all parallel operations

paths:
  image_dir:  /FolderPath/CelebAMask-HQ/CelebA-HQ-img
  mask_dir:   /FolderPath/CelebAMask-HQ-mask-anno
  output_dir: /FolderPath/CelebAMask-HQ/oval-annotation

dataset:
  total_images:       30000
  images_per_folder:  2000

processing:
  start:          0
  end:            30000
  save_masks:     false
  save_overlays:  false
```

| Key | Description |
|-----|-------------|
| `workers` | Number of parallel processes. `null` uses all CPU cores. |
| `paths.image_dir` | Directory containing the JPEG face images. |
| `paths.mask_dir` | Root directory of CelebAMask-HQ mask annotations. |
| `paths.output_dir` | Where to write binary annotation files (and optional masks/overlays). |
| `dataset.images_per_folder` | Number of images per mask subfolder (2000 for standard CelebAMask-HQ). |
| `processing.start` / `end` | Image index range to process (start inclusive, end exclusive). |
| `processing.save_masks` | If `true`, saves a binary oval mask PNG per image. |
| `processing.save_overlays` | If `true`, saves an overlay JPEG with the ellipse drawn on the original image. |

## Usage

```bash
python run.py [--config CONFIG] [--workers N] extract [options]
```

**Examples:**

```bash
# Run with default config.yaml (all 30,000 images, all CPU cores)
python run.py extract

# Process a subset of images with 8 workers
python run.py extract --start 0 --end 1000 --workers 8

# Save visual masks and overlays for inspection
python run.py extract --start 0 --end 100 --save-masks --save-overlays

# Use a custom config file
python run.py --config /path/to/my_config.yaml extract

# Override paths on the command line
python run.py extract --image-dir /data/imgs --mask-dir /data/masks --output-dir /data/out
```

**All CLI flags:**

| Flag | Description |
|------|-------------|
| `--config CONFIG` | Path to YAML config file (default: `config.yaml`) |
| `--workers N` | Number of parallel workers (overrides config) |
| `--start N` | First image index, inclusive (overrides config) |
| `--end N` | Last image index, exclusive (overrides config) |
| `--image-dir DIR` | Override `paths.image_dir` |
| `--mask-dir DIR` | Override `paths.mask_dir` |
| `--output-dir DIR` | Override `paths.output_dir` |
| `--save-masks` | Save binary oval mask PNGs |
| `--save-overlays` | Save overlay JPEGs with ellipse drawn |

## Demo

`demo_ellipse.py` visualises the ellipse mask and overlay for a single image using the paths in `config.yaml`. It requires annotations to have been extracted (or unzipped from `pre-built-annotation.zip`) first.

```bash
# Visualise image 0 (default)
python demo_ellipse.py

# Visualise a specific image index
python demo_ellipse.py --idx 42

# Use a custom config file
python demo_ellipse.py --config /path/to/my_config.yaml --idx 100
```

The window shows the binary ellipse mask and the original image with the ellipse drawn on it, side by side.

---

## Output Format

### Binary annotation files

Each image produces a 20-byte binary file containing five `float32` values in native byte order:

| Offset | Field | Description |
|--------|-------|-------------|
| 0–3 | `center_x` | Ellipse center X coordinate (pixels) |
| 4–7 | `center_y` | Ellipse center Y coordinate (pixels) |
| 8–11 | `major_axis` | Full length of the major axis (pixels) |
| 12–15 | `minor_axis` | Full length of the minor axis (pixels) |
| 16–19 | `rotation_angle` | Ellipse rotation angle (degrees) |

### Output directory structure

```
output_dir/
├── 00000.bin          # Binary ellipse annotation for image 0
├── 00001.bin
├── …
├── oval_masks/        # (optional, --save-masks)
│   ├── 00000_oval_mask.png
│   └── …
├── overlays/          # (optional, --save-overlays)
│   ├── 00000_overlay.jpg
│   └── …
└── failed.json        # List of image indices that failed (if any)
```

## License

MIT License — Copyright (c) 2026 Andri Santoso. See [LICENSE](LICENSE) for details.
