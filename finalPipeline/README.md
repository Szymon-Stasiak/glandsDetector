# GlandsFinder — README

## Overview
GlandsFinder is a command-line tool that performs tiled object detection with YOLO, merges and deduplicates detections, runs per-detection segmentation with a U-Net, and writes results in multiple formats. The script always writes a final image with segmentation overlays and can optionally write JSON/CSV lists, cropped boxes, and cropped segmentations.

## Features
- YOLO-based tiled detection on large images.
- Per-tile detection coordinate conversion to global coordinates.
- Sorting detections by confidence within tiles.
- Deduplication and ID assignment across tiles (`assign_box_ids`).
- Merging overlapping detections into final bounding boxes.
- Per-detection segmentation using a U-Net model (masks converted to global coordinates).
- Drawing and saving:
  - image with every detection,
  - image with merged detections,
  - final segmented image with overlays (always saved).
- Optional saving of:
  - merged boxes (JSON + CSV),
  - all detected boxes (JSON + CSV),
  - individual merged-box crops as image files,
  - individual segmentation crops with overlays.
- CLI with tile size and overlap control.
- Optional `-o / --output` to select output directory.
- English logging (no emojis).

## Requirements
- Python 3.8+
- Python packages:
  - `ultralytics`
  - `tifffile`
  - `numpy`
  - `opencv-python`
  - `Pillow`
- Local module `unet_core` with `UNET` interface
- YOLO model file and U-Net checkpoint (paths are set in the script; update as needed)

## Default model paths (edit in script if needed)
- YOLO: `../model/saved_models/Glands_Finder_Final_old_n150_best.pt`
- U-Net checkpoint: update the path provided to `modelUNET.set_model(...)` in the script to your actual checkpoint.

## Output behavior
- By default (no `-o`), the script creates the output folder next to the input file:

- <input_folder>/<input_stem>_outputs/
-If `-o / --output` is provided, that folder is used as the root output directory.
- The script always saves the final segmented image.

## Files and folders produced
Always:
- `<basename>_segmented.png` — final image with segmentation overlays (always saved).
- `<basename>_mergedDetection.png` — image annotated with merged boxes (ID + confidence).
- `<basename>_everyDetection.png` — image annotated with every detection (pre-merge).

If `-b` (or `-ba`):
- `<basename>_merged_boxes.json` — merged boxes list (`id, x1, y1, x2, y2, conf, cls`).
- `<basename>_merged_boxes.csv` — same data as CSV.

If `-ba`:
- `<basename>_all_boxes.json` — full list of detections from tiles.
- `<basename>_all_boxes.csv` — same as CSV.

If `-ab`:
- `merged_boxes/box_XXXXX.png` — cropped images for each merged box.

If `-as`:
- `segmentations/seg_XXXXX.png` — cropped images for each segmentation with overlay.

At the end the script prints the output directory path.

## CLI usage
python GlandsFinder.py [options]
### Arguments
- `-f`, `--file`  
  Path to input image (TIFF/PNG/JPG). Optional. If omitted, a default path defined in the script is used.

- `-b`, `--save-merged`  
  Save merged boxes as JSON and CSV files.

- `-ba`, `--save-all`  
  Save merged boxes **and** all detected boxes (both JSON and CSV).

- `-ab`, `--save-merged-boxes-images`  
  Save each merged box as a separate cropped image file in `merged_boxes/`.

- `-as`, `--save-segmentations`  
  Save each segmentation as a separate cropped image with a mask overlay in `segmentations/`.

- `-o`, `--output`  
  Path to the output directory. If not provided, the default output folder next to the input file is used.

- `--tile-size`  
  Integer. Tile size in pixels. Default: `2048`.

- `--overlap`  
  Integer. Overlap in pixels between tiles. Default: `1024`. **Must be given a value** (e.g. `--overlap 512`).

- `--help`  
  Show help text from argparse.

## Examples

Bash / WSL / Linux / macOS:
```bash
python GlandsFinder.py -f ../preprocessedData_v1/tissue_regions/1M01/tissue_region_0.tiff
python GlandsFinder.py -f ../preprocessedData_v1/tissue_regions/1M01/tissue_region_0.tiff -b
python GlandsFinder.py -f ../preprocessedData_v1/tissue_regions/1M01/tissue_region_0.tiff -ba -ab -as --tile-size 1024 --overlap 512 -o ./results/glands_run1
```

## Notes

- Keep `--overlap` and its numeric value on the same command line fragment (no line breaks).
- Flags are boolean; repeating them is redundant.

---

## Troubleshooting

**error: argument --overlap: expected one argument**  
Means `--overlap` was given without a numeric value or a line break split the value. Use `--overlap 512` on the same line.

**Input file does not exist**  
Verify the `-f` path. Use `ls` (Linux/macOS) or `Test-Path` (PowerShell) to confirm.

**Import errors (e.g., unet_core or ultralytics)**  
Install required packages via `pip` or ensure your `PYTHONPATH` includes local modules.

**Model loading errors**  
Verify the YOLO `.pt` file path and U-Net checkpoint path in the script.

**OOM / memory errors**  
Reduce `--tile-size` and/or increase `--overlap` appropriately. Smaller tiles reduce per-inference memory use at the expense of more tiles.

**Empty crops**  
Some boxes at image edges may produce empty crops; the script logs and skips these automatically.

---

## Performance & Tuning

- Tune `--tile-size` and `--overlap` to match available GPU/CPU memory and to balance detection accuracy on tile borders.
- YOLO confidence thresholds are hard-coded where the model is called (e.g., `conf=0.7` and `conf=0.9` in different calls). Modify those calls in the script to change detection sensitivity.
- Deduplication parameters (IoU / overlap / dimension similarity) live in `assign_box_ids`. Adjust `iou_thresh`, `overlap_thresh`, and `dim_thresh` there to change consolidation behavior.

---

## Where to Edit

- **YOLO model path:** replace the path provided to `YOLO(...)`.
- **U-Net checkpoint:** replace the path provided in `modelUNET.set_model(...)`.
- **Default input path:** change `default_image_path` in `main()`.
- **Output naming or structure:** adjust `output_dir` construction in `main()`.
- **Detection thresholds:** edit the `conf` values where `detectionModel(...)` is called.
- **Merge / deduplication behavior:** edit `assign_box_ids()` or `merge_overlapping_boxes()`.

---

## Potential Improvements

- `--conf-thresh` argument to set YOLO confidence threshold from CLI.
- `--out-format` to select PNG vs JPEG or CSV vs JSON only.
- `--log-file` to write logs to a file.
- A `--preview` mode to process only a subset of tiles for fast tests.
