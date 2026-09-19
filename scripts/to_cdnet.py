#!/usr/bin/env python3
"""
scripts/to_cdnet.py

Converts raw datasets (LASIESTA, BMC, INO) into standard CDnet2014 format:
  data/extra/<scene_name>/
      input/in%06d.jpg          # 3-channel JPEG, sequential from 000001
      groundtruth/gt%06d.png    # 8-bit SINGLE channel PNG, same index
      ROI.bmp                   # Static BMP, >=127 evaluated (<127 excluded)

GT pixel encoding:
    0   = background
    85  = shadow (treated as background by loader)
    170 = unknown / ignore (excluded from loss)
    255 = foreground
"""

import os
import re
import glob
import subprocess
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "extra"))


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def ensure_scene_dirs(scene_dir: str) -> Tuple[str, str]:
    in_dir = os.path.join(scene_dir, "input")
    gt_dir = os.path.join(scene_dir, "groundtruth")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    return in_dir, gt_dir


def create_roi_bmp(shape_hw: Tuple[int, int], output_path: str):
    """
    Creates a static 8-bit grayscale ROI.bmp with all pixels set to 255 (fully evaluated).
    """
    h, w = shape_hw
    roi = np.full((h, w), 255, dtype=np.uint8)
    img = Image.fromarray(roi, mode="L")
    img.save(output_path, format="BMP")


def natural_sort_key(s: str) -> List:
    """Sort strings with embedded numbers naturally (1, 2, ..., 10, 11)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


# ---------------------------------------------------------------------------
# Source 1: LASIESTA Converter
# ---------------------------------------------------------------------------

def map_lasiesta_mask(gt_rgb: np.ndarray) -> np.ndarray:
    """
    Explicit mapping for LASIESTA 24bpp ground truth:
      - (0, 0, 0) -> 0 (background)
      - (255, 0, 0) -> 255 (moving object 1)
      - (0, 255, 0) -> 255 (moving object 2)
      - (255, 255, 0) -> 255 (moving object 3)
      - (255, 255, 255) -> 255 (static moving object)
      - (128, 128, 128) -> 170 (uncertainty)
      - No shadow class in LASIESTA -> never emit 85
    """
    r = gt_rgb[:, :, 0]
    g = gt_rgb[:, :, 1]
    b = gt_rgb[:, :, 2]

    out = np.zeros(r.shape, dtype=np.uint8)

    # Uncertainty / ignore pixels (128, 128, 128)
    is_uncertain = (r == 128) & (g == 128) & (b == 128)
    out[is_uncertain] = 170

    # Moving objects (Red, Green, Yellow, White)
    is_obj1 = (r == 255) & (g == 0) & (b == 0)
    is_obj2 = (r == 0) & (g == 255) & (b == 0)
    is_obj3 = (r == 255) & (g == 255) & (b == 0)
    is_static_obj = (r == 255) & (g == 255) & (b == 255)

    is_foreground = is_obj1 | is_obj2 | is_obj3 | is_static_obj
    out[is_foreground] = 255

    return out


def convert_lasiesta_scene(scene_id: str) -> bool:
    """
    Extracts and converts a single LASIESTA sequence to CDnet format.
    """
    rar_path = os.path.join(RAW_DIR, "lasiesta", f"{scene_id}.rar")
    if not os.path.exists(rar_path):
        print(f"Archive not found: {rar_path}")
        return False

    extract_dir = os.path.join(RAW_DIR, "lasiesta", scene_id)
    # Check if already extracted
    in_sub = os.path.join(extract_dir, scene_id)
    gt_sub = os.path.join(extract_dir, f"{scene_id}-GT")
    if not (os.path.exists(in_sub) and os.path.exists(gt_sub)):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {scene_id}.rar...")
        subprocess.run(["tar", "-xf", rar_path, "-C", extract_dir], check=True)

    # Gather frames and masks
    frame_paths = sorted(
        glob.glob(os.path.join(extract_dir, scene_id, "*.bmp")) +
        glob.glob(os.path.join(extract_dir, scene_id, "*.png")),
        key=natural_sort_key
    )
    gt_paths = sorted(
        glob.glob(os.path.join(extract_dir, f"{scene_id}-GT", "*.png")) +
        glob.glob(os.path.join(extract_dir, f"{scene_id}-GT", "*.bmp")),
        key=natural_sort_key
    )

    if len(frame_paths) == 0 or len(gt_paths) == 0:
        print(f"Error: No frames found in {extract_dir}")
        return False

    if len(frame_paths) != len(gt_paths):
        print(f"Warning: Count mismatch in {scene_id}: {len(frame_paths)} frames vs {len(gt_paths)} GT masks.")
        min_len = min(len(frame_paths), len(gt_paths))
        frame_paths = frame_paths[:min_len]
        gt_paths = gt_paths[:min_len]

    scene_out = os.path.join(OUTPUT_DIR, f"lasiesta_{scene_id.lower()}")
    in_dir, gt_dir = ensure_scene_dirs(scene_out)

    print(f"Converting LASIESTA '{scene_id}' -> {scene_out} ({len(frame_paths)} frames)...")

    frame_shape = None
    for idx, (f_path, g_path) in enumerate(zip(frame_paths, gt_paths), start=1):
        # Read input frame
        img = Image.open(f_path).convert("RGB")
        if frame_shape is None:
            frame_shape = (img.height, img.width)

        in_out_path = os.path.join(in_dir, f"in{idx:06d}.jpg")
        img.save(in_out_path, format="JPEG", quality=95)

        # Read GT
        gt_pil = Image.open(g_path).convert("RGB")
        gt_rgb = np.array(gt_pil)
        mapped_gt = map_lasiesta_mask(gt_rgb)

        gt_out_path = os.path.join(gt_dir, f"gt{idx:06d}.png")
        gt_img = Image.fromarray(mapped_gt, mode="L")
        gt_img.save(gt_out_path, format="PNG")

    # Static ROI
    roi_path = os.path.join(scene_out, "ROI.bmp")
    create_roi_bmp(frame_shape, roi_path)

    print(f"  Done '{scene_id}': {len(frame_paths)} frames converted.")
    return True


# ---------------------------------------------------------------------------
# Source 2: BMC 2012 Converter
# ---------------------------------------------------------------------------

def convert_bmc_scene(video_id: str) -> bool:
    """
    Converts a BMC synthetic video pair (video mp4 + gt mp4) into CDnet format.
    """
    clean_id = video_id.replace("bmc_", "")
    # Search for matching vid and gt
    candidates = [
        (os.path.join(RAW_DIR, "bmc", f"{video_id}.mp4"), os.path.join(RAW_DIR, "bmc", f"{video_id}_gt.mp4")),
        (os.path.join(RAW_DIR, "bmc", f"{clean_id}.mp4"), os.path.join(RAW_DIR, "bmc", f"{clean_id}_gt.mp4")),
        (os.path.join(RAW_DIR, "bmc", "synth1", f"{clean_id}.mp4"), os.path.join(RAW_DIR, "bmc", "synth1", f"{clean_id}_gt.mp4")),
        (os.path.join(RAW_DIR, "bmc", "synth2", f"{clean_id}.mp4"), os.path.join(RAW_DIR, "bmc", "synth2", f"{clean_id}_gt.mp4")),
    ]

    vid_path, gt_path = None, None
    for vp, gp in candidates:
        if os.path.exists(vp) and os.path.exists(gp):
            vid_path, gt_path = vp, gp
            break

    if not vid_path:
        print(f"BMC video pair not found for {video_id} (tried {[c[0] for c in candidates]})")
        return False

    cap_vid = cv2.VideoCapture(vid_path)
    cap_gt = cv2.VideoCapture(gt_path)

    n_vid = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    n_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(n_vid, n_gt)

    scene_out = os.path.join(OUTPUT_DIR, f"{video_id.lower()}")
    in_dir, gt_dir = ensure_scene_dirs(scene_out)

    print(f"Converting BMC '{video_id}' -> {scene_out} ({total_frames} frames)...")

    frame_idx = 1
    frame_shape = None
    while True:
        ret_vid, frame_bgr = cap_vid.read()
        ret_gt, frame_gt = cap_gt.read()
        if not (ret_vid and ret_gt):
            break

        if frame_shape is None:
            frame_shape = (frame_bgr.shape[0], frame_bgr.shape[1])

        # Write 3-channel JPEG frame
        in_out_path = os.path.join(in_dir, f"in{frame_idx:06d}.jpg")
        cv2.imwrite(in_out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Convert GT:
        # Grayscale thresholding: >=128 is foreground (255), <128 is background (0)
        gt_gray = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
        mapped_gt = np.where(gt_gray >= 128, np.uint8(255), np.uint8(0))

        gt_out_path = os.path.join(gt_dir, f"gt{frame_idx:06d}.png")
        cv2.imwrite(gt_out_path, mapped_gt)

        frame_idx += 1

    cap_vid.release()
    cap_gt.release()

    # Static ROI
    roi_path = os.path.join(scene_out, "ROI.bmp")
    create_roi_bmp(frame_shape, roi_path)

    print(f"  Done '{video_id}': {frame_idx - 1} frames converted.")
    return True


# ---------------------------------------------------------------------------
# Source 3: INO Video Analytics Converter
# ---------------------------------------------------------------------------

def map_ino_mask(gt_bgr: np.ndarray) -> np.ndarray:
    """
    Explicit mapping for INO Video Analytics ground truth:
      - (0, 0, 0) -> 0 (background)
      - (70, 70, 70) -> 85 (shadow)
      - (255, 255, 255) -> 255 (foreground)
      - Any tolerance within +/-15 around 70 is mapped to 85.
    """
    b = gt_bgr[:, :, 0]
    g = gt_bgr[:, :, 1]
    r = gt_bgr[:, :, 2]

    out = np.zeros(r.shape, dtype=np.uint8)

    # Shadow: ~70
    is_shadow = (b >= 55) & (b <= 85) & (g >= 55) & (g <= 85) & (r >= 55) & (r <= 85)
    out[is_shadow] = 85

    # Foreground: > 180 (usually 255)
    is_fg = (b > 180) & (g > 180) & (r > 180)
    out[is_fg] = 255

    return out


def convert_ino_scene(scene_id: str) -> bool:
    """
    Converts an INO sequence. Extracts only annotated ground truth frames
    and renumbers them contiguously from 000001.
    """
    zip_path = os.path.join(RAW_DIR, "ino", f"{scene_id}.zip")
    extract_dir = os.path.join(RAW_DIR, "ino", scene_id.replace("ino_", ""))

    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print(f"INO zip not found: {zip_path}")
            return False
        import zipfile
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    # Find AVI and GT folder
    avi_files = glob.glob(os.path.join(extract_dir, "**", "*.avi"), recursive=True)
    gt_bmp_files = sorted(glob.glob(os.path.join(extract_dir, "**", "GT", "*.bmp"), recursive=True),
                          key=natural_sort_key)

    if not avi_files or not gt_bmp_files:
        print(f"Error: Missing AVI or GT files in {extract_dir}")
        return False

    avi_path = avi_files[0]
    cap = cv2.VideoCapture(avi_path)
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scene_out = os.path.join(OUTPUT_DIR, scene_id.lower())
    in_dir, gt_dir = ensure_scene_dirs(scene_out)

    print(f"Converting INO '{scene_id}' -> {scene_out} ({len(gt_bmp_files)} annotated frames)...")

    frame_shape = None
    converted_count = 0

    for idx, gt_file in enumerate(gt_bmp_files, start=1):
        # Extract frame index from filename e.g. Img0201.bmp -> 201
        m = re.search(r'Img(\d+)\.bmp', os.path.basename(gt_file), re.IGNORECASE)
        if not m:
            continue
        vid_frame_no = int(m.group(1))

        # Seek to frame (0-indexed or 1-indexed: video frames usually 0 to N-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, vid_frame_no - 1))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {vid_frame_no} from {avi_path}")
            continue

        if frame_shape is None:
            frame_shape = (frame.shape[0], frame.shape[1])

        # Write input frame
        in_out = os.path.join(in_dir, f"in{idx:06d}.jpg")
        cv2.imwrite(in_out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Process and write GT mask
        gt_bgr = cv2.imread(gt_file)
        mapped_gt = map_ino_mask(gt_bgr)

        gt_out = os.path.join(gt_dir, f"gt{idx:06d}.png")
        cv2.imwrite(gt_out, mapped_gt)

        converted_count += 1

    cap.release()

    if frame_shape:
        roi_path = os.path.join(scene_out, "ROI.bmp")
        create_roi_bmp(frame_shape, roi_path)

    print(f"  Done '{scene_id}': {converted_count} frames converted.")
    return True


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert raw datasets to CDnet2014 format.")
    parser.add_argument("--sources", nargs="+", default=["lasiesta", "bmc", "ino"],
                        help="Sources to convert: lasiesta, bmc, ino")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Specific scenes to convert")
    args = parser.parse_args()

    sources = [s.lower() for s in args.sources]

    if "lasiesta" in sources:
        lasiesta_rars = glob.glob(os.path.join(RAW_DIR, "lasiesta", "*.rar"))
        all_lasiesta = [os.path.splitext(os.path.basename(p))[0] for p in lasiesta_rars]
        scenes = [s for s in (args.scenes or all_lasiesta) if s in all_lasiesta or s.startswith("I_") or s.startswith("O_")]
        for s in scenes:
            convert_lasiesta_scene(s)

    if "bmc" in sources:
        bmc_mp4s = glob.glob(os.path.join(RAW_DIR, "bmc", "**", "*_gt.mp4"), recursive=True)
        all_bmc = [os.path.basename(p).replace("_gt.mp4", "") for p in bmc_mp4s]
        scenes = [s for s in (args.scenes or all_bmc) if s in all_bmc or "bmc" in s or s.isdigit()]
        for s in scenes:
            convert_bmc_scene(s)

    if "ino" in sources:
        ino_zips = glob.glob(os.path.join(RAW_DIR, "ino", "*.zip"))
        all_ino = [os.path.splitext(os.path.basename(p))[0] for p in ino_zips]
        scenes = [s for s in (args.scenes or all_ino) if s in all_ino or "ino" in s]
        for s in scenes:
            convert_ino_scene(s)

    print("\nConversion finished.")


if __name__ == "__main__":
    main()
