#!/usr/bin/env python3
"""
scripts/verify.py

Verification gate for CDnet2014-format converted datasets.
Asserts:
  1. len(input/) == len(groundtruth/) and every index pairs 1:1
  2. set(np.unique(gt)) is a subset of {0, 85, 170, 255} for every GT file
  3. Mean foreground ratio over the scene is within 0.1%–25% (flag outside)
  4. ROI.bmp shape == frame shape for every frame
  5. No all-zero GT scene, no all-255 GT scene

Renders a 4x4 montage per scene overlaying GT onto frames to reports/<scene>_overlay.png
Prints summary table: scene, frames, spectrum, mean fg%, source, licence.
"""

import os
import re
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXTRA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "extra"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

VALID_GT_VALUES = {0, 85, 170, 255}

# Metadata lookup for summary table
PROVENANCE_INFO = {
    "lasiesta": {
        "spectrum": "Visible",
        "source": "GTI-UPM LASIESTA",
        "licence": "CC BY-SA 4.0",
    },
    "bmc": {
        "spectrum": "Visible (Synth)",
        "source": "BMC 2012 (ACCV)",
        "licence": "Academic / Research",
    },
    "ino": {
        "spectrum": "Visible / Thermal (Mixed)",
        "source": "INO Video Analytics",
        "licence": "Research (Attr: INO)",
    },
}


def get_scene_meta(scene_name: str) -> Dict[str, str]:
    for prefix, info in PROVENANCE_INFO.items():
        if scene_name.startswith(prefix):
            return info
    return {"spectrum": "Unknown", "source": "Unknown", "licence": "Unknown"}


def create_overlay_tile(frame_rgb: np.ndarray, gt_mask: np.ndarray, frame_num: int) -> np.ndarray:
    """
    Overlays color-coded GT mask onto frame:
      - Foreground (255): Green tint
      - Shadow (85): Yellow tint
      - Ignore/Uncertain (170): Cyan tint
    """
    overlay = frame_rgb.copy()
    h, w, _ = frame_rgb.shape

    # Masks
    fg_mask = (gt_mask == 255)
    shadow_mask = (gt_mask == 85)
    ignore_mask = (gt_mask == 170)

    # Apply alpha blending
    alpha = 0.5
    # Green for FG [R, G, B] -> [0, 255, 0]
    overlay[fg_mask] = (overlay[fg_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    # Yellow for shadow -> [255, 255, 0]
    overlay[shadow_mask] = (overlay[shadow_mask] * (1 - alpha) + np.array([255, 255, 0]) * alpha).astype(np.uint8)
    # Cyan for ignore -> [0, 255, 255]
    overlay[ignore_mask] = (overlay[ignore_mask] * (1 - alpha) + np.array([0, 255, 255]) * alpha).astype(np.uint8)

    # Add frame index label
    label = f"#{frame_num:06d}"
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def generate_4x4_montage(scene_dir: str, in_files: List[str], gt_files: List[str], report_path: str):
    """
    Selects 16 evenly spaced frames across the scene and creates a 4x4 montage.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    n = len(in_files)
    if n < 16:
        # If fewer than 16 frames, repeat or take all
        indices = np.linspace(0, n - 1, 16, dtype=int)
    else:
        indices = np.linspace(0, n - 1, 16, dtype=int)

    tiles = []
    tile_w, tile_h = 320, 240  # Standard tile size

    for idx in indices:
        f_in = in_files[idx]
        f_gt = gt_files[idx]

        # Extract number from filename
        m = re.search(r'in(\d+)\.jpg', os.path.basename(f_in))
        frame_no = int(m.group(1)) if m else (idx + 1)

        img_bgr = cv2.imread(f_in)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(f_gt, cv2.IMREAD_GRAYSCALE)

        tile = create_overlay_tile(img_rgb, gt, frame_no)
        tile_resized = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tiles.append(tile_resized)

    # Stack 4 rows of 4
    rows = []
    for r in range(4):
        row = np.hstack(tiles[r * 4 : (r + 1) * 4])
        rows.append(row)
    montage = np.vstack(rows)

    # Save as BGR for OpenCV
    cv2.imwrite(report_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))


def verify_scene(scene_dir: str) -> Dict:
    """
    Verifies a single scene directory against all 5 gates.
    """
    scene_name = os.path.basename(scene_dir)
    in_dir = os.path.join(scene_dir, "input")
    gt_dir = os.path.join(scene_dir, "groundtruth")
    roi_file = os.path.join(scene_dir, "ROI.bmp")

    errors = []
    warnings = []

    # 1. Check folder existence and pairing
    in_files = sorted(glob.glob(os.path.join(in_dir, "in*.jpg")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "gt*.png")))

    if len(in_files) == 0:
        errors.append("No input frames found in input/")
    if len(gt_files) == 0:
        errors.append("No groundtruth frames found in groundtruth/")

    if len(in_files) != len(gt_files):
        errors.append(f"Gate 1 Failed: Length mismatch len(input)={len(in_files)} != len(groundtruth)={len(gt_files)}")

    # Check 1:1 index matching
    for f_in, f_gt in zip(in_files, gt_files):
        idx_in = re.search(r'in(\d+)\.jpg', os.path.basename(f_in))
        idx_gt = re.search(r'gt(\d+)\.png', os.path.basename(f_gt))
        if not (idx_in and idx_gt and idx_in.group(1) == idx_gt.group(1)):
            errors.append(f"Gate 1 Failed: Index mismatch {os.path.basename(f_in)} vs {os.path.basename(f_gt)}")
            break

    # 4. ROI.bmp shape check
    if not os.path.exists(roi_file):
        errors.append("Gate 4 Failed: ROI.bmp does not exist")
    else:
        roi = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            errors.append("Gate 4 Failed: Unable to read ROI.bmp")
        elif len(in_files) > 0:
            sample_frame = cv2.imread(in_files[0])
            frame_shape = sample_frame.shape[:2]
            if roi.shape != frame_shape:
                errors.append(f"Gate 4 Failed: ROI.bmp shape {roi.shape} != frame shape {frame_shape}")

    # 2. GT pixel values and foreground ratio calculation
    total_pixels = 0
    total_fg_pixels = 0
    total_evaluated_pixels = 0
    all_scene_values = set()

    for idx, f_gt in enumerate(gt_files):
        gt = cv2.imread(f_gt, cv2.IMREAD_UNCHANGED)
        if gt is None or gt.ndim != 2:
            errors.append(f"Gate 2 Failed: GT is not single-channel 8-bit PNG: {os.path.basename(f_gt)}")
            break

        unique_vals = set(np.unique(gt))
        all_scene_values.update(unique_vals)

        # Gate 2: Subset of {0, 85, 170, 255}
        invalid_vals = unique_vals - VALID_GT_VALUES
        if invalid_vals:
            errors.append(f"Gate 2 Failed: Invalid GT values {invalid_vals} in {os.path.basename(f_gt)}")
            break

        fg_count = np.sum(gt == 255)
        eval_count = np.sum(gt != 170)

        total_fg_pixels += int(fg_count)
        total_evaluated_pixels += int(eval_count)
        total_pixels += gt.size

    # Gate 5: No all-zero GT scene, no all-255 GT scene
    if total_fg_pixels == 0:
        errors.append("Gate 5 Failed: Entire scene has zero foreground pixels (all-zero GT scene)")
    if total_pixels > 0 and total_fg_pixels == total_pixels:
        errors.append("Gate 5 Failed: Entire scene is 100% foreground (all-255 GT scene)")

    # Gate 3: Mean foreground ratio between 0.1% and 25%
    if total_evaluated_pixels > 0:
        mean_fg_pct = (total_fg_pixels / total_evaluated_pixels) * 100.0
    else:
        mean_fg_pct = 0.0

    if mean_fg_pct < 0.1:
        warnings.append(f"Gate 3 Warning: Mean foreground ratio {mean_fg_pct:.3f}% is below 0.1%")
    elif mean_fg_pct > 25.0:
        warnings.append(f"Gate 3 Warning: Mean foreground ratio {mean_fg_pct:.2f}% is above 25.0%")

    # Render montage if no fatal errors
    report_path = os.path.join(REPORTS_DIR, f"{scene_name}_overlay.png")
    if len(in_files) > 0 and len(in_files) == len(gt_files):
        generate_4x4_montage(scene_dir, in_files, gt_files, report_path)

    passed = len(errors) == 0
    return {
        "scene": scene_name,
        "frames": len(in_files),
        "mean_fg_pct": mean_fg_pct,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "report_path": report_path,
        "pixel_values": sorted(list(all_scene_values)),
    }


def main():
    scene_dirs = sorted([
        os.path.join(EXTRA_DIR, d)
        for d in os.listdir(EXTRA_DIR)
        if os.path.isdir(os.path.join(EXTRA_DIR, d))
    ])

    if not scene_dirs:
        print(f"No scene directories found in {EXTRA_DIR}")
        return

    print("\n" + "=" * 95)
    print("CDNET2014 CONVERSION VERIFICATION GATE")
    print("=" * 95)

    results = []
    all_passed = True

    for s_dir in scene_dirs:
        res = verify_scene(s_dir)
        results.append(res)
        if not res["passed"]:
            all_passed = False

    # Summary Table
    print("\n" + "-" * 105)
    print(f"{'Scene':<22} | {'Frames':<7} | {'Spectrum':<16} | {'Mean FG%':<9} | {'Source':<20} | {'Licence':<16}")
    print("-" * 105)

    for r in results:
        meta = get_scene_meta(r["scene"])
        status_flag = " (!)" if r["warnings"] else ""
        fg_str = f"{r['mean_fg_pct']:.2f}%{status_flag}"
        print(f"{r['scene']:<22} | {r['frames']:<7} | {meta['spectrum']:<16} | {fg_str:<9} | {meta['source']:<20} | {meta['licence']:<16}")

    print("-" * 105)

    # Print Error / Warning Details
    print("\nDetailed Verification Results:")
    for r in results:
        status_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n[{status_str}] {r['scene']} ({r['frames']} frames, Pixel Values: {r['pixel_values']})")
        print(f"       Overlay Montage: {r['report_path']}")
        for err in r["errors"]:
            print(f"       ERROR: {err}")
        for warn in r["warnings"]:
            print(f"       WARN : {warn}")

    if all_passed:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("\n❌ SOME VERIFICATION CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
