# DATASET.md — CDnet2014-Format Change Detection Dataset Report

This dataset contains extra training scenes assembled from verified public sources with pre-existing, per-pixel ground-truth foreground masks. It is strictly formatted to match the CDnet2014 directory structure and pixel encoding for training the MSCPNet change-detection model.

---

## 1. Directory Structure and Format Specification

Every scene follows the exact CDnet2014 layout expected by the MSCPNet dataloader:

```
data/extra/<scene_name>/
    input/in%06d.jpg          # 3-channel JPEG, contiguous sequential indexing from 000001
    groundtruth/gt%06d.png    # 8-bit SINGLE-CHANNEL PNG, matching 1:1 index with input
    ROI.bmp                   # Static 8-bit BMP (255 = evaluated region of interest)
```

### Strict Ground-Truth Pixel Encoding
- `0`   = **Background**
- `85`  = **Shadow** (treated as background by the loader)
- `170` = **Unknown / Ignore** (boundary uncertainty, excluded from loss)
- `255` = **Foreground** (moving object)

No other pixel values exist in any ground-truth mask.

---

## 2. Summary of Converted Scenes

| Scene Name | Frame Count | Resolution | Spectrum | Mean FG% | Source Dataset | Licence |
|---|---|---|---|---|---|---|
| `lasiesta_i_bs_01` | 275 | 352×288 | Visible (RGB) | 1.59% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_ca_01` | 350 | 352×288 | Visible (RGB) | 5.27% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_ca_02` | 525 | 352×288 | Visible (RGB) | 7.27% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_il_01` | 300 | 352×288 | Visible (RGB) | 1.75% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_mb_01` | 450 | 352×288 | Visible (RGB) | 6.34% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_oc_01` | 250 | 352×288 | Visible (RGB) | 0.56% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_si_01` | 300 | 352×288 | Visible (RGB) | 4.21% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `lasiesta_i_si_02` | 300 | 352×288 | Visible (RGB) | 1.36% | GTI-UPM LASIESTA | CC BY-SA 4.0 |
| `bmc_111` | 1,499 | 640×480 | Visible (Synth) | 0.44% | BMC 2012 (ACCV) | Academic / Research |
| `bmc_112` | 1,502 | 640×480 | Visible (Synth) | 0.40% | BMC 2012 (ACCV) | Academic / Research |
| `bmc_121` | 1,499 | 640×480 | Visible (Synth) | 0.63% | BMC 2012 (ACCV) | Academic / Research |
| `ino_crossroads` | 12 | 320×240 | Visible / Thermal (Mixed) | 2.55% | INO Video Analytics | Research (Attr: INO) |
| **TOTAL** | **7,262** | | | | | |

---

## 3. Modality & Spectrum Statement

- **Visible-Spectrum (Real-World)**:
  - All 8 `lasiesta_*` scenes (`i_bs_01`, `i_ca_01`, `i_ca_02`, `i_il_01`, `i_mb_01`, `i_oc_01`, `i_si_01`, `i_si_02`) are captured with visible-light optical cameras under realistic indoor conditions (illumination variations, camouflage, bootstrapping, occlusions).
- **Visible-Spectrum (Synthetic Photorealistic)**:
  - All 3 `bmc_*` scenes (`bmc_111`, `bmc_112`, `bmc_121`) are rendered outdoor sequences incorporating dynamic backgrounds (swaying trees, variable lighting, weather effects).
- **Thermal / Multimodal**:
  - `ino_crossroads` originates from the INO platform, providing co-registered visible (RGB) and long-wave infrared (LWIR) modalities. Note that INO ground-truth masks are sparsely provided for key verification frames (DV images), converted here with 1:1 matching frame pairs.

---

## 4. Exact Source-Specific Mask Mappings

To prevent silent corruption, mappings were explicitly defined based on each dataset's ground-truth specification:

### A. LASIESTA
- **Native Format**: 24-bit RGB BMP/PNG with color-coded semantic annotations.
- **Conversion Rule**:
  - `RGB (0, 0, 0)` &rarr; `0` (Background)
  - `RGB (255, 0, 0)` (Object 1) &rarr; `255` (Foreground)
  - `RGB (0, 255, 0)` (Object 2) &rarr; `255` (Foreground)
  - `RGB (255, 255, 0)` (Object 3) &rarr; `255` (Foreground)
  - `RGB (255, 255, 255)` (Static Moving Object) &rarr; `255` (Foreground)
  - `RGB (128, 128, 128)` &rarr; `170` (Uncertainty / Ignore contour)
  - *No shadow class in LASIESTA: pixel value `85` is never emitted.*
  - *ROI: All-255 static mask.*

### B. BMC 2012
- **Native Format**: Grayscale video masks (`_gt.mp4`) with H.264 compression.
- **Conversion Rule**:
  - Pixel intensity $\ge 128$ &rarr; `255` (Foreground)
  - Pixel intensity $< 128$ &rarr; `0` (Background)
  - *No shadow or ignore annotations: values `85` and `170` are never emitted.*
  - *ROI: All-255 static mask.*

### C. INO Video Analytics
- **Native Format**: 24-bit BMP with separate classes for background, shadow, and moving objects.
- **Conversion Rule**:
  - `RGB (0, 0, 0)` &rarr; `0` (Background)
  - `RGB (70, 70, 70)` (within $\pm 15$) &rarr; `85` (Shadow)
  - `RGB (255, 255, 255)` &rarr; `255` (Foreground)
  - *ROI: All-255 static mask.*

---

## 5. Verification Gate Results

All 12 scenes were validated through `scripts/verify.py` against all 5 hard assertions:
1. **1:1 Index Alignment**: Passed for all 7,262 frames. No missing or mismatched indices.
2. **Pixel Value Subset**: Strictly verified $\text{unique}(gt) \subseteq \{0, 85, 170, 255\}$ for every file.
3. **Foreground Ratio Check**: Mean foreground ratios range between $0.40\%$ and $7.27\%$, completely within the required $0.1\%$ to $25.0\%$ range.
4. **ROI Alignment**: Every `ROI.bmp` matches the exact resolution of its scene's input frames.
5. **Non-Trivial Masks**: No scene is all-zero or all-255.

