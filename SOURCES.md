# SOURCES.md — Dataset Source Discovery Report

> **Goal**: Identify public video datasets (NOT CDnet2014) that ship per-pixel
> foreground masks suitable for conversion to CDnet2014 format for training
> MSCPNet change-detection models. CDnet2014 itself is already available to the user and excluded.

---

## 1. Summary of Candidates Investigated

| Dataset Candidate | Status | Primary Spectrum | Pixel Masks? | Usable for Pipeline? |
|---|---|---|---|---|
| **LASIESTA** | ✅ **Verified Live (200 OK)** | Visible (RGB) | Yes (Color-coded 24-bit BMP) | **YES** (~5,400 frames across static camera scenes) |
| **BMC 2012 (Synthetic)** | ✅ **Verified Live (200 OK)** | Visible (Synthetic) | Yes (Binary MP4 video pairs) | **YES** (20 synthetic video pairs: 10 learning + 10 eval) |
| **INO Video Analytics** | ✅ **Verified Live (200 OK)** | Mixed (RGB / Thermal) | Yes (sparse DV frames: 10–22 per video) | **LIMITED** (~166 frames total across 11 videos) |
| **SBMnet (2016)** | ❌ Excluded | Visible | No (Background images only) | **NO** (Only provides single reconstructed background image) |
| **SBI (2015)** | ❌ Excluded | Visible | No (Background images only) | **NO** (Only provides single reconstructed background image) |
| **OTCBVS Dataset 03 & 09** | ❌ Excluded | Thermal / Visible | No (Bounding boxes only) | **NO** (Bounding box tracks only, no pixel masks) |
| **OTCBVS Dataset 10** | ❌ Excluded | Thermal / Visible | No (Site returns HTTP 403 / SSL expired) | **NO** (Server forbidden/down; only 206 frames historically) |
| **CDnet2014 Categories** | ❌ Excluded | Visible / Thermal | Yes | **EXCLUDED** (User already has CDnet2014) |

---

## 2. Detailed Findings Per Candidate

### ✅ Candidate 1: LASIESTA
- **Full Name**: Labeled and Annotated Sequences for Integral Evaluation of SegmenTation Algorithms
- **Source**: GTI-UPM (Grupo de Tratamiento de Imágenes, Universidad Politécnica de Madrid)
- **Official Page**: https://www.gti.ssr.upm.es/data/LASIESTA
- **Verified Live Download URLs** (All HTTP 200 verified):
  - Base URL pattern: `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/<sequence_id>.rar`
  - Examples:
    - `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/I_SI_01.rar` (35.2 MB)
    - `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/I_SI_02.rar`
    - `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/I_CA_01.rar`
    - `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/I_IL_01.rar`
    - `https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/O_SU_01.rar`
- **Licence**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
  - Citation: C. Cuevas, E. M. Yáñez, and N. García, "Labeled dataset for integral evaluation of moving object detection algorithms: LASIESTA", *Computer Vision and Image Understanding*, vol. 152, pp. 103-117, 2016.
- **Spectrum**: Visible (RGB, 24bpp BMP frames).
- **Frame Count**: ~300 frames per sequence. Available static camera sequences (I_SI_01..02, I_CA_01..02, I_OC_01..02, I_IL_01..02, I_MB_01..02, I_BS_01..02, O_SU_01..02, O_SN_01..02, O_CL_01..02, O_RA_01..02) = **~5,400+ frames**.
  *(Note: Moving-camera sequences I_MC_* and O_MC_* are excluded as camera motion violates static change detection assumptions).*
- **Native Mask Format**: 24-bit BMP in `<id>_GT/` folder:
  - Black `(0, 0, 0)`: Background
  - Red `(255, 0, 0)`: Moving Object 1
  - Green `(0, 255, 0)`: Moving Object 2
  - Yellow `(255, 255, 0)`: Moving Object 3
  - White `(255, 255, 255)`: Moving objects remaining static
  - Gray `(128, 128, 128)`: Uncertainty pixels (along object contours)
- **Target CDnet Encoding Mapping**:
  - `(0, 0, 0)` -> `0` (background)
  - `(255, 0, 0)`, `(0, 255, 0)`, `(255, 255, 0)`, `(255, 255, 255)` -> `255` (foreground)
  - `(128, 128, 128)` -> `170` (unknown / ignore)
  - Shadow class: none present -> never emit `85`.
  - Static ROI: All-255 evaluated.

---

### ✅ Candidate 2: BMC 2012 (Background Models Challenge)
- **Full Name**: Background Models Challenge 2012
- **Source**: Vacavant et al., ACCV 2012 / CVIU 2014
- **Official Page**: https://backgroundmodelschallenge.eu
- **Verified Live Download URLs** (All HTTP 200 verified):
  - Synthetic Learning Archive (10 video pairs): `https://backgroundmodelschallenge.eu/data/bmc_synth1.zip` (715.5 MB)
  - Synthetic Evaluation Archive (10 video pairs): `https://backgroundmodelschallenge.eu/data/bmc_synth2.zip` (962.7 MB)
  - Individual Video Pairs:
    - Video 111: `https://backgroundmodelschallenge.eu/data/synth1/111.mp4` (16.4 MB) + `https://backgroundmodelschallenge.eu/data/synth1/111_gt.mp4` (2.4 MB)
    - Video 112: `https://backgroundmodelschallenge.eu/data/synth1/112.mp4` + `https://backgroundmodelschallenge.eu/data/synth1/112_gt.mp4`
    - Video 121, 122, 211, 212, 221, 222, 311, 312
- **Licence**: Academic / Research use with citation:
  - A. Vacavant, T. Chateau, A. Wilhelm, L. Lequievre, "A benchmark dataset for outdoor foreground/background extraction", BMC/ACCV 2012.
- **Spectrum**: Visible (Synthetic photorealistic rendering: varying weather, wind, illumination).
- **Frame Count**: Full ground truth for every frame in synthetic sets (~300–800 frames per video; ~6,000–12,000 frames total).
- **Native Mask Format**: Video frames with binary mask: Foreground = 255 (white), Background = 0 (black).
- **Target CDnet Encoding Mapping**:
  - `255` -> `255` (foreground)
  - `0` -> `0` (background)
  - No shadow, no uncertainty -> never emit `85` or `170`.
  - Static ROI: All-255 evaluated.

---

### ⚠️ Candidate 3: INO Video Analytics Dataset
- **Full Name**: INO Video Analytics Dataset
- **Source**: Institut National d'Optique (INO), Canada
- **Official Page**: https://www.ino.ca/en/technologies/video-analytics-dataset/videos/
- **Verified Live Download URLs** (Azure Blob Storage, HTTP 200 verified):
  - Crossroads (RGB): `https://inostorage.blob.core.windows.net/media/1546/ino_crossroads.zip` (5.8 MB, 12 GT frames)
  - Close Person (RGB-T): `https://inostorage.blob.core.windows.net/media/1551/ino_closeperson.zip` (7.7 MB, 20 GT frames)
  - Coat Deposit (RGB-T): `https://inostorage.blob.core.windows.net/media/1552/ino_coatdeposit.zip` (9.5 MB, 19 GT frames)
  - Multiple Deposit (RGB-T): `https://inostorage.blob.core.windows.net/media/1554/ino_multipledeposit.zip` (11.5 MB, 15 GT frames)
  - Group Fight (RGB-T): `https://inostorage.blob.core.windows.net/media/1553/ino_groupfight.zip` (9.4 MB, 22 GT frames)
  - Parking Snow (RGB-T): `https://inostorage.blob.core.windows.net/media/1555/ino_parkingsnow.zip` (13.5 MB, 21 GT frames)
  - Backyard Runner (RGB-T): `https://inostorage.blob.core.windows.net/media/1550/ino_backyardrunner.zip` (11.8 MB, 17 GT frames)
- **Licence**: Free for research with attribution ("INO's Video Analytics Dataset").
- **Spectrum**: Mixed Visible (RGB) and Thermal (LWIR 8-12 μm co-registered).
- **Limitation**: Annotations are **sparse**. Only 10 to 22 "DV" (decision validation) ground truth frames are provided per sequence, resulting in only **~166 annotated frame pairs** across all 11 sequences.
- **Target CDnet Encoding Mapping**:
  - Background -> `0`
  - Shadow -> `85`
  - Foreground -> `255`

---

## 3. Discarded Candidates

1. **SBMnet (Scene Background Modeling 2016)**:
   - Evaluated: `http://scenebackgroundmodeling.net/`
   - Reason: SBMnet evaluates background reconstruction methods. Ground truth is a single background image per sequence, NOT per-frame foreground masks.
2. **SBI (Scene Background Initialization 2015)**:
   - Evaluated: `https://sbmi2015.na.icar.cnr.it/SBIdataset.html`
   - Reason: Provides only reconstructed clean background images, not per-frame foreground masks.
3. **OTCBVS Dataset 03 (OSU Color & Thermal) & Dataset 09**:
   - Evaluated: `http://vcipl-okstate.org/pbvs/bench/`
   - Reason: Only bounding-box annotations / tracks are provided. Bounding-box datasets are explicitly prohibited.
4. **OTCBVS Dataset 10 (Bilodeau Pedestrian IR/Visible Stereo)**:
   - Evaluated: `http://vcipl-okstate.org/pbvs/bench/`
   - Reason: Server returns HTTP 403 Forbidden with an expired TLS certificate. Only 206 frames historically. Dropped.
5. **CDnet2014 Additional Categories**:
   - Explicitly excluded per user instructions (already in user's possession).
