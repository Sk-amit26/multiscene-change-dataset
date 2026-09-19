# CDnet2014-Format Change Detection Training Set

This repository contains the complete conversion pipeline, verification suite, and specification for an extra training dataset formatted strictly according to the **CDnet2014** standard for direct consumption by the **MSCPNet** (multi-scale contrast-preserving encoder-decoder) training pipeline.

All scenes are assembled from verified public video sources that provide pre-existing, per-pixel ground-truth foreground masks.

---

## 📦 Dataset Access on Kaggle

The pre-processed, verified dataset is hosted on Kaggle:

🔗 **Kaggle Dataset Link:** [**Kaggle Dataset (Private / Accessible via Shared Link)**](https://www.kaggle.com/datasets) *(Update with your exact link)*

### Downloading via Kaggle CLI
To download and extract the dataset directly into this repository:
```bash
# 1. Ensure your kaggle.json is placed in ~/.kaggle/kaggle.json
# 2. Download the dataset zip
kaggle datasets download -d <your-kaggle-username>/<dataset-name> -p data/extra --unzip
```

---

## 1. Dataset Overview

*   **Total Converted & Verified Frames**: **7,262 frames** across **12 diverse scenes**
*   **Verification Status**: **100% PASS** on all 5 gates in `scripts/verify.py`

| Scene Name | Frames | Resolution | Modality | Mean FG% | Source Dataset | Licence |
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
| `ino_crossroads` | 12 | 320×240 | Visible / Thermal | 2.55% | INO Video Analytics | Research (Attr: INO) |

---

## 2. Output Format Specification

The dataset is structured under `data/extra/<scene_name>/`:

```
data/extra/<scene_name>/
    input/in%06d.jpg          # 3-channel RGB JPEG, sequential from 000001
    groundtruth/gt%06d.png    # 8-bit SINGLE-CHANNEL PNG, 1:1 matching index with input
    ROI.bmp                   # Static 8-bit BMP (255 = evaluated, <127 = excluded)
```

### Exact Pixel Encodings
*   `0` = **Background**
*   `85` = **Shadow** *(treated as background by the loader)*
*   `170` = **Unknown / Ignore** *(contour uncertainty, excluded from loss computation)*
*   `255` = **Foreground** *(moving object)*

---

## 3. Repository Structure

```
.
├── reports/                   # 4x4 visual overlay montages for quality inspection
│   ├── lasiesta_i_si_01_overlay.png
│   ├── bmc_111_overlay.png
│   └── ...
├── scripts/
│   ├── fetch.py               # Download pipeline with cache and size checks
│   ├── to_cdnet.py            # Dataset converter with explicit mask mappings
│   └── verify.py              # 5-gate verification suite and montage renderer
├── DATASET.md                 # Detailed report with provenance and exact mask mappings
├── SOURCES.md                 # Discovery report with verified live download links
├── dataset_index.json         # Machine-readable JSON manifest of all scenes
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 4. Quickstart & Pipeline Usage

### Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\pip install -r requirements.txt
# On Linux/macOS:
source .venv/bin/activate && pip install -r requirements.txt
```

### 1. Download Datasets from Source
```bash
python scripts/fetch.py --sources lasiesta bmc ino
```

### 2. Convert to CDnet2014 Format
```bash
python scripts/to_cdnet.py --sources lasiesta bmc ino
```

### 3. Run Verification Gate & Generate Overlays
```bash
python scripts/verify.py
```

---

## 5. PyTorch DataLoader Example

```python
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CDnetExtraDataset(Dataset):
    def __init__(self, scene_dir, transform=None):
        self.scene_dir = scene_dir
        self.transform = transform
        self.in_files = sorted(glob.glob(os.path.join(scene_dir, "input", "in*.jpg")))
        self.gt_files = sorted(glob.glob(os.path.join(scene_dir, "groundtruth", "gt*.png")))
        self.roi = np.array(Image.open(os.path.join(scene_dir, "ROI.bmp")).convert("L"))

        assert len(self.in_files) == len(self.gt_files), "Mismatch between frames and masks"

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        # 3-channel RGB image
        frame = Image.open(self.in_files[idx]).convert("RGB")
        # 1-channel mask (0=bg, 85=shadow, 170=ignore, 255=fg)
        gt = np.array(Image.open(self.gt_files[idx]))

        # Binary label target: 255 -> 1 (fg), 0 or 85 -> 0 (bg), 170 -> 255 (ignore index)
        target = np.zeros_like(gt, dtype=np.int64)
        target[gt == 255] = 1
        target[gt == 170] = 255  # Exclude from loss via ignore_index=255

        if self.transform:
            frame = self.transform(frame)

        return frame, torch.from_numpy(target)
```
