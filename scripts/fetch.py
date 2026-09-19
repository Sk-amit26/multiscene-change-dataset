#!/usr/bin/env python3
"""
scripts/fetch.py

Downloads confirmed public datasets with pixel-level foreground masks.
Requirements:
  - Resumable / range-aware downloads
  - Checksum / file size verification
  - Local caching in data/raw/ so re-runs do not re-download
  - Prints total bytes before starting
"""

import os
import sys
import hashlib
import requests
from typing import List, Dict, Optional, Tuple

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

# ---------------------------------------------------------------------------
# Confirmed Sources Catalogs
# ---------------------------------------------------------------------------

LASIESTA_BASE_URL = "https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA"

# Static camera sequences from LASIESTA
# (Excludes moving camera I_MC_* and O_MC_*)
LASIESTA_CATALOG = {
    # Indoor simple
    "I_SI_01": {"url": f"{LASIESTA_BASE_URL}/I_SI_01.rar", "size": 35209434},
    "I_SI_02": {"url": f"{LASIESTA_BASE_URL}/I_SI_02.rar", "size": 35073169},
    # Indoor camouflage
    "I_CA_01": {"url": f"{LASIESTA_BASE_URL}/I_CA_01.rar", "size": 36173004},
    "I_CA_02": {"url": f"{LASIESTA_BASE_URL}/I_CA_02.rar", "size": 35773177},
    # Indoor occlusions
    "I_OC_01": {"url": f"{LASIESTA_BASE_URL}/I_OC_01.rar", "size": 35327244},
    "I_OC_02": {"url": f"{LASIESTA_BASE_URL}/I_OC_02.rar", "size": 35057630},
    # Indoor illumination changes
    "I_IL_01": {"url": f"{LASIESTA_BASE_URL}/I_IL_01.rar", "size": 35140306},
    "I_IL_02": {"url": f"{LASIESTA_BASE_URL}/I_IL_02.rar", "size": 35246714},
    # Indoor modified background
    "I_MB_01": {"url": f"{LASIESTA_BASE_URL}/I_MB_01.rar", "size": 35282711},
    "I_MB_02": {"url": f"{LASIESTA_BASE_URL}/I_MB_02.rar", "size": 35123927},
    # Indoor bootstrapping
    "I_BS_01": {"url": f"{LASIESTA_BASE_URL}/I_BS_01.rar", "size": 35133610},
    "I_BS_02": {"url": f"{LASIESTA_BASE_URL}/I_BS_02.rar", "size": 35150937},
    # Outdoor sunny
    "O_SU_01": {"url": f"{LASIESTA_BASE_URL}/O_SU_01.rar", "size": 35174092},
    "O_SU_02": {"url": f"{LASIESTA_BASE_URL}/O_SU_02.rar", "size": 35154366},
    # Outdoor cloudy
    "O_CL_01": {"url": f"{LASIESTA_BASE_URL}/O_CL_01.rar", "size": 35237882},
    "O_CL_02": {"url": f"{LASIESTA_BASE_URL}/O_CL_02.rar", "size": 35212571},
    # Outdoor rain
    "O_RA_01": {"url": f"{LASIESTA_BASE_URL}/O_RA_01.rar", "size": 35372421},
    "O_RA_02": {"url": f"{LASIESTA_BASE_URL}/O_RA_02.rar", "size": 35235891},
    # Outdoor snow
    "O_SN_01": {"url": f"{LASIESTA_BASE_URL}/O_SN_01.rar", "size": 35142103},
    "O_SN_02": {"url": f"{LASIESTA_BASE_URL}/O_SN_02.rar", "size": 35182967},
}

BMC_BASE_URL = "https://backgroundmodelschallenge.eu/data"

BMC_CATALOG = {
    # Synth1 learning sequences (individual mp4 pairs)
    "bmc_111": {
        "vid_url": f"{BMC_BASE_URL}/synth1/111.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/111_gt.mp4",
        "vid_size": 16394727,
        "gt_size": 2443062,
    },
    "bmc_112": {
        "vid_url": f"{BMC_BASE_URL}/synth1/112.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/112_gt.mp4",
        "vid_size": 16568285,
        "gt_size": 2187313,
    },
    "bmc_121": {
        "vid_url": f"{BMC_BASE_URL}/synth1/121.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/121_gt.mp4",
        "vid_size": 17855011,
        "gt_size": 3137953,
    },
    "bmc_122": {
        "vid_url": f"{BMC_BASE_URL}/synth1/122.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/122_gt.mp4",
        "vid_size": 17478051,
        "gt_size": 2697868,
    },
    "bmc_211": {
        "vid_url": f"{BMC_BASE_URL}/synth1/211.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/211_gt.mp4",
        "vid_size": 16641865,
        "gt_size": 2525547,
    },
    "bmc_212": {
        "vid_url": f"{BMC_BASE_URL}/synth1/212.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/212_gt.mp4",
        "vid_size": 16405786,
        "gt_size": 2244249,
    },
    "bmc_221": {
        "vid_url": f"{BMC_BASE_URL}/synth1/221.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/221_gt.mp4",
        "vid_size": 18567554,
        "gt_size": 3266938,
    },
    "bmc_222": {
        "vid_url": f"{BMC_BASE_URL}/synth1/222.mp4",
        "gt_url": f"{BMC_BASE_URL}/synth1/222_gt.mp4",
        "vid_size": 17897275,
        "gt_size": 2727182,
    },
}

INO_BASE_URL = "https://inostorage.blob.core.windows.net/media"

INO_CATALOG = {
    "ino_crossroads": {
        "url": f"{INO_BASE_URL}/1546/ino_crossroads.zip",
        "size": 5765457,
        "type": "RGB",
    },
    "ino_closeperson": {
        "url": f"{INO_BASE_URL}/1551/ino_closeperson.zip",
        "size": 7718863,
        "type": "RGB-T",
    },
    "ino_coatdeposit": {
        "url": f"{INO_BASE_URL}/1552/ino_coatdeposit.zip",
        "size": 9976785,
        "type": "RGB-T",
    },
    "ino_groupfight": {
        "url": f"{INO_BASE_URL}/1553/ino_groupfight.zip",
        "size": 9866970,
        "type": "RGB-T",
    },
}


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------

def calculate_md5(filepath: str, block_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    dst_path: str,
    expected_size: Optional[int] = None,
    expected_md5: Optional[str] = None,
    timeout: int = 30,
) -> bool:
    """
    Downloads a URL to dst_path with resumable range-header support and verification.
    Returns True if downloaded or verified from cache.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    temp_path = dst_path + ".part"

    # Check cache
    if os.path.exists(dst_path):
        current_size = os.path.getsize(dst_path)
        if expected_size is None or current_size == expected_size:
            if expected_md5:
                actual_md5 = calculate_md5(dst_path)
                if actual_md5 == expected_md5:
                    print(f"  [Cache hit] {os.path.basename(dst_path)} ({current_size:,} bytes, MD5 verified)")
                    return True
                else:
                    print(f"  [MD5 mismatch] {os.path.basename(dst_path)} - re-downloading...")
            else:
                print(f"  [Cache hit] {os.path.basename(dst_path)} ({current_size:,} bytes)")
                return True

    # Resume support: check if .part exists
    existing_bytes = 0
    if os.path.exists(temp_path):
        existing_bytes = os.path.getsize(temp_path)

    headers = {"User-Agent": "AntigravityDatasetDownloader/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    session = requests.Session()
    try:
        response = session.get(url, headers=headers, stream=True, timeout=timeout)
        mode = "wb"

        # Check server response
        if response.status_code == 206:
            # Partial content resumed
            mode = "ab"
            total_size = int(response.headers.get("Content-Range", "").split("/")[-1] or 0)
        elif response.status_code == 200:
            existing_bytes = 0
            mode = "wb"
            total_size = int(response.headers.get("Content-Length", 0))
        elif response.status_code == 416:
            # Range unsatisfiable, file might already be complete
            if expected_size and existing_bytes == expected_size:
                os.replace(temp_path, dst_path)
                return True
            mode = "wb"
            existing_bytes = 0
            response = session.get(url, stream=True, timeout=timeout)
            total_size = int(response.headers.get("Content-Length", 0))
        else:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

        downloaded = existing_bytes
        chunk_size = 128 * 1024  # 128 KB
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        # Rename part to final
        os.replace(temp_path, dst_path)

        # Verification
        final_size = os.path.getsize(dst_path)
        if expected_size and final_size != expected_size:
            print(f"  [WARNING] Size mismatch for {os.path.basename(dst_path)}: got {final_size}, expected {expected_size}")
        else:
            print(f"  [Downloaded] {os.path.basename(dst_path)} ({final_size:,} bytes)")

        if expected_md5:
            actual_md5 = calculate_md5(dst_path)
            if actual_md5 != expected_md5:
                raise ValueError(f"MD5 checksum failed for {dst_path}")
            print(f"  [Checksum] MD5 verified: {actual_md5}")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Source-Specific Fetch Functions
# ---------------------------------------------------------------------------

def fetch_lasiesta(scenes: Optional[List[str]] = None) -> List[str]:
    """
    Downloads LASIESTA sequences.
    """
    target_scenes = scenes if scenes else list(LASIESTA_CATALOG.keys())
    out_dir = os.path.join(RAW_DIR, "lasiesta")
    os.makedirs(out_dir, exist_ok=True)

    total_bytes = sum(LASIESTA_CATALOG[s]["size"] for s in target_scenes if s in LASIESTA_CATALOG)
    print(f"\n========================================================")
    print(f"Fetching LASIESTA: {len(target_scenes)} scenes ({total_bytes / (1024 * 1024):.1f} MB)")
    print(f"========================================================")

    downloaded_files = []
    for s in target_scenes:
        if s not in LASIESTA_CATALOG:
            print(f"Warning: Unknown LASIESTA scene '{s}', skipping.")
            continue
        info = LASIESTA_CATALOG[s]
        dst = os.path.join(out_dir, f"{s}.rar")
        if download_file(info["url"], dst, expected_size=info["size"]):
            downloaded_files.append(dst)

    return downloaded_files


def fetch_bmc(videos: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Downloads BMC synthetic video pairs (video mp4 + gt mp4).
    """
    target_vids = videos if videos else list(BMC_CATALOG.keys())
    out_dir = os.path.join(RAW_DIR, "bmc")
    os.makedirs(out_dir, exist_ok=True)

    total_bytes = sum(
        BMC_CATALOG[v]["vid_size"] + BMC_CATALOG[v]["gt_size"]
        for v in target_vids if v in BMC_CATALOG
    )
    print(f"\n========================================================")
    print(f"Fetching BMC 2012: {len(target_vids)} videos ({total_bytes / (1024 * 1024):.1f} MB)")
    print(f"========================================================")

    downloaded_pairs = []
    for v in target_vids:
        if v not in BMC_CATALOG:
            print(f"Warning: Unknown BMC video '{v}', skipping.")
            continue
        info = BMC_CATALOG[v]
        vid_dst = os.path.join(out_dir, f"{v}.mp4")
        gt_dst = os.path.join(out_dir, f"{v}_gt.mp4")

        ok_vid = download_file(info["vid_url"], vid_dst, expected_size=info["vid_size"])
        ok_gt = download_file(info["gt_url"], gt_dst, expected_size=info["gt_size"])
        if ok_vid and ok_gt:
            downloaded_pairs.append((vid_dst, gt_dst))

    return downloaded_pairs


def fetch_ino(sequences: Optional[List[str]] = None) -> List[str]:
    """
    Downloads INO Video Analytics sequences.
    """
    target_seqs = sequences if sequences else list(INO_CATALOG.keys())
    out_dir = os.path.join(RAW_DIR, "ino")
    os.makedirs(out_dir, exist_ok=True)

    total_bytes = sum(INO_CATALOG[s]["size"] for s in target_seqs if s in INO_CATALOG)
    print(f"\n========================================================")
    print(f"Fetching INO: {len(target_seqs)} sequences ({total_bytes / (1024 * 1024):.1f} MB)")
    print(f"========================================================")

    downloaded_files = []
    for s in target_seqs:
        if s not in INO_CATALOG:
            print(f"Warning: Unknown INO sequence '{s}', skipping.")
            continue
        info = INO_CATALOG[s]
        dst = os.path.join(out_dir, f"{s}.zip")
        if download_file(info["url"], dst, expected_size=info["size"]):
            downloaded_files.append(dst)

    return downloaded_files


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch public change-detection datasets.")
    parser.add_argument("--sources", nargs="+", default=["lasiesta", "bmc", "ino"],
                        help="Sources to fetch: lasiesta, bmc, ino")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Specific scenes to fetch (default: all curated)")
    args = parser.parse_args()

    print("Target raw directory:", RAW_DIR)
    sources = [s.lower() for s in args.sources]

    if "lasiesta" in sources:
        fetch_lasiesta(scenes=args.scenes)
    if "bmc" in sources:
        fetch_bmc(videos=args.scenes)
    if "ino" in sources:
        fetch_ino(sequences=args.scenes)

    print("\nFetch complete.")


if __name__ == "__main__":
    main()
