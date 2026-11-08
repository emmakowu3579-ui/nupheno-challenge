"""
PanNuke Data Acquisition Utility.
Downloads and verifies the official PanNuke Dataset from Kaggle.

Dataset: andrewmvd/cancer-instance-segmentation-and-classification-2
Source: Tissue Image Analytics (TIA) Centre, University of Warwick
URL: https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KAGGLE_DATASET = "andrewmvd/cancer-instance-segmentation-and-classification-2"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"


def download_official_dataset(output_dir: Path):
    """Downloads the official PanNuke dataset using the Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("🔬 DOWNLOADING OFFICIAL PANNUKE HISTOPATHOLOGY DATASET")
    print(f"📦 Kaggle Source: {KAGGLE_DATASET}")
    print(f"🔗 URL: {KAGGLE_URL}")
    print(f"📂 Target Directory: {output_dir}")
    print("=" * 72)

    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(output_dir),
            "--unzip"
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Download and extraction complete!")
    except FileNotFoundError:
        print("\n❌ Error: 'kaggle' CLI tool not found on PATH.")
        print("Please install it with: pip install kaggle")
        print("Or download manually from: " + KAGGLE_URL)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during download: {e}")
        print("Ensure your kaggle.json API token is placed in ~/.kaggle/ or C:\\Users\\<user>\\.kaggle\\")
        sys.exit(1)


def verify_dataset(data_dir: Path):
    """Verifies that the downloaded dataset contains the expected PanNuke numpy arrays."""
    if not data_dir.exists():
        print(f"❌ Error: Directory does not exist: {data_dir}")
        return False

    npy_files = list(data_dir.rglob("*.npy"))
    if len(npy_files) > 0:
        print(f"✅ Verified: Found {len(npy_files)} numpy data files in {data_dir}:")
        for f in npy_files[:10]:
            print(f"   - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
        return True
    else:
        print(f"⚠️ Warning: No .npy files found in {data_dir}.")
        return False


def print_instructions():
    print(f"""
========================================================================
🔬 OFFICIAL PANNUKE DATASET INSTRUCTIONS
========================================================================
1. Kaggle Dataset:
   {KAGGLE_DATASET}
   URL: {KAGGLE_URL}

2. Contents:
   - 7,000+ real 256x256 H&E tissue biopsy patches across 19 human tissue types.
   - ~200,000 semi-automatically segmented and pathologist-verified cell nuclei.
   - 5 nuclear phenotypes: Neoplastic, Inflammatory, Connective, Dead, Epithelial.
   - Formats: images.npy, masks.npy, types.npy across 3 folds.

3. To download locally with Kaggle CLI:
   kaggle datasets download -d {KAGGLE_DATASET} -p data/pannuke_raw/ --unzip

4. To convert numpy arrays into standard COCO object detection format:
   python data/prepare_coco_pannuke.py --raw-dir data/pannuke_raw/ --output-dir data/processed_coco/

5. On Kaggle Notebooks:
   In the right-hand panel under 'Input', click '+ Add Input', search for:
   '{KAGGLE_DATASET}' and click Add.
========================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire and verify official PanNuke dataset")
    parser.add_argument("--download", action="store_true", help="Download official dataset via Kaggle CLI")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded dataset in target directory")
    parser.add_argument("--data-dir", type=str, default="data/pannuke_raw", help="Path to raw dataset directory")
    args = parser.parse_args()

    target_dir = Path(args.data_dir)

    if args.download:
        download_official_dataset(target_dir)
        verify_dataset(target_dir)
    elif args.verify:
        verify_dataset(target_dir)
    else:
        print_instructions()
