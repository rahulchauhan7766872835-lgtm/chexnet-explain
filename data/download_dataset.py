"""
Dataset Download & Preparation Utility for CheXNet-Explain

Dataset: Kermany et al. / Chest X-Ray Images (Pneumonia)
Source: Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
Total: 5,856 X-Ray images (JPEG) categorized into NORMAL and PNEUMONIA
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import generate_mock_radiograph


def setup_sample_data(sample_dir: Path, count_per_class: int = 15):
    """
    Generates synthetic anatomically structured radiograph pairs
    for instantaneous smoke testing, continuous integration, and demo.
    """
    normal_dir = sample_dir / "NORMAL"
    pneumonia_dir = sample_dir / "PNEUMONIA"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Generating {count_per_class} sample radiographs per class in {sample_dir}...")
    for i in range(count_per_class):
        generate_mock_radiograph(normal_dir / f"normal_{i:03d}.png", is_pneumonia=False)
        generate_mock_radiograph(pneumonia_dir / f"pneumonia_{i:03d}.png", is_pneumonia=True)

    print("[SUCCESS] Sample dataset prepared successfully.")


def print_kaggle_instructions():
    """Prints instructions for fetching the full 2GB Kaggle dataset."""
    print("=" * 70)
    print("HOW TO DOWNLOAD FULL CHEST X-RAY DATASET (5,856 Images):")
    print("=" * 70)
    print("Method 1: Using the Kaggle CLI (Recommended)")
    print("  1. Ensure you have your kaggle.json token in ~/.kaggle/")
    print("  2. Run:")
    print("     kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("  3. Unzip into data/chest_xray:")
    print("     tar -xf chest-xray-pneumonia.zip -C data/chest_xray")
    print("\nMethod 2: Direct Web Download")
    print("  1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  2. Click 'Download' (approx 1.2 GB)")
    print("  3. Extract folders into 'data/chest_xray/' so the structure is:")
    print("       data/chest_xray/NORMAL/*.jpeg")
    print("       data/chest_xray/PNEUMONIA/*.jpeg")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheXNet Dataset Preparation")
    parser.add_argument("--samples-only", action="store_true", help="Generate local sample radiographs")
    parser.add_argument("--count", type=int, default=15, help="Number of sample images per class")
    args = parser.parse_args()

    sample_path = Path("data/sample_images")
    setup_sample_data(sample_path, count_per_class=args.count)
    print_kaggle_instructions()
