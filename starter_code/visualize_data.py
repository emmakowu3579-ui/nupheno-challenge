"""
Starter Data Visualization & Exploratory Data Analysis (EDA) for NuPheno (PanNuke Challenge).
Visualizes ground-truth nuclei bounding boxes, phenotypic class distribution,
cellular density per patch, and nuclear morphology statistics.

Dataset: andrewmvd/cancer-instance-segmentation-and-classification-2
Source: Tissue Image Analytics (TIA) Centre, University of Warwick

Usage:
    python starter_code/visualize_data.py
    python starter_code/visualize_data.py --annotations data/public/train_annotations.csv --output starter_code/eda_pannuke_overview.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CATEGORY_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

CATEGORY_COLORS = {
    1: "#E41A1C",  # Red - Neoplastic (Tumor)
    2: "#4DAF4A",  # Green - Inflammatory (Immune)
    3: "#377EB8",  # Blue - Connective (Stroma)
    4: "#FF7F00",  # Orange - Dead (Necrotic/Apoptotic)
    5: "#984EA3",  # Purple - Epithelial (Benign)
}


def find_patch_image(img_id: int, search_roots):
    """Searches for real histology image patch on disk if downloaded."""
    for root in search_roots:
        if not root or not root.exists():
            continue
        for ext in [".png", ".jpg", ".tif"]:
            candidates = [
                root / f"patch_{img_id:05d}{ext}",
                root / f"patch_{img_id:04d}{ext}",
                root / f"{img_id}{ext}",
                root / "images" / f"patch_{img_id:05d}{ext}",
            ]
            for cand in candidates:
                if cand.exists():
                    return np.array(Image.open(cand).convert("RGB"))
    return None


def create_pannuke_eda_dashboard(
    annotations_csv: Path,
    search_dirs: list,
    output_path: Path,
    show_plot: bool = False
):
    if not annotations_csv.exists():
        print(f"❌ Error: Annotations file not found at {annotations_csv}")
        sys.exit(1)

    df = pd.read_csv(annotations_csv)
    df["area"] = df["width"] * df["height"]
    df["category_name"] = df["category_id"].map(CATEGORY_NAMES)

    print(f"📊 Loaded {len(df):,} nuclei annotations across {df['image_id'].nunique()} H&E patches.")

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.suptitle(
        "NuPheno: Pan-Cancer Nuclear Phenotyping & Detection — Exploratory Data Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # 3 Sample Patches (Top Row)
    sample_patch_ids = df["image_id"].unique()[:3]

    for idx, img_id in enumerate(sample_patch_ids):
        ax = fig.add_subplot(2, 3, idx + 1)
        patch_df = df[df["image_id"] == img_id]
        real_img = find_patch_image(int(img_id), search_dirs)

        if real_img is not None:
            ax.imshow(real_img)
            ax.set_title(f"H&E Patch #{img_id} (Real Image)", fontsize=11, fontweight="bold", pad=8)
        else:
            # Clean coordinate canvas for bounding boxes
            canvas = np.full((256, 256, 3), 248, dtype=np.uint8)
            ax.imshow(canvas)
            ax.set_title(f"Patch #{img_id} Ground Truth Box Layout", fontsize=11, fontweight="bold", pad=8)

        # Overlay color-coded bounding boxes
        for _, box in patch_df.iterrows():
            cid = int(box["category_id"])
            color = CATEGORY_COLORS.get(cid, "#000000")
            rect = patches.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                linewidth=1.8,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.axis("on")
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)

    # ==================== BOTTOM ROW: STATISTICAL DISTRIBUTIONS ====================
    # 1. Phenotypic Class Distribution
    ax_class = fig.add_subplot(2, 3, 4)
    cat_counts = df["category_name"].value_counts()[list(CATEGORY_NAMES.values())]
    colors = [CATEGORY_COLORS[cid] for cid in CATEGORY_NAMES.keys()]
    bars = ax_class.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="black", alpha=0.85)
    ax_class.set_title("Cellular Phenotype Distribution", fontsize=11, fontweight="bold")
    ax_class.set_ylabel("Total Annotated Nuclei", fontsize=10)
    ax_class.tick_params(axis="x", rotation=25)
    ax_class.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax_class.text(b.get_x() + b.get_width() / 2, b.get_height() + 10, f"{int(b.get_height()):,}", ha="center", fontsize=8, fontweight="bold")

    # 2. Nuclei Density per Patch
    ax_density = fig.add_subplot(2, 3, 5)
    density = df.groupby("image_id").size()
    ax_density.hist(density, bins=15, color="#2563EB", edgecolor="white", alpha=0.85)
    ax_density.axvline(density.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {int(density.median())}")
    ax_density.set_title("Cellular Density (Nuclei / 256x256 Patch)", fontsize=11, fontweight="bold")
    ax_density.set_xlabel("Number of Nuclei", fontsize=10)
    ax_density.set_ylabel("Frequency (Patches)", fontsize=10)
    ax_density.legend(loc="upper right", fontsize=9)
    ax_density.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Nuclear Aspect Ratio & Size Distribution
    ax_size = fig.add_subplot(2, 3, 6)
    for cid, name in CATEGORY_NAMES.items():
        sub = df[df["category_id"] == cid]
        ax_size.scatter(sub["width"], sub["height"], s=10, alpha=0.45, label=name, color=CATEGORY_COLORS[cid])
    ax_size.set_title("Nucleus Dimensions (Width vs Height in px)", fontsize=11, fontweight="bold")
    ax_size.set_xlabel("Bounding Box Width (px)", fontsize=10)
    ax_size.set_ylabel("Bounding Box Height (px)", fontsize=10)
    ax_size.legend(loc="upper right", markerscale=2, fontsize=8)
    ax_size.grid(True, linestyle="--", alpha=0.5)

    plt.subplots_adjust(top=0.91, bottom=0.09, left=0.06, right=0.96, hspace=0.32, wspace=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"✅ Generated Exploratory Data Analysis Overview: {output_path}")

    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate EDA visualization for NuPheno PanNuke Challenge")
    parser.add_argument("--annotations", type=str, default=str(root / "data" / "public" / "train_annotations.csv"), help="Path to annotations CSV")
    parser.add_argument("--output", type=str, default=str(root / "starter_code" / "eda_pannuke_overview.png"), help="Output path for EDA image")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    search_roots = [
        root / "data" / "pannuke_raw",
        root / "data" / "processed_coco",
        root / "data" / "processed_coco" / "images",
    ]

    create_pannuke_eda_dashboard(
        annotations_csv=Path(args.annotations),
        search_dirs=search_roots,
        output_path=Path(args.output),
        show_plot=args.show
    )
