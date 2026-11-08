"""
Starter Data Visualization & Exploratory Data Analysis (EDA) for PanNuke Challenge.
Visualizes sample H&E tissue patches with color-coded nuclei bounding boxes alongside
dataset-wide class distribution, cellular density per patch, and nucleus morphology statistics.

Usage:
    python starter_code/visualize_data.py
    python starter_code/visualize_data.py --annotations data/public/train_annotations.csv --output starter_code/pannuke_eda.png
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


def synthesize_he_patch_from_boxes(df_patch: pd.DataFrame, width: int = 256, height: int = 256, seed: int = 42) -> np.ndarray:
    """
    Generate a realistic H&E-stained histology background and render dark hematoxylin
    nuclei at the exact ground-truth bounding box coordinates when raw images are not yet downloaded.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Eosin (pinkish-purple stroma background)
    img[:, :, 0] = rng.integers(215, 242, (height, width))
    img[:, :, 1] = rng.integers(175, 208, (height, width))
    img[:, :, 2] = rng.integers(205, 235, (height, width))

    y_grid, x_grid = np.ogrid[:height, :width]
    for _, row in df_patch.iterrows():
        cx = row["x"] + row["width"] / 2.0
        cy = row["y"] + row["height"] / 2.0
        rx = max(2.0, row["width"] / 2.0)
        ry = max(2.0, row["height"] / 2.0)
        ellipse = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 <= 1.0
        # Dark purple-blue Hematoxylin nuclear stain
        img[ellipse, 0] = rng.integers(65, 110)
        img[ellipse, 1] = rng.integers(30, 70)
        img[ellipse, 2] = rng.integers(95, 150)

    return img


def load_or_render_patch(img_id: int, df_patch: pd.DataFrame, images_dir: Path) -> np.ndarray:
    """Load patch image from disk if available, otherwise synthesize from ground-truth boxes."""
    if images_dir and images_dir.exists():
        for ext in [".png", ".jpg", ".tif"]:
            candidates = [
                images_dir / f"patch_{img_id:04d}{ext}",
                images_dir / f"{img_id}{ext}",
                images_dir / f"image_{img_id}{ext}",
            ]
            for cand in candidates:
                if cand.exists():
                    return np.array(Image.open(cand).convert("RGB"))
    return synthesize_he_patch_from_boxes(df_patch, seed=int(img_id) + 100)


def create_pannuke_eda_dashboard(
    annotations_csv: Path,
    images_dir: Path,
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
        "PanNuke Histopathology Nuclei Detection & Phenotyping — Exploratory Data Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # ==================== TOP ROW: 3 SAMPLE H&E PATCHES WITH BOUNDING BOXES ====================
    unique_imgs = df["image_id"].unique()
    sample_ids = unique_imgs[:3] if len(unique_imgs) >= 3 else unique_imgs

    for idx, img_id in enumerate(sample_ids):
        ax = fig.add_subplot(2, 3, idx + 1)
        df_patch = df[df["image_id"] == img_id]
        img_np = load_or_render_patch(int(img_id), df_patch, images_dir)

        ax.imshow(img_np)
        for _, row in df_patch.iterrows():
            cid = int(row["category_id"])
            color = CATEGORY_COLORS.get(cid, "#FFFF00")
            rect = patches.Rectangle(
                (row["x"], row["y"]),
                row["width"],
                row["height"],
                linewidth=1.8,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect)

        ax.set_title(f"H&E Patch #{img_id} ({len(df_patch)} Nuclei)", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Add shared phenotype color legend on the first subplot
    legend_handles = [
        patches.Patch(edgecolor=CATEGORY_COLORS[cid], facecolor=CATEGORY_COLORS[cid], label=f"{cid}: {name}", alpha=0.8)
        for cid, name in CATEGORY_NAMES.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        frameon=True,
        fontsize=10,
        title="Cellular Phenotype Classes"
    )

    # ==================== BOTTOM-LEFT: CLASS DISTRIBUTION ====================
    ax_cls = fig.add_subplot(2, 3, 4)
    counts = df["category_id"].value_counts().reindex(range(1, 6), fill_value=0)
    names = [CATEGORY_NAMES[i] for i in range(1, 6)]
    colors = [CATEGORY_COLORS[i] for i in range(1, 6)]

    bars = ax_cls.bar(names, counts.values, color=colors, edgecolor="black", alpha=0.85)
    ax_cls.set_title("Nuclei Phenotype Class Distribution", fontsize=12, fontweight="bold")
    ax_cls.set_ylabel("Number of Annotated Nuclei")
    ax_cls.tick_params(axis="x", rotation=20)
    ax_cls.grid(axis="y", linestyle="--", alpha=0.4)

    total_nuclei = max(1, len(df))
    for bar, cnt in zip(bars, counts.values):
        pct = 100.0 * cnt / total_nuclei
        ax_cls.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (counts.max() * 0.015),
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )
    ax_cls.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 10)

    # ==================== BOTTOM-CENTER: CELLULAR DENSITY PER PATCH ====================
    ax_dens = fig.add_subplot(2, 3, 5)
    per_patch = df.groupby("image_id").size()
    ax_dens.hist(per_patch.values, bins=12, color="#377EB8", edgecolor="black", alpha=0.8)
    mean_cells = per_patch.mean()
    ax_dens.axvline(mean_cells, color="#E41A1C", linestyle="--", linewidth=2, label=f"Mean: {mean_cells:.1f} cells/patch")
    ax_dens.set_title("Cellular Density (Nuclei per 256×256 Patch)", fontsize=12, fontweight="bold")
    ax_dens.set_xlabel("Nuclei Count per Patch")
    ax_dens.set_ylabel("Number of Patches")
    ax_dens.legend(loc="upper right")
    ax_dens.grid(linestyle="--", alpha=0.4)

    # ==================== BOTTOM-RIGHT: NUCLEUS AREA BY PHENOTYPE ====================
    ax_box = fig.add_subplot(2, 3, 6)
    area_data = [df[df["category_id"] == i]["area"].values for i in range(1, 6)]
    valid_data = [d if len(d) > 0 else np.array([0]) for d in area_data]

    bp = ax_box.boxplot(valid_data, tick_labels=names, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax_box.set_title("Bounding Box Area Distribution by Phenotype", fontsize=12, fontweight="bold")
    ax_box.set_ylabel("Bounding Box Area (pixels²)")
    ax_box.tick_params(axis="x", rotation=20)
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved PanNuke EDA visualization dashboard to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Visualize PanNuke training annotations and patch samples.")
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(root / "data" / "public" / "train_annotations.csv"),
        help="Path to train_annotations.csv"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(root / "data" / "mock_coco" / "images"),
        help="Optional directory containing extracted patch PNG images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root / "starter_code" / "eda_pannuke_overview.png"),
        help="Output path for the generated EDA figure"
    )
    parser.add_argument("--show", action="store_true", help="Display interactive matplotlib window")
    args = parser.parse_args()

    create_pannuke_eda_dashboard(
        annotations_csv=Path(args.annotations),
        images_dir=Path(args.images_dir),
        output_path=Path(args.output),
        show_plot=args.show
    )


if __name__ == "__main__":
    main()
