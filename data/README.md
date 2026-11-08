# 📂 NuPheno: PanNuke Dataset & Public Files

This directory contains the dataset documentation, annotations, and data preparation utilities for the NuPheno Pan-Cancer Nuclei Detection & Phenotyping Challenge.

---

## 🔬 Official Dataset Source

- **Dataset:** [Cancer Instance Segmentation and Classification 2 (`andrewmvd/cancer-instance-segmentation-and-classification-2`)](https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2)
- **Original Source:** Tissue Image Analytics (TIA) Centre, University of Warwick ([PanNuke Official Release](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke))
- **Patches:** **7,000+ real 256x256 H&E stained pathology image patches** across 19 human organs/tissues.
- **Nuclei:** **~200,000 pathologist-verified individual nuclei annotations**.
- **Phenotypic Classes:**
  1. `Neoplastic` (Malignant tumor cells)
  2. `Inflammatory` (Immune cells: lymphocytes, neutrophils, plasma cells)
  3. `Connective` (Stroma: fibroblasts, endothelial cells)
  4. `Dead` (Apoptotic and necrotic cells)
  5. `Epithelial` (Benign non-neoplastic epithelial cells)

---

## 📄 Public Files (`data/public/`)

These files are distributed directly to participants for training and local benchmarking:

| File | Description | Schema / Columns |
|---|---|---|
| `train_annotations.csv` | Training nuclei bounding box annotations | `image_id, category_id, x, y, width, height, category_name` |
| `test_images.csv` | Test image IDs list | `image_id, patch_id, tissue_type` |
| `sample_submission.csv` | Schema-compliant candidate predictions | `image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, confidence` |

---

## ⬇️ How to Acquire the Dataset

### Option A: In Kaggle Notebooks (Recommended)
1. Open your notebook on Kaggle.
2. In the right-hand sidebar under **Input**, click **`+ Add Input`**.
3. Search for: `andrewmvd/cancer-instance-segmentation-and-classification-2`
4. Click **`+` (Add)** to mount it directly into `/kaggle/input`.

### Option B: Local Download via Kaggle CLI
```bash
kaggle datasets download -d andrewmvd/cancer-instance-segmentation-and-classification-2 -p data/pannuke_raw/ --unzip
```

To convert raw fold numpy arrays (`images.npy`, `masks.npy`, `types.npy`) to COCO format:
```bash
python data/prepare_coco_pannuke.py --raw-dir data/pannuke_raw/ --output-dir data/processed_coco/
```
