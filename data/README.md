# 📂 PanNuke Dataset & Public Files

This directory contains the data preparation scripts, dataset manifests, and public files for the PanNuke Histopathology Nuclei Detection & Phenotyping Challenge.

## Public Files (`data/public/`)

These files are public and distributed directly to participants for training and local benchmarking:

| File | Description | Schema / Columns |
|---|---|---|
| `train_annotations.csv` | Training nuclei bounding boxes and cellular phenotypes | `image_id, category_id, x, y, width, height` |
| `test_images.csv` | Test patch metadata without annotations | `image_id, file_name, width, height` |
| `sample_submission.csv` | Schema-compliant sample submission file | `image_id, category_id, x, y, width, height, score` |

---

## Cellular Phenotypes

| ID | Class Name | Biological Description |
|:---:|---|---|
| **1** | **Neoplastic** | Malignant or transformed tumor cells |
| **2** | **Inflammatory** | Leukocytes, lymphocytes, plasma cells |
| **3** | **Connective** | Fibroblasts, endothelial cells, stromal elements |
| **4** | **Dead** | Apoptotic, pyknotic, and necrotic debris |
| **5** | **Epithelial** | Normal benign epithelial cells |

---

## Generating Mock Dataset (for Rapid Dev & CI)

To generate synthetic H&E pathology patches and COCO annotations:
```bash
python data/download.py --generate-mock --output-dir data/mock_coco
```

## Downloading Official PanNuke Dataset

1. Download Fold 1, Fold 2, Fold 3 from Warwick TIA Centre / Zenodo:
   - https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
   - https://zenodo.org/records/3901832
2. Place unzipped folders into `data/pannuke_raw/`:
   ```text
   data/pannuke_raw/
   ├── Fold 1/
   │   ├── images.npy
   │   ├── masks.npy
   │   └── types.npy
   ├── Fold 2/
   └── Fold 3/
   ```
3. Run COCO conversion:
   ```bash
   python data/prepare_coco_pannuke.py --raw-dir data/pannuke_raw/ --output-dir data/processed_coco/
   ```
