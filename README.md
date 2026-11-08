# 🔬 NuPheno: Pan-Cancer Nuclear Phenotyping & Detection Challenge

[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-Web%20Dashboard-brightgreen?style=for-the-badge&logo=githubpages)](https://emmakowu3579-ui.github.io/nupheno-challenge/)
[![Metric](https://img.shields.io/badge/Metric-mAP%40%5B0.50%3A0.95%5D-green?style=for-the-badge)](LEADERBOARD.md)
[![Dataset](https://img.shields.io/badge/Dataset-PanNuke%2019%20Tissues-blue?style=for-the-badge)](https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2)
[![Submissions](https://img.shields.io/badge/Submissions-Encrypted%20PRs-blueviolet?style=for-the-badge)](submissions/)

## Welcome to the NuPheno Challenge 🚀

**NuPheno** (**Nu**clear **Pheno**typing & Detection) bridges **Computational Pathology**, **Cellular Oncology**, and **Computer Vision**. Your task is to build a high-performance vision model that can detect cell nuclei and classify their cellular phenotype across 19 human tissue types from hematoxylin and eosin (H&E) stained tissue biopsies. Accurate nuclear phenotyping is fundamental for automated cancer grading, tumor microenvironment profiling, and precision oncology.

---

## 🎯 The Task: Multi-Class Nuclei Detection

Given $256 \times 256$ pathology image patches, your model must:
1. **Detect** individual cell nuclei with precise bounding boxes $[x, y, w, h]$.
2. **Phenotype** each detected nucleus into one of 5 clinically distinct biological categories.
3. **Assign** a well-calibrated confidence score $[0.0, 1.0]$.

### 🔍 Challenge Motivation

Cellular morphology and spatial organization vary dramatically across human tissues (e.g., breast ductal carcinoma vs. colon adenocarcinoma vs. prostate adenocarcinoma). Standard detectors often struggle on rare classes (such as apoptotic or necrotic debris) or confuse infiltrating lymphocytes with stromal fibroblasts. This challenge tests cross-tissue generalization and phenotypic discrimination.

---

## 📂 The Dataset (PanNuke / Warwick TIA)

The data is derived from the renowned [PanNuke dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) curated by the Tissue Image Analytics (TIA) Centre at the University of Warwick, and is available on Kaggle at [Cancer Instance Segmentation and Classification 2](https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2).

```text
[ Histology Patch ]  -->  [ Nuclei Detection ]  -->  [ Cellular Phenotyping ]
        🖼️                       📍                             🏷️
    (256 x 256)            (Bounding Box)             (5 Biological Classes)
                                                                 |
                                                     (mAP@[0.50:0.95] Scoring)
```

### 1. Cellular Phenotype Classes

| Class ID | Cell Phenotype | Description & Clinical Significance | Color Code |
|:---:|---|---|---|
| **1** | **Neoplastic** | Malignant or transformed tumor cells with enlarged, irregular, hyperchromatic nuclei. | 🔴 Red |
| **2** | **Inflammatory** | Leukocytes, lymphocytes, and plasma cells reflecting tumor microenvironment immune response. | 🟢 Green |
| **3** | **Connective** | Fibroblasts, endothelial cells, and stromal cells forming structural tumor stroma. | 🔵 Blue |
| **4** | **Dead** | Apoptotic and pyknotic nuclear fragments indicating necrosis or treatment response. | ⚪ Gray |
| **5** | **Epithelial** | Non-neoplastic, benign epithelial cells. | 🟣 Magenta |

---

## 2. File Structure (`data/public/`)

| File | Description | Schema / Columns |
|---|---|---|
| `train_annotations.csv` | Public ground-truth bounding boxes & cell labels | `image_id, category_id, x, y, width, height` |
| `test_images.csv` | Unseen test patch metadata (without ground truth) | `image_id, file_name, width, height` |
| `sample_submission.csv` | Example submission format | `image_id, category_id, x, y, width, height, score` |

---

## 📝 Submission Format

Create a single CSV file named strictly using your team name: `submissions/<team_name>.csv` (e.g. `submissions/emmanuel_owusu.csv`).

```csv
image_id,category_id,x,y,width,height,score
41,1,105.0,78.5,22.0,19.5,0.9520
41,2,42.0,185.0,10.5,9.8,0.8840
42,1,88.0,94.2,24.1,20.0,0.9120
```

### Submission Requirements:
- Header must be exactly: `image_id,category_id,x,y,width,height,score`
- Coordinates: $x, y$ are top-left bounding box pixel coordinates in $[0, 256]$.
- Dimensions: $width, height > 0$.
- Category IDs: Integers in range $[1, 5]$.
- Scores: Confidence probabilities in range $[0.0, 1.0]$.

---

## 🚀 How to Participate

### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/emmakowu3579-ui/nupheno-challenge.git
cd nupheno-challenge
pip install -r starter_code/requirements.txt
```

### 2️⃣ Run the Baseline Model
A starter baseline detector is provided in `starter_code/`:
```bash
python starter_code/baseline.py
```
This generates your initial submission file at `submissions/baseline_submission.csv`.

### 3️⃣ Encrypt Your Submission
To prevent test set label leakage, you **must encrypt** your submission CSV using the challenge public key before committing:

```bash
# Usage: python starter_code/encrypt.py submissions/<team_name>.csv
python starter_code/encrypt.py submissions/emmanuel_owusu.csv
```
This produces `submissions/emmanuel_owusu.csv.enc`.

### 4️⃣ Submit via GitHub Pull Request
> ⚠️ **IMPORTANT:** Do **NOT** commit raw `.csv` files. Commit **ONLY** the `.enc` file with your team name.

```bash
git checkout -b submission-emmanuel_owusu
git add submissions/emmanuel_owusu.csv.enc
git commit -m "Submit encrypted predictions for emmanuel_owusu"
git push origin submission-emmanuel_owusu
```
Then open a **Pull Request** against `main` on GitHub!

---

## 🤖 Instant Auto-Grading

Once your PR is opened:
- ✅ The **GitHub Actions Auto-Grader Bot** triggers automatically.
- 🔓 Your submission is decrypted securely using the maintainer private key.
- 📊 Official **mAP@[0.50:0.95]** and **mAP@0.50** metrics are evaluated.
- 💬 The scorecard is posted as a **comment directly on your PR**:

```text
### 🤖 Automated Grading Report
| Challenge | Metric | Score | Status |
| :--- | :--- | :--- | :--- |
| **PanNuke Detection** | **mAP@[0.50:0.95]** | **0.4285** | ✅ Valid |
```

When your PR is merged by a maintainer, 🎉 your score appears on the live Leaderboard!

---

## 📏 Rules & Restrictions

| Rule | Detail |
|---|---|
| **Primary Metric** | Mean Average Precision ($\text{mAP}@[0.50:0.95]$) |
| **Allowed Models** | Any detection model (Faster R-CNN, YOLO, RT-DETR, Co-DETR, Swin Transformer) |
| **External Data** | Pretrained ImageNet / COCO weights are allowed. External pathology datasets are forbidden. |
| **Submissions** | One official team submission per PR. |
| **Format** | Strictly encrypted `.csv.enc` via `starter_code/encrypt.py`. |

---

## 🏆 Leaderboard

- **Markdown Leaderboard Table:** [📈 View LEADERBOARD.md](LEADERBOARD.md)
- **Interactive Web Leaderboard:** [🌐 View Interactive PanNuke Leaderboard](index.html)
