# 🚀 NuPheno Starter Code: Nuclear Phenotyping & Detection

This directory contains starter materials to help you explore the dataset, build a baseline detector, evaluate predictions, and encrypt your competition submission for the **NuPheno Challenge**.

![PanNuke Exploratory Data Analysis](eda_pannuke_overview.png)

## Files
- `visualize_data.py`: Exploratory Data Analysis (EDA) & patch visualization script that plots H&E histology patches with color-coded nuclei bounding boxes, phenotype class balance, cellular density per patch, and nucleus area distributions (`eda_pannuke_overview.png`).
- `baseline.py`: Trainable starter baseline script producing `submissions/baseline_submission.csv`.
- `encrypt.py`: Utility script to encrypt your submission file using the official RSA public key.
- `public_key.pem`: Challenge public key for submission encryption.
- `requirements.txt`: Python package requirements.

## Quick Start
```bash
# 1. Install dependencies
pip install -r starter_code/requirements.txt

# 2. Visualize the dataset & generate EDA overview plot
python starter_code/visualize_data.py

# 3. Run the baseline model
python starter_code/baseline.py

# 4. Encrypt your submission
python starter_code/encrypt.py submissions/<team_name>.csv

# 5. Commit and submit PR
git add submissions/<team_name>.csv.enc
git commit -m "Submit encrypted predictions for <team_name>"
git push origin <your-branch>
```
