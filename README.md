# Financial Distress Prediction (FDP) Streaming Pipeline

This repository accompanies the thesis  
**“Financial Distress Prediction in Non-Stationary Data Streams:  
A Comparative Study of Machine-Learning Algorithms on Brazilian Listed Enterprises.”**

---

## 📁 Repository Structure

```text
my-fdp-project/
├── data/
│   └── cvm_indicators_dataset_2011-2021.csv    # Original CVM dataset
├── notebooks/                                  # Jupyter notebooks for each step
│   ├── 01_logistic_regression.ipynb
│   ├── 02_xgboost.ipynb
│   ├── 03_lightgbm.ipynb
│   ├── 04_mlp.ipynb
│   ├── 05_lstm.ipynb
│   ├── 06_bigru.ipynb
│   ├── 07_arf_adwin.ipynb
│   ├── 08_naive_bayes.ipynb
│   ├── 09_autoencoder.ipynb
│   ├── 10_LightGBM + SMOTE.ipynb
│   ├── CombinedVis (F1score).ipynb            # Cumulative F1 visualizations
│   ├── LightGBM SHAP.ipynb                    # SHAP analysis for LightGBM
│   ├── Naive Bayes SHAP.ipynb                  # SHAP analysis for Naive Bayes
│   └── Wilcoxon.ipynb                          # Wilcoxon tests & effect sizes
├── requirements.txt                            # Pinned Python dependencies
├── .gitignore                                  # Ignore rules for Git
└── README.md                                   # This file
```


🚀 Getting Started
Prerequisite: Python 3.9 or newer (use conda / venv for isolation).

1 · Clone the repository
git clone https://github.com/JianfengWang-cloud/fdp-streaming.git
cd fdp-streaming

2 · Install dependencies
pip install -r requirements.txt

3 · Prepare the data
Make sure the CVM CSV is present:
data/cvm_indicators_dataset_2011-2021.csv

4 · Launch Jupyter Lab
jupyter lab
Run the notebooks in sequence:

01_logistic_regression.ipynb

02_xgboost.ipynb

03_lightgbm.ipynb

…..
