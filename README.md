# Financial Distress Prediction (FDP) Streaming Pipeline

This repository contains the code and notebooks for our thesis project:
“Financial Distress Prediction in Non-Stationary Data Streams:
A Comparative Study of Machine Learning Algorithms on Brazilian Listed Enterprises.”

## Structure

- `models.py`            – model definitions and wrappers  
- `tune_hyperparams.py`  – hyperparameter search scripts  
- `data/`                – (if you include the CVM CSV here)  
- `notebooks/`           – Jupyter notebooks for analysis  
- `requirements.txt`     – pinned Python dependencies  
- `README.md`            – this file  
- `.gitignore`           – files/folders to exclude from version control  

## Getting Started

```bash
# clone repo
git clone https://github.com/JianfengWang-cloud/fdp-streaming.git
cd fdp-streaming

# install dependencies
pip install -r requirements.txt

# run hyperparameter tuning
python tune_hyperparams.py

# etc.
