# A6 — Imputation via Regression for Missing Data

This repository contains a **Jupyter Notebook (`asgn6.ipynb`)** and a **project report (`DA5401 A6 Imputation via Regression.pdf`)** for Assignment 6 of DA5401.  
The objective of this assignment is to explore different **imputation strategies** for handling missing data and to evaluate their impact on **classification performance** using Logistic Regression.

---

## Problem Statement

We will be working on a **credit risk assessment project** using the [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).  
The dataset has missing values in key columns, which makes it unsuitable for direct training of classification models.

**Goal:**  
Implement and compare three imputation strategies and a baseline listwise deletion method, then train a Logistic Regression model on each to measure performance.

---

## Tasks Overview

### **Part A — Data Preprocessing and Imputation**
1. **Load & Prepare Data**
   - Artificially introduce Missing At Random (MAR) values (5–10%) in numerical columns.
2. **Imputation Strategy 1 — Simple Imputation (Dataset A)**  
   - Fill missing values with the **median** of each column.
3. **Imputation Strategy 2 — Linear Regression Imputation (Dataset B)**  
   - Use Linear Regression to predict missing values based on other features.
4. **Imputation Strategy 3 — Non-Linear Regression Imputation (Dataset C)**  
   - Use a non-linear model such as **KNN Regression** to predict missing values.

---

### **Part B — Model Training and Evaluation**
1. **Data Splitting:**  
   - For each dataset (A, B, C), create train-test splits.  
   - Also create **Dataset D** using listwise deletion.
2. **Feature Standardization:**  
   - Apply `StandardScaler` to all datasets.
3. **Logistic Regression Model:**  
   - Train on each dataset and compute:
     - Accuracy
     - Precision
     - Recall
     - F1-score

---

### **Part C — Comparative Analysis**
- Compare performance of:
  - **Model A** (Median Imputation)  
  - **Model B** (Linear Regression Imputation)  
  - **Model C** (Non-Linear Regression Imputation)  
  - **Model D** (Listwise Deletion)
- Create a **summary table** of metrics.
- Discuss:
  - Why listwise deletion might underperform.
  - Which regression approach is best suited.
  - Implications of the relationship between imputed features and predictors.

---

## 📈 Expected Outcome
| Model | Imputation Strategy                | Accuracy | Precision | Recall | F1-score |
|-------|-------------------------------------|----------|-----------|--------|-----------|
| A     | Median                             |    0.7585     |     0.5994     |   0.6144    |    0.8062      |
| B     | Linear Regression                  |    0.7542     |     0.5997     |   0.6147    |    0.8056      |
| C     | Non-Linear Regression (KNN/DT)     |    0.7546     |     0.6002     |   0.6154    |    0.8058      |
| D     | Listwise Deletion                  |    0.7641     |     0.6125     |   0.6331    |    0.8164      |

*(Values to be filled based on your run results.)*

---

##  Requirements

- Python 3.x  
- Jupyter Notebook  
- NumPy  
- Pandas  
- scikit-learn

Install dependencies:
```bash
pip install numpy pandas scikit-learn jupyter
```

---

##  File Structure

```
 Project Folder
 ┣  asgn6.ipynb                # Main Jupyter notebook with code
 ┣  UCI_Credit_Card            # Dataset
 ┣  README.md                  # Project documentation
```

