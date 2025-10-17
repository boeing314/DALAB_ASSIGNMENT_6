# 🧠 DA5401 A6 — Imputation via Regression for Missing Data

This repository contains a **Jupyter Notebook (`asgn6.ipynb`)** and a **project report (`DA5401 A6 Imputation via Regression.pdf`)** for Assignment 6 of DA5401.  
The objective of this assignment is to explore different **imputation strategies** for handling missing data and to evaluate their impact on **classification performance** using Logistic Regression.

---

## 📌 Problem Statement

You are working on a **credit risk assessment project** using the [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).  
The dataset has missing values in key columns, which makes it unsuitable for direct training of classification models.

**Goal:**  
Implement and compare three imputation strategies and a baseline listwise deletion method, then train a Logistic Regression model on each to measure performance.

---

## 📊 Tasks Overview

### **Part A — Data Preprocessing and Imputation**
1. **Load & Prepare Data**
   - Artificially introduce Missing At Random (MAR) values (5–10%) in numerical columns (e.g., `AGE` and `BILL_AMT`).
2. **Imputation Strategy 1 — Simple Imputation (Dataset A)**  
   - Fill missing values with the **median** of each column.
3. **Imputation Strategy 2 — Linear Regression Imputation (Dataset B)**  
   - Use Linear Regression to predict missing values based on other features.
4. **Imputation Strategy 3 — Non-Linear Regression Imputation (Dataset C)**  
   - Use a non-linear model such as **KNN Regression** or **Decision Tree Regression** to predict missing values.

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

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

yhat = model.predict(X_A_test.values)

accuracy = accuracy_score(y_A_test, yhat)
precision = precision_score(y_A_test, yhat, average='binary')
recall = recall_score(y_A_test, yhat, average='binary')
f1 = f1_score(y_A_test, yhat, average='binary')

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
```

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
| A     | Median                             |    –     |     –     |   –    |    –      |
| B     | Linear Regression                  |    –     |     –     |   –    |    –      |
| C     | Non-Linear Regression (KNN/DT)     |    –     |     –     |   –    |    –      |
| D     | Listwise Deletion                  |    –     |     –     |   –    |    –      |

*(Values to be filled based on your run results.)*

---

## 🧰 Requirements

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

## 📁 File Structure

```
📦 Project Folder
 ┣ 📜 asgn6.ipynb                # Main Jupyter notebook with code
 ┣ 📜 DA5401 A6 Imputation via Regression.pdf   # Assignment brief
 ┣ 📜 README.md                  # Project documentation
```

---

## 📝 Author
This project was created as part of **DA5401 coursework** to demonstrate the impact of imputation strategies on classification model performance.

---

✅ *Tip:* Prefer regression-based imputation when the missingness is MAR and the predictor relationship is strong.
