# Logistic Regression Evaluation

This repository contains a Jupyter Notebook (`asgn6.ipynb`) that demonstrates the training and evaluation of a **Logistic Regression** model.

## 📌 Contents
- **Model Training:** LogisticRegression model from scikit-learn.
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score.
- **Dataset Handling:** Train-test split and scaling applied to input features.

## ⚡ How to Run
1. Open the notebook `asgn6.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells sequentially.
3. The model will output classification metrics for the test set.

## 📊 Metrics Calculated
- **Accuracy:** Overall correctness of the model.
- **Precision:** Positive predictive value.
- **Recall:** Sensitivity of the model.
- **F1-score:** Harmonic mean of precision and recall.

## 🧰 Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- jupyter

Install the required dependencies with:
```bash
pip install scikit-learn pandas numpy jupyter
```

## 📈 Example Code Snippet
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

## 📝 Author
This notebook was created as part of an assignment on evaluating Logistic Regression models.

---
⭐ **Tip:** Use `classification_report` for a more detailed performance summary.
