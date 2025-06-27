#  Diabetes Prediction using Logistic Regression

This project performs binary classification to predict diabetes from medical data using logistic regression. It involves data cleaning, outlier detection, model training, evaluation, and threshold tuning. All steps are documented in Jupyter notebooks.

---

##  Files

- `diabetes_prediction_dataset.csv` – Raw dataset containing patient health records
- `modeling(logistic).ipynb` – Logistic regression pipeline **after removing outliers**
- `modeling(logistic_with_outlier).ipynb` – Logistic regression pipeline **with outliers retained**

---

##  Project Workflow

### 1️ Data Preprocessing
- Loaded the dataset and checked for missing values
- Categorical variables (like `gender`) were encoded using one-hot encoding
- Features (`X`) and target variable (`diabetes`) were separated

### 2️ Outlier Detection & Removal
- Boxplots were used to detect outliers in the following columns:
  - `age`
  - `bmi`
  - `HbA1c_level`
  - `blood_glucose_level`
- Outliers were removed using the **IQR (Interquartile Range) method**
- The `diabetes` target column was preserved to avoid class imbalance

### 3️ Train/Test Split and Standardization
- Data was split using stratified train-test split (80/20)
- Features were standardized using `StandardScaler` to improve model convergence

### 4️ Logistic Regression Modeling
- Trained a logistic regression model on both:
  - Cleaned dataset (after outlier removal)
  - Full dataset (with outliers)
- Compared the performance of both models

### 5️ Model Evaluation
- Evaluation metrics included:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Score

### 6️ Threshold Tuning
- Custom thresholds from 0.1 to 0.9 were tested
- Precision, recall, and F1 score were evaluated at each threshold
- The best threshold (≈ 0.85) was selected based on F1 performance

---

## Observations

| Scenario               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| **With Outliers**      | 0.9605   | 0.86      | 0.64   | 0.73     | 0.96    |
| **Without Outliers**   | 0.8560   | 0.49      | 0.59   | 0.53     | 0.94    |

- Including outliers improved performance, likely because extreme diabetic cases made class separation easier.
- Removing outliers led to a cleaner dataset but slightly lower model performance.
- Threshold tuning significantly improved **F1 Score** by balancing precision and recall.

---

## Libraries Used

- Python 3.x
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## ✅ Next Steps

- Try advanced models (e.g., Random Forest, XGBoost)
- Handle class imbalance using SMOTE or class weights
- Deploy the model via Streamlit or Flask API
