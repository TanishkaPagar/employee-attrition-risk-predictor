# 👥 Employee Attrition Risk Predictor

An HR analytics tool that predicts employee attrition risk using Machine Learning, with comprehensive evaluation methodology and explainable AI.

## 🔗 Live Demo

[https://employee-attrition-risk-predictor.streamlit.app/](https://employee-attrition-risk-predictor.streamlit.app/)

## 🛠️ Tech Stack

* Python 3.13
* XGBoost
* SHAP (Explainable AI)
* Streamlit
* Scikit-learn
* Imbalanced-learn (SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler)
* Plotly
* OpenPyXL

## ✨ Features

* Single employee risk prediction with interactive form
* Bulk prediction via CSV upload
* Risk gauge meter (Low / Medium / High)
* SHAP-based explainability (top factors driving risk)
* Downloadable color-coded Excel report
* OneHotEncoding for categorical features (no false ordinal relationships)
* Comprehensive evaluation: Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix
* Stratified 5-Fold Cross Validation for reliable performance estimates
* Comparison of 5 imbalance-handling techniques to select the best performing method

## ⚙️ How to Run Locally

### Prerequisites

* Python 3.13
* pip

### Steps

1. Clone the repository
git clone https://github.com/TanishkaPagar/employee-attrition-risk-predictor.git
cd employee-attrition-risk-predictor
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py
4. Open in browser
http://localhost:8501

## 📊 Model Performance

### Test Set Metrics

* Accuracy: 84.4%
* Precision: 0.513
* Recall: 0.426
* F1-Score: 0.465
* ROC-AUC: 0.782
* PR-AUC: 0.501

### Stratified 5-Fold Cross Validation

* CV F1: 0.444
* CV Recall: 0.321
* CV Precision: 0.736
* CV ROC-AUC: 0.817

### Imbalance Handling Comparison

|Method|F1 Score|Recall|
|-|-|-|
|RandomOverSampler|0.5109|0.4895|
|ADASYN|0.4627|0.3421|
|SMOTE|0.4561|0.3368|
|RandomUnderSampler|0.4436|0.6895|
|No Resampling|0.3809|0.2632|

**Best Method Selected:** RandomOverSampler (based on F1-score)

### Why Recall \& Precision Matter More Than Accuracy

The dataset is imbalanced (\~84% retained, \~16% attrited). A model predicting "No Attrition" for everyone would already score \~84% accuracy while being completely useless. Recall measures how many actual leavers were correctly identified, while Precision measures how many flagged high-risk predictions were correct — both are more meaningful than raw accuracy for this problem.

* Algorithm: XGBoost
* Training Data: IBM HR Analytics Dataset (1,470 employees)

## 📂 Project Structure

employee\_attrition\_risk\_predictor/
├── app.py                  # Main Streamlit app
├── model.py                # ML training, evaluation \& imbalance comparison
├── predict.py               # Prediction \& SHAP logic
├── utils.py                 # Excel report generator
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── model/
└── attrition\_model.pkl  # Trained model

## 👩‍💻 Developed By

Tanishka Pagar — LABTECH Internship 2026

