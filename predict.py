import pickle
import numpy as np
import pandas as pd
import shap

MODEL_PATH = 'model/attrition_model.pkl'

NUMERIC_FEATURES = [
    'Age', 'DistanceFromHome', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'WorkLifeBalance', 'JobInvolvement', 'EnvironmentSatisfaction',
    'RelationshipSatisfaction', 'NumCompaniesWorked', 'TrainingTimesLastYear',
    'StockOptionLevel', 'DailyRate', 'HourlyRate', 'MonthlyRate'
]

CATEGORICAL_FEATURES = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_input(df, model_data):
    preprocessor = model_data['preprocessor']
    num_feats = model_data['numeric_features']
    cat_feats = model_data['categorical_features']

    # Fill missing numeric
    for col in num_feats:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill missing categorical
    for col in cat_feats:
        if col not in df.columns:
            df[col] = 'Unknown'
        df[col] = df[col].fillna('Unknown').astype(str)

    X = df[num_feats + cat_feats]
    return preprocessor.transform(X)

def predict_single(input_dict, model_data):
    model = model_data['model']
    features = model_data['features']
    df = pd.DataFrame([input_dict])
    X = preprocess_input(df, model_data)
    prob = model.predict_proba(X)[0][1]
    label = get_risk_label(prob)
    factors = get_shap_factors(model, X, features)
    return prob, label, factors

def predict_bulk(df_raw, model_data):
    model = model_data['model']
    df = df_raw.copy()
    for col in ['Attrition', 'attrition']:
        if col in df.columns:
            df = df.drop(columns=[col])
    X = preprocess_input(df, model_data)
    probs = model.predict_proba(X)[:, 1]
    labels = [get_risk_label(p) for p in probs]
    return probs, labels

def get_risk_label(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.6:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def get_shap_factors(model, X, features, top_n=5):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        vals = shap_values[0]
        indices = np.argsort(np.abs(vals))[::-1][:top_n]
        factors = []
        for i in indices:
            factors.append({
                'feature': features[i],
                'impact': float(vals[i]),
                'value': float(X[0][i])
            })
        return factors
    except:
        return []