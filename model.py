import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import os

# Features to use
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

DROP_COLS = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    return df

def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', 'passthrough', NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ])

def compare_imbalance_methods(X_train, y_train):
    """Compare different imbalance handling methods"""
    methods = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'RandomOverSampler': RandomOverSampler(random_state=42),
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'No Resampling': None
    }

    results = {}
    model = XGBClassifier(n_estimators=100, max_depth=4,
                          learning_rate=0.05, eval_metric='logloss',
                          random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, sampler in methods.items():
        f1_scores = []
        recall_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if sampler is not None:
                try:
                    X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
                except:
                    pass

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            f1_scores.append(f1_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val, y_pred))

        results[name] = {
            'F1': np.mean(f1_scores),
            'Recall': np.mean(recall_scores)
        }
        print(f"{name}: F1={np.mean(f1_scores):.4f}, Recall={np.mean(recall_scores):.4f}")

    # Pick best method by F1
    best_method = max(results, key=lambda x: results[x]['F1'])
    print(f"\n✅ Best Method: {best_method}")
    return best_method, methods[best_method], results

def train_model(data_path):
    df = load_data(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Attrition']

    # Preprocess
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=== Comparing Imbalance Handling Methods ===")
    best_method_name, best_sampler, imbalance_results = compare_imbalance_methods(X_train, y_train)

    # Apply best sampler
    if best_sampler is not None:
        try:
            X_train_res, y_train_res = best_sampler.fit_resample(X_train, y_train)
        except:
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    # Train final model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Stratified Cross Validation
    print("\n=== Stratified 5-Fold Cross Validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_processed, y,
        cv=skf,
        scoring=['f1', 'recall', 'precision', 'roc_auc'],
        return_train_score=False
    )
    print(f"CV F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"CV Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
    print(f"CV Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"CV ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")

    # Get feature names after OHE
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + ohe_feature_names

    # Save everything
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'features': all_feature_names,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'best_sampler': best_method_name,
        'imbalance_results': imbalance_results,
        'metrics': {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': average_precision_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        },
        'cv_results': {
            'f1': cv_results['test_f1'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'precision': cv_results['test_precision'].mean(),
            'roc_auc': cv_results['test_roc_auc'].mean()
        }
    }

    os.makedirs('model', exist_ok=True)
    with open('model/attrition_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\n✅ Model saved to model/attrition_model.pkl")
    return model_data

if __name__ == '__main__':
    train_model('WA_Fn-UseC_-HR-Employee-Attrition.csv')