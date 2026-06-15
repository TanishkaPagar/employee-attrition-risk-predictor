import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import os

def detect_target_column(df):
    priority_keywords = [
        'attrition', 'churn', 'default', 'fraud', 'target',
        'label', 'outcome', 'risk', 'leave', 'exit', 'status'
    ]
    cols_lower = {col.lower(): col for col in df.columns}
    for keyword in priority_keywords:
        for col_lower, col_original in cols_lower.items():
            if keyword in col_lower:
                return col_original
    return df.columns[-1]

def auto_preprocess(df, target_col):
    df = df.copy()

    # Drop useless columns
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique() <= 1:
            drop_cols.append(col)
        if df[col].nunique() == len(df) and df[col].dtype == object:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)

    # Fill missing values safely
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if df[col].dtype == object or str(df[col].dtype) == 'string':
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                )
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        except:
            df[col] = df[col].fillna('Unknown')

    # Encode target column
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

    # Encode all remaining categorical columns
    encoders = {}
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders, target_encoder, drop_cols

def train_on_any_dataset(df, target_col):
    df_processed, encoders, target_encoder, drop_cols = auto_preprocess(df, target_col)

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only if enough samples
    min_class = y_train.value_counts().min()
    if min_class >= 6:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    model_data = {
        'model': model,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'features': list(X.columns),
        'target_col': target_col,
        'drop_cols': drop_cols,
        'accuracy': report['accuracy'],
        'auc': auc,
        'classes': list(target_encoder.classes_)
    }

    os.makedirs('model', exist_ok=True)
    with open('model/universal_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    return model_data
