# diabetes_model_rf.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Feature Engineering
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Pregnancies_Age'] = df['Pregnancies'] * df['Age']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, float('inf')],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, float('inf')],
                             labels=['Young', 'Middle', 'Old'])

    df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

    return X_resampled, y_resampled, X_test, y_test

def train_and_evaluate_model(X_resampled, y_resampled, X_test, y_test):
    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=20, cv=5, scoring='f1', random_state=42,
        verbose=1, n_jobs=-1
    )
    random_search.fit(X_resampled, y_resampled)
    best_rf = random_search.best_estimator_

    print("\n🌟 Best Parameters Found:", random_search.best_params_)

    # Training dengan model terbaik
    best_rf.fit(X_resampled, y_resampled)

    # Predict dan evaluasi
    proba = best_rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n🎯 AUC Score: {auc:.4f}")

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n✅ Threshold Otomatis Terbaik (berdasarkan F1-score): {best_threshold:.4f}")

    y_pred_best = (proba >= best_threshold).astype(int)
    print("\n📊 Classification Report (Threshold Otomatis F1):")
    print(classification_report(y_test, y_pred_best))

if __name__ == "__main__":
    X, y = load_and_prepare_data("Dataset/diabetes.csv")
    X_resampled, y_resampled, X_test, y_test = preprocess_data(X, y)
    train_and_evaluate_model(X_resampled, y_resampled, X_test, y_test)