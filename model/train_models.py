""""
Training for classification Models
===============================================
Dataset : Adult Census Income (UCL ML Repo)
URL : https://archive.ics.uci.edu/datasets/2/adult

Features: 14(6numerical, 8 categorical)
Instances: ~30,162(after cleaning)
Task Binary Classification (Income >50k or <=50k)
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the Adult Census Income dataset from UCI ML Repository."""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income'
    ]

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    print("Loading dataset from UCI ML Repository...")
    df = pd.read_csv(
        url,
        names=column_names,
        skipinitialspace=True,
        na_values='?'
    )

    print(f"Raw dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}\n")

    # Drop rows with missing values
    df = df.dropna()
    print(f"Dataset shape after removing missing values: {df.shape}")
    print(f"Number of features: {len(column_names) - 1}")
    print(f"\nTarget distribution:\n{df['income'].value_counts()}\n")
    return df

def preprocess_and_split(df):
    """Preprocess the dataset and split into train/test sets."""
    df = df.copy()

    # Encode target variable (handle both 'adult.data' and 'adult.test' formats)
    df['income'] = df['income'].str.strip().str.rstrip('.')
    df['income'] = (df['income'] == '>50K').astype(int)

    # Separate features and target
    X = df.drop('income', axis=1)
    y = df['income']

    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save raw test data CSV for Streamlit upload testing
    test_raw = X_test.copy()
    test_raw['income'] = y_test.values
    # Convert back to string labels for the raw CSV
    test_raw['income'] = test_raw['income'].map({1: '>50K', 0: '<=50K'})

    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    # Encode categorical variables (fit on train, transform both)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    # Scale features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = list(X.columns)

    return (X_train_scaled, X_test_scaled, y_train.values, y_test.values,
            scaler, label_encoders, feature_names,
            categorical_cols, numerical_cols, test_raw)


def get_models():
    """Return a dictionary of classification models to train."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=10
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, algorithm='kd_tree', n_jobs=-1
        ),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=100, random_state=42, eval_metric='logloss'
        )
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics, confusion matrix, and report."""
    y_pred = model.predict(X_test)

    # Probability predictions for AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(auc, 4) if auc is not None else 'N/A',
        'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return metrics, cm, report


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Train all models and compute evaluation metrics."""
    models_dict = get_models()

    all_results = {}
    trained_models = {}
    all_confusion_matrices = {}
    all_reports = {}

    for name, model in models_dict.items():
        print(f"\n{'-' * 50}")
        print(f"  Training: {name}")
        print(f"{'-' * 50}")

        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate
        metrics, cm, report = evaluate_model(model, X_test, y_test)

        all_results[name] = metrics
        all_confusion_matrices[name] = cm
        all_reports[name] = report

        # Print metrics
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric_name}: {value:.4f}")
            else:
                print(f"    {metric_name}: {value}")

    return trained_models, all_results, all_confusion_matrices, all_reports


def main():
    print("=" * 60)
    print("  ML Classification Model Training Pipeline")
    print("  Dataset: Adult Census Income (UCI)")
    print("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess and split
    (X_train, X_test, y_train, y_test,
     scaler, label_encoders, feature_names,
     categorical_cols, numerical_cols, test_raw) = preprocess_and_split(df)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Step 3: Train and evaluate all models
    trained_models, results, conf_matrices, class_reports = train_and_evaluate_all(
        X_train, X_test, y_train, y_test
    )

    # Step 4: Print comparison table
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'
    print(results_df.to_string())

    # Step 5: Save everything
    save_data = {
        'models': trained_models,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'results': results,
        'confusion_matrices': conf_matrices,
        'classification_reports': class_reports,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'X_test': X_test,
        'y_test': y_test
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save trained models and results
    model_path = os.path.join(script_dir, 'trained_models.pkl')
    joblib.dump(save_data, model_path)
    print(f"\nAll models and results saved to: {model_path}")

    # Save test data CSV for upload testing
    test_csv_path = os.path.join(script_dir, 'test_data.csv')
    test_raw.to_csv(test_csv_path, index=False)
    print(f"Test data saved to: {test_csv_path}")

    # Print file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")

    print(f"\n{'=' * 60}")
    print("  Training complete! Ready for Streamlit deployment.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


