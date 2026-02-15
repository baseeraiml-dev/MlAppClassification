
""""
ML Assignment 2 2025AA05593
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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


# ─────────────────────────────────────────────
# Load Pre-trained Models
# ─────────────────────────────────────────────
def load_saved_models():
    """Load pre-trained models and evaluation results."""
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'trained_models.pkl')
    if not os.path.exists(model_path):
        print("ERROR: Model file not found! Please run `model/train_models.py` first.")
        return None
    data = joblib.load(model_path)
    return data


print("=" * 60)
print("  Income Classification - Debug Script (No Streamlit)")
print("=" * 60)

data = load_saved_models()
if data is None:
    exit(1)

models = data['models']
results = data['results']
conf_matrices = data['confusion_matrices']
class_reports = data['classification_reports']
scaler = data['scaler']
label_encoders = data['label_encoders']
feature_names = data['feature_names']
categorical_cols = data['categorical_cols']
numerical_cols = data['numerical_cols']
X_test_default = data['X_test']
y_test_default = data['y_test']

print(f"\nLoaded {len(models)} models: {list(models.keys())}")
print(f"Feature names ({len(feature_names)}): {feature_names}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")
print(f"Test set shape: X={X_test_default.shape}, y={y_test_default.shape}")


# ─────────────────────────────────────────────
# 1. Model Comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  1. MODEL PERFORMANCE COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model"
print("\n", results_df.to_string(float_format=lambda x: f"{x:.4f}"))

# Visual comparison - bar charts
model_names = list(models.keys())
metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    values = []
    labels = []
    for name in model_names:
        v = results[name].get(metric, 0)
        values.append(v if isinstance(v, (int, float)) else 0)
        labels.append(name.replace(' (Ensemble)', '\n(Ensemble)').replace(' (Gaussian)', '\n(Gaussian)'))

    bars = ax.barh(labels, values, color=colors)
    ax.set_title(metric, fontweight='bold', fontsize=12)

    max_val = max(values) if values else 1
    ax.set_xlim(0, max(max_val * 1.15, 0.1))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8
        )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('debug_model_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: debug_model_comparison.png")
plt.show()


# ─────────────────────────────────────────────
# 2. Individual Model Details
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  2. INDIVIDUAL MODEL DETAILS")
print("=" * 60)

for selected_model in model_names:
    print(f"\n{'─' * 50}")
    print(f"  Model: {selected_model}")
    print(f"{'─' * 50}")

    # Metrics
    metrics = results[selected_model]
    for metric_name, value in metrics.items():
        disp = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        print(f"  {metric_name}: {disp}")

    # Confusion Matrix
    cm = conf_matrices[selected_model]
    print(f"\n  Confusion Matrix:\n{cm}")

    # Classification Report
    report = class_reports[selected_model]
    report_df = pd.DataFrame(report).T
    rename_map = {'0': '<=50K (0)', '1': '>50K (1)'}
    report_df.index = [rename_map.get(str(idx), idx) for idx in report_df.index]
    print(f"\n  Classification Report:\n{report_df.to_string(float_format=lambda x: f'{x:.4f}')}")

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f"\n  Normalized Confusion Matrix (%):\n{np.array2string(cm_norm, precision=1)}")

# Plot confusion matrices for all models
n_models = len(model_names)
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
fig_cm.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

for idx, name in enumerate(model_names):
    ax = axes_cm[idx // 3, idx % 3]
    cm = conf_matrices[name]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['<=50K', '>50K'],
        yticklabels=['<=50K', '>50K'],
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(name, fontweight='bold', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('debug_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\nSaved: debug_confusion_matrices.png")
plt.show()


# ─────────────────────────────────────────────
# 3. Upload & Predict (using test_data.csv)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  3. PREDICT ON TEST DATA (model/test_data.csv)")
print("=" * 60)

test_csv_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data.csv')

if os.path.exists(test_csv_path):
    upload_df = pd.read_csv(test_csv_path)
    print(f"\nFile loaded: {test_csv_path}")
    print(f"Shape: {upload_df.shape}")
    print(f"\nData Preview (first 5 rows):\n{upload_df.head().to_string()}")
    print(f"\nColumn dtypes:\n{upload_df.dtypes}")
    print(f"\nNull counts:\n{upload_df.isnull().sum()}")

    # Check for target column
    has_target = 'income' in upload_df.columns
    print(f"\nHas target column ('income'): {has_target}")

    if has_target:
        y_raw = upload_df['income']
        print(f"Income value counts:\n{y_raw.value_counts()}")
        if pd.api.types.is_numeric_dtype(y_raw):
            y_upload = y_raw.astype(int)
        else:
            y_upload = (y_raw.astype(str).str.strip().str.rstrip('.') == '>50K').astype(int)
        X_upload = upload_df.drop('income', axis=1)
    else:
        X_upload = upload_df.copy()
        y_upload = None

    # Verify columns
    missing_cols = set(feature_names) - set(X_upload.columns)
    extra_cols = set(X_upload.columns) - set(feature_names)

    if missing_cols:
        print(f"\nERROR: Missing columns: {missing_cols}")
        print(f"Expected columns: {feature_names}")
    else:
        if extra_cols:
            print(f"\nWARNING: Extra columns ignored: {extra_cols}")

        # Select and reorder columns
        X_upload = X_upload[feature_names]

        # Encode categorical variables
        for col in categorical_cols:
            if col in X_upload.columns and X_upload[col].dtype == 'object':
                le = label_encoders[col]
                known_classes = set(le.classes_)
                unknown_vals = set(X_upload[col].unique()) - known_classes
                if unknown_vals:
                    print(f"  WARNING: Unknown values in '{col}': {unknown_vals} -> mapped to '{le.classes_[0]}'")
                X_upload[col] = X_upload[col].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                X_upload[col] = le.transform(X_upload[col])

        # Scale features
        X_upload_scaled = scaler.transform(X_upload)
        print(f"\nScaled features shape: {X_upload_scaled.shape}")
        print(f"Scaled features sample (first row):\n{X_upload_scaled[0]}")

        # Predict with ALL models
        print(f"\n{'─' * 50}")
        print("  Predictions from ALL models")
        print(f"{'─' * 50}")

        for selected_model in model_names:
            model = models[selected_model]
            predictions = model.predict(X_upload_scaled)
            pred_labels = ['<=50K' if p == 0 else '>50K' for p in predictions]
            pred_counts = pd.Series(pred_labels).value_counts()

            print(f"\n  [{selected_model}]")
            print(f"    Prediction distribution: {dict(pred_counts)}")

            if has_target and y_upload is not None:
                if hasattr(model, 'predict_proba'):
                    y_proba_up = model.predict_proba(X_upload_scaled)[:, 1]
                    auc_up = roc_auc_score(y_upload, y_proba_up)
                else:
                    auc_up = None

                up_metrics = {
                    'Accuracy': accuracy_score(y_upload, predictions),
                    'AUC': auc_up if auc_up is not None else 'N/A',
                    'Precision': precision_score(y_upload, predictions, average='weighted', zero_division=0),
                    'Recall': recall_score(y_upload, predictions, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_upload, predictions, average='weighted', zero_division=0),
                    'MCC': matthews_corrcoef(y_upload, predictions)
                }
                for m_name, m_val in up_metrics.items():
                    disp_u = f"{m_val:.4f}" if isinstance(m_val, (int, float)) else str(m_val)
                    print(f"    {m_name}: {disp_u}")

                # Confusion matrix
                cm_up = confusion_matrix(y_upload, predictions)
                print(f"    Confusion Matrix:\n{cm_up}")

                # Classification report
                report_up = classification_report(y_upload, predictions, zero_division=0)
                print(f"    Classification Report:\n{report_up}")

else:
    print(f"\nWARNING: Test file not found at {test_csv_path}")
    print("Skipping prediction step. Place a test CSV in model/test_data.csv")


# ─────────────────────────────────────────────
# 4. Dataset Statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  4. TEST SET FEATURE STATISTICS (Scaled)")
print("=" * 60)

stats_df = pd.DataFrame(X_test_default, columns=feature_names).describe()
print(f"\n{stats_df.to_string(float_format=lambda x: f'{x:.3f}')}")


print("\n" + "=" * 60)
print("  Debug script complete!")
print("=" * 60)