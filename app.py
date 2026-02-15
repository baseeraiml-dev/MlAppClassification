
""""
ML Assignment 2 2025AA05593 

Streamlit Web Application - ML Classification Dashboard
====================================================================
Dataset : Adult Census Income (UCI ML repo)
Task : Binary classification (Income >50K or <=50k)
Models : Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""
import streamlit as st
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
# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Income Classification - ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Pre-trained Models
# ─────────────────────────────────────────────
@st.cache_resource
def load_saved_models():
    """Load pre-trained models and evaluation results."""
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'trained_models.pkl')
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please run `model/train_models.py` first.")
        return None
    data = joblib.load(model_path)
    return data


			   
															  
			   

data = load_saved_models()
if data is None:
    st.stop()

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

															  
															   
												 
											 
																			


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📊 Income Classification Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">'
    'Predicting whether annual income exceeds $50K based on census data '
    '| 6 ML Models Compared'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("---")

									
								  
																	

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# Model Selection Dropdown
model_names = list(models.keys())
selected_model = st.sidebar.selectbox(
    "🤖 Select ML Model",
    model_names,
    help="Choose a classification model to view its detailed performance"
)

# File Upload
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help=(
        "Upload a CSV with the same feature columns as the training data. "
        "Include an 'income' column (values: '>50K' or '<=50K') to compute "
        "evaluation metrics. A sample file is in the model/ directory."
    )
)
# Download sample test data
test_data_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data.csv')
if os.path.exists(test_data_path):
    with open(test_data_path, 'rb') as f:
        st.sidebar.download_button(
            label="⬇️ Download Sample Test CSV",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
            help="Download the sample test_data.csv to try the upload feature"
        )

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Adult Census Income (UCI)\n\n"
    "**Task:** Binary Classification\n\n"
    "**Features:** 14\n\n"
    "**Instances:** 30,162+"
)
																										 

												
														

# ─────────────────────────────────────────────
# Main Content Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🔍 Model Details",
    "📁 Upload & Predict",
    "📋 Dataset Info"
])

									  
				
											 
											   
												 
		 

# ═══════════════════════════════════════════════
# Tab 1: Model Comparison
# ═══════════════════════════════════════════════
with tab1:
    st.header("Model Performance Comparison")
    st.markdown("All 6 classification models evaluated on the same test set.")

    # Comparison table
    results_df = pd.DataFrame(results).T
    results_df.index.name = "ML Model"

    st.dataframe(
        results_df.style
            .highlight_max(axis=0, color='#90EE90')
            .highlight_min(axis=0, color='#FFB6C1')
            .format(precision=4),
        use_container_width=True,
        height=280
    )
    st.caption("🟢 Green = Best  |  🔴 Pink = Worst for each metric")

    # Visual comparison - bar charts
    st.subheader("Visual Comparison")
									   
						  

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
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# Tab 2: Selected Model Details
# ═══════════════════════════════════════════════
with tab2:
    st.header(f"Model Details: {selected_model}")

    # Metrics row
    st.subheader("Evaluation Metrics")
    metrics = results[selected_model]
    metric_icons = ['🎯', '📈', '✅', '🔄', '⚖️', '📐']
																				
										 

    cols = st.columns(6)
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            disp = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            st.metric(label=f"{metric_icons[i]} {metric_name}", value=disp)

    st.markdown("---")

    # Confusion Matrix + Classification Report side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = conf_matrices[selected_model]

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'],
            ax=ax_cm,
            linewidths=0.5
        )
        ax_cm.set_xlabel('Predicted Label', fontsize=12)
        ax_cm.set_ylabel('True Label', fontsize=12)
        ax_cm.set_title(f'Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col_right:
        st.subheader("Classification Report")
        report = class_reports[selected_model]
        report_df = pd.DataFrame(report).T

        # Rename index for clarity
        rename_map = {'0': '<=50K (0)', '1': '>50K (1)'}
        report_df.index = [rename_map.get(str(idx), idx) for idx in report_df.index]

        st.dataframe(
            report_df.style.format(precision=4),
            use_container_width=True,
            height=250
        )

    # Normalized confusion matrix
    st.subheader("Normalized Confusion Matrix (%)")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
																						   

										
						   
    fig_cmn, ax_cmn = plt.subplots(figsize=(6, 5))
																				  

										
								   
							
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='YlOrRd',
        xticklabels=['<=50K', '>50K'],
        yticklabels=['<=50K', '>50K'],
        ax=ax_cmn,
        linewidths=0.5
    )
    ax_cmn.set_xlabel('Predicted Label', fontsize=12)
    ax_cmn.set_ylabel('True Label', fontsize=12)
    ax_cmn.set_title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cmn)
    plt.close()

									  
																		 
											  
		  

# ═══════════════════════════════════════════════
# Tab 3: Upload & Predict
# ═══════════════════════════════════════════════
with tab3:
    st.header("Upload Data & Get Predictions")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {upload_df.shape}")
														
			   

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(upload_df.head(10), use_container_width=True)

            # Check for target column
            has_target = 'income' in upload_df.columns
											
									  
																			
												  
														

            if has_target:
                y_raw = upload_df['income']
                if y_raw.dtype == 'object':
                    y_upload = (y_raw.str.strip().str.rstrip('.') == '>50K').astype(int)
                else:
                    y_upload = y_raw.astype(int)
                X_upload = upload_df.drop('income', axis=1)
            else:
                X_upload = upload_df.copy()
                y_upload = None

            # Verify columns
								   
															  
												
            missing_cols = set(feature_names) - set(X_upload.columns)
            extra_cols = set(X_upload.columns) - set(feature_names)
												
										
			 
																							
												   
		 
								   
					   

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.info(f"Expected columns: {feature_names}")
            else:
                if extra_cols:
                    st.warning(f"⚠️ Extra columns ignored: {extra_cols}")

                # Select and reorder columns
                X_upload = X_upload[feature_names]

                # Encode categorical variables
                for col in categorical_cols:
                    if col in X_upload.columns and X_upload[col].dtype == 'object':
                        le = label_encoders[col]
                        known_classes = set(le.classes_)
                        X_upload[col] = X_upload[col].apply(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        X_upload[col] = le.transform(X_upload[col])

                # Scale features
                X_upload_scaled = scaler.transform(X_upload)

                # Predict with selected model
                model = models[selected_model]
                predictions = model.predict(X_upload_scaled)

                st.subheader(f"Predictions using **{selected_model}**")
                pred_labels = ['<=50K' if p == 0 else '>50K' for p in predictions]
                result_df = upload_df.copy()
                result_df['Predicted_Income'] = pred_labels
                st.dataframe(result_df, use_container_width=True)

                # Prediction distribution
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Prediction Distribution")
                    pred_counts = pd.Series(pred_labels).value_counts()
                    fig_pd, ax_pd = plt.subplots(figsize=(5, 4))
                    pred_counts.plot(
                        kind='bar', color=['#2ca02c', '#d62728'],
                        ax=ax_pd, edgecolor='black'
                    )
                    ax_pd.set_title('Prediction Distribution', fontweight='bold')
                    ax_pd.set_ylabel('Count')
                    ax_pd.set_xticklabels(ax_pd.get_xticklabels(), rotation=0)
                    for i, (idx_val, val) in enumerate(pred_counts.items()):
                        ax_pd.text(i, val + 0.5, str(val), ha='center', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_pd)
                    plt.close()

                # Metrics if ground truth available
                if has_target and y_upload is not None:
                    st.markdown("---")
                    st.subheader("📊 Evaluation on Uploaded Data")

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

                    cols_up = st.columns(6)
                    for i, (m_name, m_val) in enumerate(up_metrics.items()):
                        with cols_up[i]:
                            disp_u = f"{m_val:.4f}" if isinstance(m_val, (int, float)) else str(m_val)
                            st.metric(m_name, disp_u)

                    # Confusion matrix for uploaded data
                    with col_b:
                        st.subheader("Confusion Matrix")
                        cm_up = confusion_matrix(y_upload, predictions)
                        fig_cmu, ax_cmu = plt.subplots(figsize=(5, 4))
                        sns.heatmap(
                            cm_up, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['<=50K', '>50K'],
                            yticklabels=['<=50K', '>50K'],
                            ax=ax_cmu, linewidths=0.5
                        )
                        ax_cmu.set_xlabel('Predicted')
                        ax_cmu.set_ylabel('Actual')
                        ax_cmu.set_title('Confusion Matrix (Uploaded Data)', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_cmu)
                        plt.close()

                    # Classification report
                    st.subheader("Classification Report (Uploaded Data)")
                    report_up = classification_report(
                        y_upload, predictions, output_dict=True, zero_division=0
                    )
                    report_up_df = pd.DataFrame(report_up).T
                    rename_map = {'0': '<=50K (0)', '1': '>50K (1)'}
                    report_up_df.index = [rename_map.get(str(idx), idx) for idx in report_up_df.index]
                    st.dataframe(
                        report_up_df.style.format(precision=4),
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure the CSV has the correct format and columns.")

    else:
        st.info("👈 Upload a CSV file from the sidebar to see predictions.")
        st.markdown("""
        **How to use:**
        1. Upload a CSV file with the same feature columns as the training data
        2. Optionally include an `income` column (values: `>50K` or `<=50K`) for evaluation
        3. Select a model from the sidebar dropdown
        4. View predictions and evaluation metrics

        **A sample test file** (`test_data.csv`) is available in the `model/` directory of this project.
        """)

        st.markdown("**Required feature columns:**")
        cols_display = ", ".join([f"`{c}`" for c in feature_names])
																		   
										
												
																		  
								
																												  
													
																		 
        st.markdown(cols_display)
										
												
																		  
								
																												  
													
																		 
				 
														   

						
													
																  
																		   

# ═══════════════════════════════════════════════
# Tab 4: Dataset Info
# ═══════════════════════════════════════════════
with tab4:
    st.header("Dataset Information")

    st.markdown("""
    ### Adult Census Income Dataset
														
																			  
															   

    **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
																	  

    **Description:** Predict whether an individual's annual income exceeds $50K/year
    based on census data. Also known as the "Census Income" dataset.
    Extracted from the 1994 Census Bureau database.
																
					 
								 

    **Task:** Binary Classification
																	  
																   
																											 
																									   
																									 
																   
				 
														
																							  
													

    **Classes:** `<=50K` (majority) and `>50K` (minority)
															   
														

    ---
																						 
																 

    ### Features (14)
															   
																			  

    | # | Feature | Type | Description |
    |---|---------|------|-------------|
    | 1 | age | Numerical | Age of the individual |
    | 2 | workclass | Categorical | Type of employment (Private, Govt, Self-emp, etc.) |
    | 3 | fnlwgt | Numerical | Census sampling weight |
    | 4 | education | Categorical | Highest education level attained |
    | 5 | education_num | Numerical | Education level (numeric encoding) |
    | 6 | marital_status | Categorical | Marital status |
    | 7 | occupation | Categorical | Type of occupation |
    | 8 | relationship | Categorical | Relationship in household |
    | 9 | race | Categorical | Race |
    | 10 | sex | Categorical | Gender |
    | 11 | capital_gain | Numerical | Capital gains |
    | 12 | capital_loss | Numerical | Capital losses |
    | 13 | hours_per_week | Numerical | Average hours worked per week |
    | 14 | native_country | Categorical | Country of origin |

    **Target Variable:** `income` — Binary: `<=50K` or `>50K`

    ---

    ### Preprocessing Steps
    1. **Missing values:** Rows with `?` values removed (~7% of data)
    2. **Categorical encoding:** Label encoding applied to all categorical features
    3. **Feature scaling:** StandardScaler applied to all features
    4. **Train/Test split:** 80/20 stratified split (random_state=42)
    """)

    # Test set statistics
    st.subheader("Test Set Feature Statistics (Scaled)")
    stats_df = pd.DataFrame(X_test_default, columns=feature_names).describe()
    st.dataframe(stats_df.style.format(precision=3), use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.85rem;'>"
    "ML Classification Dashboard | Built with Streamlit | "
    "Dataset: Adult Census Income (UCI ML Repository)"
    "</div>",
    unsafe_allow_html=True
)


					  
								 
			   