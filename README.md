# Adult Census Income Classification — ML Dashboard

## Problem Statement

Predict whether an individual's annual income exceeds **$50,000/year** based on demographic and employment-related features from the 1994 U.S. Census Bureau database. This is a **binary classification** problem where the goal is to classify individuals into two income brackets: `<=50K` and `>50K`.

Accurate income prediction can assist in policy-making, targeted marketing, financial risk assessment, and socioeconomic research.

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Name** | Adult Census Income (Census Income) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) |
| **Instances** | 30,162 (after removing missing values) |
| **Features** | 14 (6 numerical + 8 categorical) |
| **Target** | `income` — Binary: `<=50K` or `>50K` |
| **Missing Values** | ~7% rows had `?` values (removed) |
| **Class Distribution** | ~76% `<=50K`, ~24% `>50K` (imbalanced) |

### Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | age | Numerical | Age of the individual |
| 2 | workclass | Categorical | Employment type (Private, Govt, Self-emp, etc.) |
| 3 | fnlwgt | Numerical | Census sampling weight |
| 4 | education | Categorical | Highest education level |
| 5 | education_num | Numerical | Education level (numeric) |
| 6 | marital_status | Categorical | Marital status |
| 7 | occupation | Categorical | Occupation type |
| 8 | relationship | Categorical | Relationship in household |
| 9 | race | Categorical | Race |
| 10 | sex | Categorical | Gender |
| 11 | capital_gain | Numerical | Capital gains |
| 12 | capital_loss | Numerical | Capital losses |
| 13 | hours_per_week | Numerical | Average hours worked per week |
| 14 | native_country | Categorical | Country of origin |

### Preprocessing
- Rows with missing values (`?`) were removed
- Categorical features encoded using **LabelEncoder**
- All features scaled using **StandardScaler**
- 80/20 stratified train-test split (random_state=42)

---

## Models Used

Six classification models were implemented and evaluated on the same dataset:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8175 | 0.8501 | 0.8060 | 0.8175 | 0.8018 | 0.4613 |
| Decision Tree | 0.8510 | 0.8848 | 0.8448 | 0.8510 | 0.8453 | 0.5794 |
| K-Nearest Neighbors | 0.8190 | 0.8498 | 0.8133 | 0.8190 | 0.8154 | 0.4993 |
| Naive Bayes (Gaussian) | 0.7978 | 0.8498 | 0.7830 | 0.7978 | 0.7697 | 0.3798 |
| Random Forest (Ensemble) | 0.8541 | 0.9024 | 0.8489 | 0.8541 | 0.8500 | 0.5927 |
| XGBoost (Ensemble) | 0.8616 | 0.9204 | 0.8567 | 0.8616 | 0.8574 | 0.6131 |

### Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves 81.75% accuracy as a solid baseline. Being a linear model, it struggles to capture non-linear relationships in the data (MCC = 0.4613), particularly for predicting the minority class (>50K). AUC of 0.8501 shows reasonable discriminative ability. It is fast to train and highly interpretable. |
| Decision Tree | Significantly outperforms Logistic Regression and KNN with 85.10% accuracy and MCC of 0.5794. With max_depth=10, it captures complex feature interactions (e.g., education + occupation). AUC of 0.8848 indicates strong class separation. However, single trees can overfit on training data. |
| K-Nearest Neighbors | Performs comparably to Logistic Regression (81.90% accuracy) but is computationally much more expensive during prediction due to distance calculations over ~24K training instances. MCC of 0.4993 is moderate. Performance is sensitive to the number of neighbors and feature scaling. |
| Naive Bayes (Gaussian) | Lowest performer at 79.78% accuracy and 0.3798 MCC. The conditional independence assumption is violated by correlated features (e.g., education and education_num, occupation and workclass). Despite this, it achieves a competitive AUC of 0.8498, suggesting its probability estimates have some discriminative power even if the hard predictions are less accurate. |
| Random Forest (Ensemble) | Second-best performer with 85.41% accuracy and the highest non-XGBoost AUC of 0.9024. Aggregating 100 decision trees reduces overfitting and variance compared to a single Decision Tree. MCC of 0.5927 reflects strong balanced performance across both income classes. Robust to outliers and can handle mixed feature types well. |
| XGBoost (Ensemble) | Best overall performer across all metrics — highest accuracy (86.16%), AUC (0.9204), precision (0.8567), recall (0.8616), F1 (0.8574), and MCC (0.6131). Gradient boosting sequentially corrects errors from previous trees, effectively learning complex patterns. Particularly strong at handling the class imbalance (76% vs 24%) compared to other models. |

---

## Project Structure

```
ml-classification-app/
│── app.py                          # Streamlit web application
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation
│── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
│── model/
│   ├── train_models.py             # Model training and evaluation script
│   ├── trained_models.pkl          # Saved trained models and results
│   └── test_data.csv               # Sample test data for upload testing
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone <https://github.com/baseeraiml-dev/MlAppClassification>
cd ml-classification-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (generates model/trained_models.pkl)
```bash
python model/train_models.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Streamlit App Features

- **Model Comparison Table**: Side-by-side metrics for all 6 models with visual highlighting
- **Visual Charts**: Bar charts comparing each metric across models
- **Model Details**: Detailed view with confusion matrix and classification report for the selected model
- **CSV Upload**: Upload test data to get predictions and evaluation metrics
- **Model Selection**: Dropdown to switch between all 6 trained models
- **Dataset Info**: Complete dataset description and feature statistics

---

## Deployment

Deployed on **Streamlit Community Cloud**.

**Live App Link:** https://mlappclassification-kcht6etxwpn3xh7dwtydxx.streamlit.app/

**GitHub Repository:** https://github.com/baseeraiml-dev/MlAppClassification

---

## Technologies Used

- Python 3.x
- Streamlit
- scikit-learn
- XGBoost
- pandas, NumPy
- matplotlib, seaborn
- joblib
