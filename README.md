# 📞 Telecom Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**Prediksi pelanggan berisiko churn menggunakan Machine Learning untuk strategi retensi yang efektif**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Business Context](#-business-context)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Business Recommendations](#-business-recommendations)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

Proyek ini mengembangkan model **Machine Learning** untuk memprediksi **customer churn** di industri telekomunikasi dengan fokus utama pada **deteksi dini pelanggan berisiko tinggi**. Model ini dirancang untuk membantu perusahaan mengoptimalkan strategi retensi pelanggan dan mengurangi kerugian finansial akibat churn.

### 🏆 Key Achievements

- ✅ **Recall: 88.37%** - Mendeteksi 9 dari 10 pelanggan berisiko churn
- ✅ **ROC-AUC: 0.832** - Kemampuan diskriminasi sangat baik
- ✅ **Production Ready** - Siap diimplementasikan dengan strategi tiered retention
- ✅ **Interpretable** - Feature importance yang jelas untuk business insights

---

## 💼 Business Context

### Problem Statement

Industri telekomunikasi menghadapi tantangan besar dalam mempertahankan pelanggan:

- 📉 **Customer churn** menyebabkan penurunan recurring revenue
- 💰 **Biaya akuisisi pelanggan baru 5x lebih mahal** dibanding retensi
- 🎯 **Program retensi tidak efektif** karena diberikan secara merata tanpa targeting

### Business Goals

1. **Mengurangi tingkat customer churn** melalui deteksi dini
2. **Mengoptimalkan biaya retensi** dengan targeting yang tepat
3. **Meningkatkan ROI** kampanye retensi pelanggan
4. **Memberikan insights** untuk strategi bisnis berbasis data

### Impact Analysis

| Metrik | Nilai | Dampak Bisnis |
|--------|-------|---------------|
| **False Negative Cost** | ~$2,000+ | Kehilangan CLV + biaya akuisisi |
| **False Positive Cost** | ~$50-100 | Biaya insentif retensi |
| **Cost Ratio** | 1:20 | FN jauh lebih merugikan |

**Strategic Decision:** Model dioptimasi untuk **maximizing Recall** (minimize False Negative) karena dampak finansialnya jauh lebih besar.

---

## 📊 Dataset

### Data Source
Dataset pelanggan telekomunikasi dengan 4,930 records setelah data cleaning.

### Features Overview

**Target Variable:**
- `Churn` - Yes (26.7%) / No (73.3%)

**Categorical Features (9):**
- `Contract` - Month-to-month, One year, Two year
- `InternetService` - DSL, Fiber optic, No
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` - Yes/No
- `PaperlessBilling` - Yes/No
- `Dependents` - Yes/No

**Numerical Features (2):**
- `tenure` - Lama berlangganan (0-72 bulan)
- `MonthlyCharges` - Biaya bulanan ($19-$118)

**Engineered Features (1):**
- `TotalServices` - Agregasi jumlah layanan tambahan (0-4)

### Data Characteristics

- ⚠️ **Imbalanced Dataset** - Churn: 26.7%, Non-Churn: 73.3%
- ✅ **No Missing Values** after cleaning
- 🔧 **77 Duplicates Removed**
- 📐 **No Significant Outliers** in numerical features

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analisis distribusi target variable
- Identifikasi pola churn vs non-churn
- Feature engineering (`TotalServices`)
- Data cleaning (duplicates removal)

### 2. Feature Selection
- **Chi-Square Test** untuk fitur kategorikal
- **ANOVA Test** untuk fitur numerikal
- **Result:** Semua fitur signifikan (p-value < 0.05)

### 3. Data Preprocessing
- **Encoding:**
  - OneHot Encoding untuk categorical features
  - Ordinal Encoding untuk Contract (ordered)
- **Scaling:** MinMaxScaler untuk numerical features
- **Imbalance Treatment:** SMOTE (Synthetic Minority Over-sampling Technique)

### 4. Model Development

**Algorithms Tested:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- LightGBM

**Imbalance Techniques:**
- Random Over Sampling (ROS)
- Random Under Sampling (RUS)
- SMOTE ✅ (Best performing)

**Validation:**
- Stratified 5-Fold Cross-Validation
- Focus on **Recall** as primary metric

### 5. Hyperparameter Tuning

**Method:** GridSearchCV with Stratified K-Fold

**Best Configuration:**
```python
{
    'model__C': 0.01,
    'model__penalty': 'l1',
    'model__solver': 'saga',
    'resampling__k_neighbors': 3,
    'resampling__sampling_strategy': 'auto'
}
```

**Improvement:**
- Before Tuning: 81.78% Recall
- After Tuning: 88.37% Recall
- **Gain: +6.59%**

---

## 📈 Model Performance

### Final Model: Logistic Regression + SMOTE + L1 Regularization

#### Performance Metrics (Test Set)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Recall** ⭐ | **88.37%** | Mendeteksi 228 dari 258 pelanggan churn |
| **Accuracy** | 68.80% | Prediksi benar secara keseluruhan |
| **Precision** | 45.51% | Trade-off yang dapat diterima |
| **F1-Score** | 60.08% | Keseimbangan precision-recall |
| **ROC-AUC** | 0.832 | Kemampuan diskriminasi sangat baik |

#### Confusion Matrix

|  | Predicted: Yes | Predicted: No |
|---|---|---|
| **Actual: Yes** | ✅ TP: 228 | ❌ FN: 30 |
| **Actual: No** | ⚠️ FP: 273 | ✅ TN: 455 |

**Key Insights:**
- ✅ **88.37% pelanggan churn terdeteksi** (hanya 30 yang terlewat)
- ⚠️ **273 false positives** - pelanggan loyal yang dapat insentif (acceptable trade-off)
- 💡 **Business Logic:** Biaya FP << Biaya FN (rasio ~1:20)

### ROC & PR Curves

**ROC-AUC: 0.832**
- Model memiliki 83.2% probabilitas memberi skor lebih tinggi pada pelanggan churn
- Kurva jauh di atas diagonal (random classifier)

**Average Precision: 0.624**
- Performa konsisten di berbagai threshold
- Cocok untuk imbalanced dataset

---

## 🔑 Key Insights

### Feature Importance (Top 4)

Model menggunakan **L1 Regularization** yang secara otomatis mengeliminasi fitur tidak relevan (koefisien = 0).

| Rank | Feature | Coefficient | Odds Ratio | Business Insight |
|------|---------|-------------|------------|------------------|
| 1 | **Contract** | -1.229 | 0.29 | Kontrak jangka panjang menurunkan churn hingga **71%** |
| 2 | **Fiber Optic** | +0.823 | 2.28 | Risiko churn **2x lipat** - masalah kualitas/harga |
| 3 | **Tenure** | -0.552 | 0.58 | Pelanggan lama lebih loyal, fokus pada pelanggan baru |
| 4 | **Paperless Billing** | +0.166 | 1.18 | Pelanggan digital lebih sensitif harga |

### Features Eliminated by L1

Fitur dengan **koefisien = 0** (tidak signifikan setelah regularisasi):
- Dependents, TotalServices, MonthlyCharges
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
- InternetService (No)

---

## 💻 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Clone Repository
```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
dython>=0.6.0
scipy>=1.7.0
```

---

## 🚀 Usage

### 1. Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data_telco_customer_churn.csv')

# Remove duplicates
df = df.drop_duplicates()

# Feature engineering
df = df.replace('No internet service', 'No')

# Split data
X = df.drop(columns='Churn')
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    stratify=y, 
    test_size=0.2, 
    random_state=42
)
```

### 2. Model Training
```python
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder

# Define preprocessor
num_cols = ['tenure', 'MonthlyCharges', 'TotalServices']
cat_cols = ['Dependents', 'OnlineSecurity', 'OnlineBackup', 
            'InternetService', 'DeviceProtection', 'TechSupport', 
            'PaperlessBilling']
ord_cols = ['Contract']
contract_order = [['Month-to-month', 'One year', 'Two year']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
        ('ordinal', OrdinalEncoder(categories=contract_order), ord_cols)
    ]
)

# Create pipeline with best parameters
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('resampling', SMOTE(k_neighbors=3, sampling_strategy='auto', random_state=20)),
    ('model', LogisticRegression(C=0.01, penalty='l1', solver='saga', 
                                  random_state=42, max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)
```

### 3. Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
```

### 4. Prediction on New Data
```python
# Example: Predict churn for new customer
new_customer = pd.DataFrame({
    'Dependents': ['No'],
    'tenure': [12],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['Yes'],
    'InternetService': ['Fiber optic'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'MonthlyCharges': [85.50],
    'TotalServices': [1]
})

# Predict
churn_probability = pipeline.predict_proba(new_customer)[0, 1]
churn_prediction = pipeline.predict(new_customer)[0]

print(f"Churn Probability: {churn_probability:.2%}")
print(f"Prediction: {churn_prediction}")

# Tiered retention strategy
if churn_probability > 0.8:
    print("🔴 HIGH RISK - Immediate intervention required")
elif churn_probability > 0.5:
    print("🟡 MEDIUM RISK - Proactive retention recommended")
else:
    print("🟢 LOW RISK - Standard service")
```

---

## 📁 Project Structure
```
telecom-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── data_telco_customer_churn.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   └── feature_importance.png
│   └── presentation/
│       └── churn_prediction_presentation.pdf
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Results

### Business Impact

#### 📉 Revenue Protection
- **88.37% Detection Rate** - Menyelamatkan 9 dari 10 pelanggan berisiko
- Mencegah kehilangan Customer Lifetime Value (CLV)
- Menjaga stabilitas recurring revenue

#### 💰 Cost Optimization
- **Targeted Marketing** - Promosi hanya untuk pelanggan berisiko tinggi
- **ROI Improvement** - Efisiensi biaya retensi vs broadcast promotion
- **Financial Logic** - Biaya retensi ($50-100) << Biaya akuisisi ($315)

#### 🎯 Strategic Advantages
- Data-driven decision making
- Priority list untuk tim retention
- Proactive intervention vs reactive response

### Model Comparison

| Model | Imbalance | CV Recall | Test Recall | Test Precision | Test F1 |
|-------|-----------|-----------|-------------|----------------|---------|
| Logistic Regression | SMOTE ✅ | 89.22% | **88.37%** | 45.51% | 60.08% |
| Logistic Regression | ROS | 84.15% | 83.72% | 42.18% | 56.13% |
| Random Forest | SMOTE | 82.45% | 79.84% | 48.92% | 60.74% |
| LightGBM | SMOTE | 81.33% | 78.29% | 47.35% | 58.91% |

**Winner:** Logistic Regression + SMOTE dengan L1 Regularization

---

## 💡 Business Recommendations

### A. Contract Strategy 📋

**Objective:** Migrasi pelanggan ke kontrak jangka panjang

**Actions:**
- Kampanye agresif Month-to-month → 1-2 year contract
- Insentif: diskon biaya bulanan, upgrade gratis 3 bulan
- Evaluasi struktur pricing kontrak bulanan

**Expected Impact:** Penurunan churn hingga 71%

### B. Fiber Optic Audit 🔧

**Objective:** Investigasi dan perbaikan layanan Fiber Optic

**Actions:**
- Network quality audit (downtime, kecepatan)
- Competitive pricing analysis
- Bundling dengan streaming services
- Value proposition improvement

**Rationale:** Fiber Optic users memiliki risiko churn 2x lipat

### C. Onboarding Program 📅

**Objective:** Program khusus untuk pelanggan baru

**Actions:**
- Fokus 90 hari pertama (critical period)
- Courtesy calls: minggu 1 dan bulan 1
- Fast-track technical support
- Welcome benefits

**Rationale:** Pelanggan dengan tenure rendah berisiko tertinggi

### D. Tiered Retention Strategy 🎚️

**Implementation:** Berdasarkan probabilitas churn

| Risk Level | Probability | Actions |
|------------|-------------|---------|
| 🔴 **HIGH** | > 80% | Diskon besar, teknisi prioritas, account manager |
| 🟡 **MEDIUM** | 50-80% | Bonus layanan, poin loyalitas, early renewal |
| 🟢 **LOW** | < 50% | Standard service, upselling opportunities |

### E. CRM Integration 🔄

**Technical Implementation:**
- Real-time churn scoring
- Automated alerts untuk tim retention
- Monthly model retraining
- A/B testing untuk validasi impact

---

## 🔮 Future Improvements

### Model Enhancement
- [ ] Ensemble methods (stacking/blending)
- [ ] Deep learning approaches (Neural Networks)
- [ ] Time-series analysis untuk churn patterns
- [ ] Feature expansion (customer service logs, network quality)

### Business Intelligence
- [ ] Customer segmentation analysis
- [ ] Lifetime Value (CLV) prediction
- [ ] Next-best-action recommendation system
- [ ] Churn risk dashboard (real-time)

### Technical Optimization
- [ ] Model deployment via API (FastAPI/Flask)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] MLOps pipeline (MLflow/Kubeflow)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

**Project Link:** [https://github.com/yourusername/telecom-churn-prediction](https://github.com/yourusername/telecom-churn-prediction)

---

## 🙏 Acknowledgments

- Dataset source: [Telco Customer Churn Dataset](https://drive.google.com/drive/folders/1_fR7R0srpZgnFnanbrmELgnK-xmzMAHp)
- Inspiration from industry best practices in churn prediction
- scikit-learn and imbalanced-learn communities

---

## 📚 References

1. [Handling Imbalanced Datasets in Machine Learning](https://imbalanced-learn.org/)
2. [Customer Churn Prediction: A Comprehensive Guide](https://towardsdatascience.com/)
3. [Business Metrics for Customer Retention](https://hbr.org/)
4. [Regularization in Machine Learning](https://scikit-learn.org/stable/modules/linear_model.html)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for data-driven customer retention

</div>
