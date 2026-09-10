# Medical Prediction System AI

A clinical-grade diagnostic dashboard engineered with Streamlit, Scikit-Learn, and Plotly. The system provides real-time physiological telemetry and calibrated machine learning inference across three core diagnostic modules: ECG arrhythmia classification, heart failure survival risk stratification, and multi-parameter chronic kidney disease (CKD) assessment.

---

## 🔗 Live Streamlit App: [medicalpredictionai.streamlit.app](https://medicalpredictionai.streamlit.app/)

---

## 👥 Project Team & Contributions

* **Ahmed El Zayat** — Chronic Kidney Disease (CKD) Dataset Curation & Renal Panel Pipeline
* **Marvel Akram** — ECG Arrhythmia Dataset & Signal Processing Model
* **Karen Hany** — Heart Failure Survival Risk Predictor
* **Mark George** — Streamlit Dashboard Engineering, Pipeline Integration & Cloud Deployment

---

## Diagnostic Modules

### 1. ECG Arrhythmia Telemetry & Classification
* **Data Standard:** Evaluates 188-point single-beat cardiac vectors aligned with the MIT-BIH Arrhythmia benchmark.
* **Target Classes:** 
  * Normal Sinus Rhythm (`N`)
  * Supraventricular Ectopic Beat (`SVEB`)
  * Ventricular Ectopic Beat (`VEB`)
  * Fusion Beat (`F`)
  * Unknown / Paced Beat (`Q`)
* **Features:** Ground-truth morphology presets, interactive lead trace visualization with deflection peak annotations, and calibrated multi-class probability distributions. Supports batch CSV upload.

### 2. Heart Failure Survival Risk Evaluator
* **Clinical Focus:** Quantitative mortality risk stratification over longitudinal observation periods.
* **Key Biomarkers:** Left ventricular ejection fraction (LVEF), serum creatinine, serum sodium, CPK, platelets, and comorbid indicators (hypertension, diabetes, anemia).
* **Inference:** True calibrated class probabilities via model `predict_proba` with dynamic continuous risk score fallbacks.

### 3. Chronic Kidney Disease (CKD) Diagnostic
* **Clinical Focus:** Comprehensive renal function evaluation across a 24-parameter diagnostic panel.
* **Lab Profile:** Urinalysis (specific gravity, albumin, pus cells), blood chemistry (urea, creatinine, hemoglobin, electrolyte panels), and systemic history.
* **Pipeline:** Automated type coercion, median/mode imputation, selective feature scaling, and categorical one-hot feature alignment.

---

## Tech Stack

* **Frontend / Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression / SVM)
* **Visualization:** Plotly Graph Objects & Plotly Express
* **Data Processing:** NumPy, Pandas
* **Model Serialization:** Joblib

---

## Repository Structure

```text
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Environment dependencies
├── Notebooks/                  # Experimental and training notebooks
│   ├── ckd_project.ipynb       # CKD model exploration & training
│   ├── ecg_project.ipynb       # ECG waveform modeling
│   ├── heart_failure.ipynb     # Heart failure survival analysis
│   └── Datasets/               # Project datasets repository
│       ├── chronic_kidney_disease_clean.arff  # CKD benchmark data
│       └── heart_failure_clinical_records_dataset.csv # Heart failure data
│       └── MITBIH ECG Dataset Link # Open link and download dataset
│       └── ckd_dataset_v2_compatible_test.csv # Extra Testing on ckd dataset
├── best_ecg_model.pkl          # Trained ECG classification artifact
├── ecg_scaler.pkl              # Signal normalizer for ECG telemetry
├── best_hf_model.pkl           # Heart failure survival risk estimator
├── hf_scaler.pkl               # Continuous feature standardizer (Heart)
├── best_ckd_model.pkl          # CKD renal panel classifier
├── ckd_numeric_imputer.pkl     # Median imputer for lab values
├── ckd_scaler.pkl              # Robust scaler for continuous renal markers
├── ckd_cat_imputer.pkl         # Categorical imputer
└── ckd_model_columns.pkl       # Feature alignment registry for CKD pipeline
