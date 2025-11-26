---
title: Diabetes Risk Prediction Application
emoji: 🩺
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: app_v1.py
pinned: false
---

# 🩺 Diabetes Risk Prediction Application  
An end-to-end **AI-driven clinical analytics solution** for exploring health indicators and predicting diabetes risk based on lifestyle, demographic, and biometric factors.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Framework-red.svg)](https://streamlit.io/)
[![Tableau](https://img.shields.io/badge/Tableau-Analytics%20Dashboards-orange.svg)](https://tableau.com)
[![TabPy](https://img.shields.io/badge/TabPy-Model%20Deployment-green.svg)](https://github.com/tableau/TabPy)
[![License](https://img.shields.io/badge/License-Apache-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)

---

## 🚀 Overview

The application provides an interactive, user-friendly interface for predicting diabetes risk based on key medical, demographic, and lifestyle indicators. Users can input values such as HbA1c, glucose levels, BMI, blood pressure, age, and activity levels, and the app instantly communicates with a deployed machine learning model to classify whether the user is at risk of diabetes. The app also displays the predicted probability of risk, making the output easy to interpret for both clinical and non-clinical users. Designed with clean navigation and responsive layout, the tool serves as a practical decision-support system for health awareness, early detection, and proactive management.

The **Diabetes Risk Prediction Application** combines:
- Data analytics  
- Machine learning  
- Clinical risk scoring  
- Tableau interactive dashboards  
- Python-based predictive modeling  

…to provide clinicians, analysts, and researchers with a powerful tool for **diabetes monitoring and early detection**.

The system analyzes:
- **Demographics**
- **Lifestyle behaviors**
- **Health history**
- **Biometric indicators**
- **Glucose/HbA1c measurements**

A deployed ML model (via **TabPy**) allows **real-time diabetes risk prediction** directly inside Tableau dashboards.

---

## Data Sources 📊
The dataset contains 31 columns and 100,000 individual patient records, where each record represents a unique health profile. Every profile includes a binary diabetes diagnosis (Yes/No) and a risk score, indicating the likelihood of being diabetic. This comprehensive structure enables both individual-level and population-level analysis, supporting detailed exploration of health patterns, risk correlations, and predictive factors across diverse patient demographics. 
https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset

## Model Evaluation 🧠📉
Model Development: Two machine learning models explored (Python Notebook attached):
•	K-Nearest Neighbors (KNN)
•	Random Forest Classifier
 
Best Model: Random Forest 
•	Accuracy: 92%
•	AUC Score: 94%
•	F1 Score: 93% (Positive prediction), 92% (Negative prediction 

## 🔍 Key Features

### 🧬 **1. Advanced Exploratory Analytics**
- Age-based diabetes prevalence  
- BMI and diet score influence  
- Sleep vs. screen time behavior  
- Cardiovascular & hypertension profiles  
- Clinical biomarker relationships (HbA1c, fasting glucose, insulin)

### 📊 **2. Tableau Dashboards**
- Risk Distribution Dashboard  
- Clinical Deep-Dive Dashboard  
- Lifestyle Impact Dashboard  
- Manual & automatic segmentation (Clustering)

### 🤖 **3. Real-Time ML Model Integration (TabPy)**
Predicts:
- **Diabetes Yes/No**
- **Risk Level (Low / Moderate / High)**
- **Confidence score**

Model uses:
- HbA1c  
- Glucose (fasting + postprandial)  
- BMI  
- Age  
- Physical activity  
- Family history  
- Systolic blood pressure  

### 🔮 **4. Automated Clinical Insights**
- Trend line analysis  
- Risk correlation summaries  
- Cluster profiling  
- Biomarker relationships  

---

## Getting Started 🏁
### Clone the repository 🧬
git clone https://github.com/kaisarhossain/Diabetes-Risk-Prediction.git

## 🧩 Project Structure

Diabetes-Risk-Prediction/
│
├── app_v1.py # Streamlit application
├── diabetes_model.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── TabPy/
│ ├── deploy_model.py # Script to publish model to TabPy
│ └── setup_instructions.txt
├── data/
│ ├── diabetes_dataset.csv
├── dashboards/
│ ├── Diabetes_Risk.twbx # Tableau dashboard
└── .gitignore


---

## ⚙️ Installation & Setup

### 1️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run Streamlit App
streamlit run app_v1.py

### TabPy Model Deployment
tabpy

### Deploy your ML model
python TabPy/deploy_model.py

## 🧾 Analytics Covered
📌 1. Demographics
Age
Gender
Ethnicity

📌 2. Lifestyle Behaviors
Diet score
Screen time
Sleep hours
Smoking / Alcohol
Physical activity

📌 3. Medical History
Family history
Hypertension
Cardiovascular disorders

📌 4. Clinical Biomarkers
HbA1c
Glucose (Fasting & Postprandial)
Insulin
Cholesterol panel
Systolic / Diastolic BP

## 📈 Example Insights Generated
Higher BMI groups show increased diabetes prevalence
HbA1c sharply rises with obesity categories
Sedentary individuals (high screen time, low sleep) show higher risk
Insulin and fasting glucose demonstrate strong positive correlation
Combining Age + BMI + HbA1c forms strong predictor of risk clusters

## 💡 Future Enhancements
AR-based health visualization
Automated data ingestion pipeline
Deep learning risk model (LSTM for time series)
EHR integration
Patient-level progress tracking

## 👨‍💻 Author
Mohammed Golam Kaisar Hossain Bhuyan
AI | ML | Deep Learning | Data Analytics
🔗 LinkedIn: https://www.linkedin.com/in/kaisarhossain
🔗 GitHub: https://github.com/kaisarhossain


## 🪪 License
Licensed under Apache 2.0 License.
You are free to use, modify, and distribute.