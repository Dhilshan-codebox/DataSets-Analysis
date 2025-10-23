# Data-Sets-Analysis
# 🩺 Cardiovascular Risk Prediction System

![App Preview](https://raw.githubusercontent.com/<your-username>/<repo-name>/main/preview.png)
> *(Replace the above link with your uploaded dashboard screenshot in the repository — e.g., `app_screenshot.png`)*

An **AI-powered Streamlit web application** that predicts the **risk of cardiovascular disease** using machine learning models like **XGBoost** and **TensorFlow**.  
The app provides personalized predictions, data visualizations, health advice, and feature importance analysis — all in an intuitive interface.

---

## 🚀 Features

### 💡 Core ML Features
- 🔍 **AI Prediction Engine (XGBoost + TensorFlow)** — predicts cardiovascular risk accurately.
- 🧠 **Feature Importance Chart** — highlights the top predictors influencing your health risk.
- 💾 **Save Predictions Automatically** — every user prediction is stored in `predictions_log.csv`.
- 🩸 **Health Advice Generator** — dynamic lifestyle recommendations based on your results.
- 📊 **Risk Summary Dashboard** — bar and pie charts showing high vs low risk ratios.

---

## 📊 Dashboard Visualizations

- **Feature Importance Chart:** Visualizes which factors (age, cholesterol, BP, etc.) affect risk most.
- **Risk Distribution Chart:** Displays percentage of High vs. Low risk predictions.
- **Prediction History Table:** Lists all saved predictions with timestamps and patient details.

---

## 🧠 Machine Learning Model

The model (`cardio_risk_model.pkl`) is trained using:
- **Algorithm:** XGBoost Classifier (`binary:logistic`)
- **Data Scaling:** `StandardScaler` for numerical features
- **Dataset Size:** 3000 synthetic patient records
- **Key Features:**
  - Age, Height, Weight, Gender
  - Blood Pressure (Systolic, Diastolic)
  - Cholesterol & Glucose Levels
  - Lifestyle (Smoking, Alcohol, Activity)
  - **Newly Added Features:**
    - Body Mass Index (BMI)
    - Family History
    - Diet Quality
    - Sleep Hours
    - Stress Level
    - Resting Heart Rate

---

## 🖥️ How to Run the Project

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
