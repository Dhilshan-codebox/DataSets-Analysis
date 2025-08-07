import streamlit as st
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# Load the trained model and scaler
model = joblib.load('cardio_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/cardio_db"
client = MongoClient(MONGO_URI)
db = client['cardio_db']
collection = db['risk_predictions']

st.title("CardioRiskPredict: Cardiovascular Risk Prediction for Diabetics")
st.write("Enter patient data to assess cardiovascular risk.")

# Input form
with st.form("risk_form"):
    name = st.text_input("Enter Patient Name")
    age = st.slider("Age", 20, 90, 50)
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
    bp = st.slider("Blood Pressure (Systolic)", 80, 200, 120)
    glucose = st.slider("Blood Glucose Level", 70, 250, 100)
    cholesterol = st.slider("Cholesterol Level", 100, 350, 200)
    smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
    diabetes_duration = st.slider("Duration of Diabetes (Years)", 0, 30, 5)
    
    submitted = st.form_submit_button("Predict Risk")

# Preprocess input and predict
if submitted:
    smoking_val = 1 if smoking == "Smoker" else 0
    features = np.array([[age, bmi, bp, glucose, cholesterol, smoking_val, diabetes_duration]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    risk_label = "High Risk" if prediction[0] == 1 else "Low Risk"

    # Display result
    st.write(f"### Hello, {name}!")
    if prediction[0] == 1:
        st.error("You are at **High Risk** of Cardiovascular Disease ⚠️")
    else:
        st.success("You are at **Low Risk** of Cardiovascular Disease ✅")

    # Store in MongoDB
    record = {
        "name": name,
        "age": age,
        "bmi": bmi,
        "blood_pressure": bp,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "smoking": smoking,
        "diabetes_duration": diabetes_duration,
        "risk": risk_label,
        "timestamp": datetime.now()
    }
    collection.insert_one(record)
    st.info("Patient data and prediction saved to database 🗂️")
