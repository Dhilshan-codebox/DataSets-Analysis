import streamlit as st
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# Load the trained model and scaler
try:
    model = joblib.load('cardio_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error("⚠️ Error loading model or scaler files.")
    st.stop()

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/cardio_db"
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['cardio_db']
    collection = db['risk_predictions']
except Exception as e:
    st.error("⚠️ Could not connect to MongoDB.")
    st.stop()

st.title("CardioRiskPredict: Cardiovascular Risk Prediction for Diabetics")
st.write("Enter patient data to assess cardiovascular risk.")

# Input form
with st.form("risk_form"):
    name = st.text_input("Enter Patient Name")
    age = st.number_input("Age", min_value=20, max_value=90, value=50)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")
    bp = st.number_input("Blood Pressure (Systolic)", min_value=70, max_value=250, value=120)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
    diabetes_duration = st.number_input("Duration of Diabetes (Years)", min_value=0, max_value=40, value=5)

    submitted = st.form_submit_button("Predict Risk")



if submitted:
    if not name.strip():
        st.warning("Please enter the patient's name.")
    else:
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
            "name": names,
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
