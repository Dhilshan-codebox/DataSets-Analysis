# complete_cardio_app.py (Corrected Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pymongo import MongoClient
from datetime import datetime
import os

# --- PART 1: HELPER FUNCTIONS FOR DATA & MODEL ---

def generate_data():
    """Generates a synthetic CSV dataset for training."""
    with st.spinner("Generating synthetic patient data..."):
        n_samples = 2000
        np.random.seed(42)
        data = {
            'age': np.random.randint(20, 90, n_samples),
            'bmi': np.random.uniform(18, 50, n_samples),
            'bp': np.random.randint(90, 200, n_samples),
            'glucose': np.random.randint(70, 300, n_samples),
            'cholesterol': np.random.randint(120, 350, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'diabetes_duration': np.random.randint(0, 40, n_samples)
        }
        df = pd.DataFrame(data)
        probability = (df['age']/90 + df['bmi']/50 + df['bp']/200 + df['glucose']/300 + df['cholesterol']/350 + df['smoking'] + df['diabetes_duration']/40) / 7
        df['risk'] = (probability > np.random.uniform(0.3, 0.7, n_samples)).astype(int)
        df.to_csv('cardio_data.csv', index=False)
    st.success("✅ `cardio_data.csv` generated successfully!")
    st.info(f"Generated {n_samples} samples. Here's a preview:")
    st.dataframe(df.head())

def train_model():
    """Loads data, trains an XGBoost model, and saves it."""
    if not os.path.exists('cardio_data.csv'):
        st.error("⚠️ `cardio_data.csv` not found. Please generate data first.")
        return

    with st.spinner("Training the model... This may take a moment."):
        df = pd.read_csv('cardio_data.csv')
        
        features = ['age', 'bmi', 'bp', 'glucose', 'cholesterol', 'smoking', 'diabetes_duration']
        target = 'risk'
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success("✅ Model training complete!")
        st.metric(label="Model Accuracy on Test Set", value=f"{accuracy * 100:.2f}%")
        
        report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'], output_dict=True)
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        joblib.dump(model, 'cardio_risk_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.success("✅ Model (`cardio_risk_model.pkl`) and Scaler (`scaler.pkl`) saved.")
        st.balloons()

def load_artifacts():
    """Loads the trained model and scaler from disk."""
    try:
        model = joblib.load('cardio_risk_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

def setup_mongodb():
    """Sets up and returns a connection to the MongoDB collection."""
    MONGO_URI = "mongodb://localhost:27017/cardio_db"
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client['cardio_db']
        collection = db['risk_predictions']
        return collection
    except Exception:
        st.error("⚠️ Could not connect to MongoDB. Predictions will not be saved.")
        return None

# --- PART 2: STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="CardioRiskPredict", layout="wide")

st.sidebar.title("🛠️ Admin Panel")
st.sidebar.header("Data & Model Management")

if st.sidebar.button("Step 1: Generate Sample Data"):
    generate_data()

if st.sidebar.button("Step 2: Train New Model"):
    train_model()
    
st.sidebar.markdown("---")
st.sidebar.info("First, generate data. Then, train the model. The prediction app will become active once the model is trained.")

st.title("CardioRiskPredict: Cardiovascular Risk Prediction for Diabetics")
st.write("Enter patient data to assess cardiovascular risk. If the form is disabled, please use the admin panel to train a model first.")

model, scaler = load_artifacts()
collection = setup_mongodb()

if model is None or scaler is None:
    st.warning("Model not found. Please train a model using the 'Admin Panel' in the sidebar.")
else:
    st.success("✅ Model loaded successfully. Ready for predictions.")
    with st.form("risk_form"):
        name = st.text_input("Enter Patient Name")
        age = st.number_input("Age", min_value=20, max_value=90, value=50)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")
        bp = st.number_input("Blood Pressure (Systolic)", min_value=70, max_value=250, value=120)
        glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100)
        cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
        smoking_option = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
        diabetes_duration = st.number_input("Duration of Diabetes (Years)", min_value=0, max_value=40, value=5)
        
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        if not name.strip():
            st.warning("Please enter the patient's name.")
        else:
            smoking_val = 1 if smoking_option == "Smoker" else 0
            features = np.array([[age, bmi, bp, glucose, cholesterol, smoking_val, diabetes_duration]])
            scaled_features = scaler.transform(features)
            
            prediction = model.predict(scaled_features)
            risk_proba = model.predict_proba(scaled_features)[0][1] # Probability of 'High Risk'
            risk_label = "High Risk" if prediction[0] == 1 else "Low Risk"

            st.write(f"---")
            st.write(f"### Prediction for {name}")
            
            if prediction[0] == 1:
                st.error(f"Prediction: **{risk_label}** ⚠️")
            else:
                st.success(f"Prediction: **{risk_label}** ✅")

            st.progress(int(risk_proba * 100)) # <-- THIS LINE IS FIXED
            st.write(f"Probability of High Risk: **{risk_proba*100:.2f}%**")

            if collection is not None:
                record = {
                    "name": name, "age": age, "bmi": bmi, "blood_pressure": bp, "glucose": glucose,
                    "cholesterol": cholesterol, "smoking": smoking_option, "diabetes_duration": diabetes_duration,
                    "risk_prediction": risk_label, "risk_probability": float(risk_proba), "timestamp": datetime.now()
                }
                collection.insert_one(record)
                st.info("Patient data and prediction saved to database 🗂️")