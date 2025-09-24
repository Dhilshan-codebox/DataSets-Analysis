import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# -------------------------------
# Generate Synthetic Data
# -------------------------------
def generate_data():
    """Generates synthetic dataset with all 12 features."""
    st.info("Generating synthetic cardio dataset with all 12 features...")
    np.random.seed(42)
    n_samples = 3000

    df = pd.DataFrame({
        "age": np.random.randint(30*365, 70*365, n_samples),   # in days
        "height": np.random.randint(150, 200, n_samples),
        "weight": np.random.randint(50, 120, n_samples),
        "gender": np.random.choice([1, 2], n_samples),         # 1=female, 2=male
        "ap_hi": np.random.randint(90, 200, n_samples),        # systolic
        "ap_lo": np.random.randint(60, 120, n_samples),        # diastolic
        "cholesterol": np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15]),
        "gluc": np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]),
        "smoke": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "alco": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "active": np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })

    # simple probability of cardiovascular disease
    risk_factor = (
        (df["age"]/365 > 50).astype(int) +
        (df["ap_hi"] > 140).astype(int) +
        (df["cholesterol"] > 1).astype(int) +
        (df["gluc"] > 1).astype(int) +
        df["smoke"] + df["alco"] + (1-df["active"])
    )
    df["cardio"] = (risk_factor >= 3).astype(int)

    df.to_csv("cardio_data.csv", index=False)
    st.success("✅ `cardio_data.csv` generated successfully with 3000 rows.")
    st.dataframe(df.head())


# -------------------------------
# Train Model Function
# -------------------------------
def train_model(model_type="xgboost"):
    if not os.path.exists("cardio_data.csv"):
        st.error("⚠️ `cardio_data.csv` not found. Please generate data first.")
        return

    df = pd.read_csv("cardio_data.csv")

    features = ['age','height','weight','gender','ap_hi','ap_lo',
                'cholesterol','gluc','smoke','alco','active']
    target = 'cardio'

    # ensure dataset has these features
    missing = [col for col in features+[target] if col not in df.columns]
    if missing:
        st.error(f"Dataset is missing columns: {missing}")
        return

    X = df[features].copy()
    y = df[target]

    # scale continuous only
    continuous = ['age','height','weight','ap_hi','ap_lo']
    scaler = StandardScaler()
    X[continuous] = scaler.fit_transform(X[continuous])

    joblib.dump((scaler, continuous), "scaler.pkl")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "xgboost":
        with st.spinner("Training XGBoost model..."):
            model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success("✅ XGBoost training complete!")
            st.metric("Accuracy", f"{acc*100:.2f}%")
            st.dataframe(pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).transpose())

            joblib.dump(model, "cardio_risk_model.pkl")

    else:
        with st.spinner("Training TensorFlow model..."):
            model = Sequential([
                Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            st.success("✅ TensorFlow training complete!")
            st.metric("Accuracy", f"{acc*100:.2f}%")

            model.save("cardio_risk_tf_model.h5")


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="centered")
st.title("🩺 Cardiovascular Risk Prediction App")

menu = ["Generate Data","Train Model", "Predict Risk"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Generate Data":
    generate_data()

elif choice == "Train Model":
    model_type = st.radio("Choose Model Type:", ("xgboost", "tensorflow"))
    if st.button("Train Model"):
        train_model(model_type)

elif choice == "Predict Risk":
    st.header("🔮 Predict Cardiovascular Risk")

    with st.form("risk_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age (years)", 20, 90, 50) * 365
        height = st.number_input("Height (cm)", 120, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        gender = 1 if st.selectbox("Gender", ["Female","Male"])=="Female" else 2
        ap_hi = st.number_input("Systolic BP", 80, 250, 120)
        ap_lo = st.number_input("Diastolic BP", 50, 150, 80)
        cholesterol = {"Normal":1,"Above Normal":2,"Well Above Normal":3}[st.selectbox("Cholesterol",["Normal","Above Normal","Well Above Normal"])]
        gluc = {"Normal":1,"Above Normal":2,"Well Above Normal":3}[st.selectbox("Glucose",["Normal","Above Normal","Well Above Normal"])]
        smoke = 1 if st.selectbox("Smoking",["No","Yes"])=="Yes" else 0
        alco = 1 if st.selectbox("Alcohol Intake",["No","Yes"])=="Yes" else 0
        active = 1 if st.selectbox("Physical Activity",["Inactive","Active"])=="Active" else 0
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # load model
            if os.path.exists("cardio_risk_model.pkl"):
                model = joblib.load("cardio_risk_model.pkl")
                model_type = "xgboost"
            elif os.path.exists("cardio_risk_tf_model.h5"):
                from tensorflow.keras.models import load_model
                model = load_model("cardio_risk_tf_model.h5")
                model_type = "tensorflow"
            else:
                st.error("No trained model found.")
                st.stop()

            scaler, continuous = joblib.load("scaler.pkl")

            features = pd.DataFrame([[age,height,weight,gender,ap_hi,ap_lo,
                                      cholesterol,gluc,smoke,alco,active]],
                                    columns=['age','height','weight','gender',
                                             'ap_hi','ap_lo','cholesterol','gluc',
                                             'smoke','alco','active'])
            features[continuous] = scaler.transform(features[continuous])

            if model_type=="xgboost":
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0][1]
            else:
                proba = model.predict(features)[0][0]
                pred = int(proba>=0.5)

            st.subheader("Result")
            st.write(f"👤 Patient: {name if name else 'Anonymous'}")
            if pred==1:
                st.error(f"⚠️ High Risk ({proba*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk ({(1-proba)*100:.2f}%)")
        except Exception as e:
            st.error(f"Error: {e}")
