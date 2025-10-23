import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt

# ---------------------------------------
# 🩺 DATA GENERATION
# ---------------------------------------
def generate_data():
    st.info("Generating synthetic cardio dataset with advanced features...")
    np.random.seed(42)
    n_samples = 3000

    df = pd.DataFrame({
        "age": np.random.randint(30*365, 70*365, n_samples),   # in days
        "height": np.random.randint(150, 200, n_samples),
        "weight": np.random.randint(50, 120, n_samples),
        "gender": np.random.choice([1, 2], n_samples),         # 1=female, 2=male
        "ap_hi": np.random.randint(90, 200, n_samples),
        "ap_lo": np.random.randint(60, 120, n_samples),
        "cholesterol": np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15]),
        "gluc": np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]),
        "smoke": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "alco": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "active": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        # New features
        "bmi": np.round(np.random.uniform(18, 35, n_samples), 1),
        "family_history": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "diet": np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]),  # 1=healthy,2=moderate,3=poor
        "sleep_hours": np.random.randint(4, 10, n_samples),
        "stress_level": np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.35, 0.15]),
        "heart_rate": np.random.randint(55, 110, n_samples),
    })

    # Risk factor logic
    risk_factor = (
        (df["age"]/365 > 50).astype(int) +
        (df["ap_hi"] > 140).astype(int) +
        (df["cholesterol"] > 1).astype(int) +
        (df["gluc"] > 1).astype(int) +
        (df["bmi"] > 27).astype(int) +
        df["family_history"] +
        (df["diet"] == 3).astype(int) +
        (df["sleep_hours"] < 6).astype(int) +
        (df["stress_level"] == 3).astype(int) +
        (df["heart_rate"] > 100).astype(int) +
        df["smoke"] + df["alco"] + (1 - df["active"])
    )
    df["cardio"] = (risk_factor >= 4).astype(int)

    df.to_csv("cardio_data.csv", index=False)
    st.success("✅ cardio_data.csv generated successfully with 3000 rows.")
    st.dataframe(df.head())


# ---------------------------------------
# ⚙️ TRAINING FUNCTION
# ---------------------------------------
def train_model():
    if not os.path.exists("cardio_data.csv"):
        st.error("⚠️ Please generate data first.")
        return

    df = pd.read_csv("cardio_data.csv")

    features = [
        'age','height','weight','gender','ap_hi','ap_lo','cholesterol','gluc',
        'smoke','alco','active','bmi','family_history','diet',
        'sleep_hours','stress_level','heart_rate'
    ]
    target = 'cardio'
    continuous = ['age','height','weight','ap_hi','ap_lo','bmi','sleep_hours','heart_rate']

    scaler = StandardScaler()
    df[continuous] = scaler.fit_transform(df[continuous])
    joblib.dump((scaler, continuous), "scaler.pkl")

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    st.spinner("Training XGBoost model...")
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

    st.success("✅ Model trained successfully!")
    st.metric("Accuracy", f"{acc*100:.2f}%")

    joblib.dump(model, "cardio_risk_model.pkl")

    # Feature importance chart
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)


# ---------------------------------------
# 📈 DASHBOARD FUNCTION
# ---------------------------------------
def show_dashboard():
    if not os.path.exists("predictions_log.csv"):
        st.warning("No prediction logs found yet.")
        return
    df = pd.read_csv("predictions_log.csv")
    st.subheader("📈 Risk Summary Dashboard")
    st.dataframe(df.tail())

    # Pie chart of risk distribution
    counts = df["risk"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)


# ---------------------------------------
# 💬 HEALTH ADVICE FUNCTION
# ---------------------------------------
def health_advice(pred, features):
    tips = []
    row = features.iloc[0]
    if pred == 1:
        tips.append("Reduce sodium intake to help lower blood pressure.")
        if row['bmi'] > 27: tips.append("Consider weight management through a balanced diet and exercise.")
        if row['cholesterol'] > 1: tips.append("Monitor cholesterol and avoid fried foods.")
        if row['gluc'] > 1: tips.append("Control sugar levels and eat more fiber.")
        if row['sleep_hours'] < 6: tips.append("Aim for at least 7 hours of sleep per night.")
        if row['stress_level'] == 3: tips.append("Try stress-reduction techniques such as yoga or meditation.")
        tips.append("Consult your physician for regular heart checkups.")
    else:
        tips.append("Maintain your healthy lifestyle and stay active!")
    return tips


# ---------------------------------------
# 💾 SAVE PREDICTION LOG
# ---------------------------------------
def save_prediction(entry):
    file = "predictions_log.csv"
    df = pd.DataFrame([entry])
    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)


# ---------------------------------------
# 🎯 PREDICTION SECTION
# ---------------------------------------
def predict_section():
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

        # New fields
        bmi = st.number_input("BMI", 10.0, 45.0, 25.0)
        family_history = 1 if st.selectbox("Family History of Heart Disease", ["No","Yes"])=="Yes" else 0
        diet = {"Healthy":1,"Moderate":2,"Poor":3}[st.selectbox("Diet Quality",["Healthy","Moderate","Poor"])]
        sleep_hours = st.number_input("Sleep Hours per Day", 3, 12, 7)
        stress_level = {"Low":1,"Medium":2,"High":3}[st.selectbox("Stress Level",["Low","Medium","High"])]
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 75)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            model = joblib.load("cardio_risk_model.pkl")
            scaler, continuous = joblib.load("scaler.pkl")

            features = pd.DataFrame([[age,height,weight,gender,ap_hi,ap_lo,
                                      cholesterol,gluc,smoke,alco,active,
                                      bmi,family_history,diet,sleep_hours,stress_level,heart_rate]],
                                    columns=['age','height','weight','gender','ap_hi','ap_lo',
                                             'cholesterol','gluc','smoke','alco','active',
                                             'bmi','family_history','diet','sleep_hours','stress_level','heart_rate'])
            features[continuous] = scaler.transform(features[continuous])

            proba = model.predict_proba(features)[0][1]
            pred = int(proba >= 0.5)

            st.subheader("🧾 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ High Risk ({proba*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk ({(1-proba)*100:.2f}%)")

            # 💬 Advice
            tips = health_advice(pred, features)
            st.subheader("💬 Health Advice")
            for t in tips:
                st.write(f"• {t}")

            # 💾 Log prediction
            save_prediction({"name": name or "Anonymous", "risk": "High" if pred==1 else "Low", "probability": round(proba*100, 2)})
            st.success("Prediction saved to log.")

            # 🧾 PDF Report
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4
            c.setTitle("Cardiovascular Risk Report")

            c.setFont("Helvetica-Bold", 20)
            c.drawString(1*inch, height - 1*inch, "🩺 Cardiovascular Risk Report")

            c.setFont("Helvetica", 12)
            c.drawString(1*inch, height - 1.5*inch, f"Name: {name or 'Anonymous'}")
            c.drawString(1*inch, height - 1.8*inch, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(colors.red if pred==1 else colors.green)
            c.drawString(1*inch, height - 2.3*inch, f"Risk Level: {'HIGH' if pred==1 else 'LOW'}")
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 12)
            c.drawString(1*inch, height - 2.6*inch, f"Predicted Probability: {proba*100:.2f}%")

            c.setFont("Helvetica-Bold", 13)
            c.drawString(1*inch, height - 3.1*inch, "Recommended Health Advice:")
            y = height - 3.4*inch
            c.setFont("Helvetica", 11)
            for tip in tips:
                c.drawString(1.1*inch, y, f"• {tip}")
                y -= 0.25*inch

            c.showPage()
            c.save()
            pdf_buffer.seek(0)

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"CardioRiskReport_{name or 'Patient'}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error: {e}")


# ---------------------------------------
# 🧭 APP STRUCTURE
# ---------------------------------------
st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="wide")
st.title("🩺 Cardiovascular Risk Prediction App")

menu = ["Generate Data", "Train Model", "Predict Risk", "Dashboard"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Generate Data":
    generate_data()
elif choice == "Train Model":
    train_model()
elif choice == "Predict Risk":
    predict_section()
elif choice == "Dashboard":
    show_dashboard()
