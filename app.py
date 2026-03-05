import streamlit as st
import joblib
import numpy as np

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="AI Health Predictor", layout="wide")

st.title("🩺 AI Heart Disease & Stroke Prediction System")
st.write("Enter patient details on the left and run prediction.")

# -----------------------------
# SIDEBAR INPUT (LEFT SIDE)
# -----------------------------

st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type (Heart)", [0,1,2,3])

trestbps = st.sidebar.number_input("Resting Blood Pressure", value=120)

chol = st.sidebar.number_input("Cholesterol Level", value=200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])

restecg = st.sidebar.selectbox("Rest ECG", [0,1,2])

thalach = st.sidebar.number_input("Max Heart Rate", value=150)

exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])

oldpeak = st.sidebar.number_input("Old Peak", value=1.0)

slope = st.sidebar.selectbox("Slope", [0,1,2])

ca = st.sidebar.selectbox("Number of Major Vessels", [0,1,2,3])

thal = st.sidebar.selectbox("Thal", [0,1,2,3])


# Stroke inputs
hypertension = st.sidebar.selectbox("Hypertension", [0,1])

heart_disease_input = st.sidebar.selectbox("Existing Heart Disease (Stroke Model)", [0,1])

avg_glucose = st.sidebar.number_input("Average Glucose Level", value=100.0)

bmi = st.sidebar.number_input("BMI", value=25.0)

smoking = st.sidebar.selectbox("Smoking Status", [0,1,2])

# -----------------------------
# HEART MODEL INPUT
# -----------------------------

heart_features = np.array([[

    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal

]])

# -----------------------------
# STROKE MODEL INPUT
# -----------------------------

stroke_features = np.array([[

    age,
    hypertension,
    heart_disease_input,
    avg_glucose,
    bmi,
    smoking

]])

# -----------------------------
# PREDICTION BUTTONS
# -----------------------------

col1, col2, col3 = st.columns(3)

# HEART PREDICTION
if col1.button("Predict Heart Disease"):
    
    try:
        heart_pred = heart_model.predict(heart_features)

        if heart_pred[0] == 1:
            st.error("⚠ Patient HAS Heart Disease")
        else:
            st.success("✅ Patient does NOT have Heart Disease")

    except:
        st.error("Heart prediction failed. Model input mismatch.")


# STROKE PREDICTION
if col2.button("Predict Stroke"):
    
    try:
        stroke_pred = stroke_model.predict(stroke_features)

        if stroke_pred[0] == 1:
            st.error("⚠ Patient HAS Stroke Risk")
        else:
            st.success("✅ Patient does NOT have Stroke")

    except:
        st.error("Stroke prediction failed. Model input mismatch.")


# BOTH PREDICTION
if col3.button("Predict Both"):

    try:
        heart_pred = heart_model.predict(heart_features)
        stroke_pred = stroke_model.predict(stroke_features)

        st.subheader("Combined Result")

        if heart_pred[0] == 1:
            st.error("⚠ Patient HAS Heart Disease")
        else:
            st.success("✅ No Heart Disease")

        if stroke_pred[0] == 1:
            st.error("⚠ Patient HAS Stroke Risk")
        else:
            st.success("✅ No Stroke Detected")

    except:
        st.error("Prediction failed. Check model inputs.")
