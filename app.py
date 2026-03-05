import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# ---------------- LOAD MODELS ---------------- #

try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except Exception as e:
    st.error("Model loading failed. Make sure heart_model.pkl and stroke_model.pkl exist.")
    st.stop()

# ---------------- TITLE ---------------- #

st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.write("Enter patient details in the sidebar and select a prediction.")

# ---------------- SIDEBAR INPUTS ---------------- #

st.sidebar.header("Patient Details")

# General
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=0)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

# Heart Disease Features
cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 0)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])
restecg = st.sidebar.slider("Rest ECG", 0, 2, 0)
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.slider("Slope", 0, 2, 0)
ca = st.sidebar.slider("Number of vessels (ca)", 0, 4, 0)
thal = st.sidebar.slider("Thal", 0, 3, 0)

# Stroke Features
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Existing Heart Disease", [0, 1])

glucose = st.sidebar.number_input("Glucose Level", 50, 300, 100)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

smoking = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes"]
)

married = st.sidebar.selectbox(
    "Ever Married",
    ["Yes", "No"]
)

work = st.sidebar.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children"]
)

residence = st.sidebar.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

# ---------------- ENCODING ---------------- #

sex = 1 if sex == "Male" else 0
married = 1 if married == "Yes" else 0
residence = 1 if residence == "Urban" else 0

smoking_map = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2
}

work_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3
}

smoking = smoking_map[smoking]
work = work_map[work]

# ---------------- MODEL INPUTS ---------------- #

heart_input = np.array([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]])

stroke_input = np.array([[
    sex, age, hypertension, heart_disease,
    married, work, residence,
    glucose, bmi, smoking
]])

# ---------------- BUTTONS ---------------- #

col1, col2, col3 = st.columns(3)

heart_btn = col1.button("Predict Heart Disease")
stroke_btn = col2.button("Predict Stroke")
both_btn = col3.button("Predict Both")

# ---------------- HEART PREDICTION ---------------- #

if heart_btn:
    try:
        pred = heart_model.predict(heart_input)[0]

        st.subheader("Heart Disease Result")

        if pred == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error("Heart prediction failed. Model input mismatch.")

# ---------------- STROKE PREDICTION ---------------- #

if stroke_btn:
    try:
        pred = stroke_model.predict(stroke_input)[0]

        st.subheader("Stroke Result")

        if pred == 1:
            st.error("⚠ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

    except Exception as e:
        st.error("Stroke prediction failed. Model input mismatch.")

# ---------------- BOTH PREDICTION ---------------- #

if both_btn:

    st.subheader("Overall Diagnosis")

    try:
        heart_pred = heart_model.predict(heart_input)[0]

        if heart_pred == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except:
        st.warning("Heart model failed.")

    try:
        stroke_pred = stroke_model.predict(stroke_input)[0]

        if stroke_pred == 1:
            st.error("⚠ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

    except:
        st.warning("Stroke model failed.")
