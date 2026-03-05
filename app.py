import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# ---------------- LOAD MODELS ---------------- #

try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except:
    st.error("Model files not found. Make sure heart_model.pkl and stroke_model.pkl exist.")
    st.stop()

# ---------------- TITLE ---------------- #

st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.write("Enter patient details and choose a prediction option.")

# ---------------- SIDEBAR INPUT ---------------- #

st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 1, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

cp = st.sidebar.slider("Chest Pain Type", 0, 3)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])
restecg = st.sidebar.slider("Rest ECG", 0, 2)
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.slider("Slope", 0, 2)
ca = st.sidebar.slider("Major Vessels", 0, 4)
thal = st.sidebar.slider("Thal", 0, 3)

# Stroke features
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

# ---------------- INPUT ARRAYS ---------------- #

heart_input = np.array([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang,
    oldpeak, slope, ca, thal
]])

stroke_input = np.array([[
    sex, age, hypertension, heart_disease,
    married, work, residence,
    glucose, bmi, smoking
]])

# ---------------- BUTTONS ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    heart_btn = st.button("❤️ Predict Heart Disease")

with col2:
    stroke_btn = st.button("🧠 Predict Stroke")

with col3:
    both_btn = st.button("🔎 Predict Both")

# ---------------- HEART PREDICTION ---------------- #

if heart_btn:

    st.subheader("Heart Disease Result")

    try:
        pred = heart_model.predict(heart_input)[0]

        if hasattr(heart_model, "predict_proba"):
            prob = heart_model.predict_proba(heart_input)[0][1]
        else:
            prob = 0.5

        if pred == 1:
            st.error(f"High Risk of Heart Disease ({prob*100:.2f}%)")
        else:
            st.success(f"Low Risk of Heart Disease ({prob*100:.2f}%)")

    except Exception as e:
        st.error("Heart prediction failed.")

# ---------------- STROKE PREDICTION ---------------- #

if stroke_btn:

    st.subheader("Stroke Result")

    try:
        pred = stroke_model.predict(stroke_input)[0]

        if hasattr(stroke_model, "predict_proba"):
            prob = stroke_model.predict_proba(stroke_input)[0][1]
        else:
            prob = 0.5

        if pred == 1:
            st.error(f"High Risk of Stroke ({prob*100:.2f}%)")
        else:
            st.success(f"No Stroke Risk ({prob*100:.2f}%)")

    except Exception as e:
        st.error("Stroke prediction failed. Model feature mismatch.")

# ---------------- BOTH PREDICTION ---------------- #

if both_btn:

    st.subheader("Full Health Diagnosis")

    # HEART
    try:
        heart_pred = heart_model.predict(heart_input)[0]

        if hasattr(heart_model, "predict_proba"):
            heart_prob = heart_model.predict_proba(heart_input)[0][1]
        else:
            heart_prob = 0.5

        if heart_pred == 1:
            st.error(f"❤️ High Risk of Heart Disease ({heart_prob*100:.2f}%)")
        else:
            st.success(f"❤️ Low Risk of Heart Disease ({heart_prob*100:.2f}%)")

    except:
        st.warning("Heart prediction unavailable.")

    # STROKE
    try:
        stroke_pred = stroke_model.predict(stroke_input)[0]

        if hasattr(stroke_model, "predict_proba"):
            stroke_prob = stroke_model.predict_proba(stroke_input)[0][1]
        else:
            stroke_prob = 0.5

        if stroke_pred == 1:
            st.error(f"🧠 High Risk of Stroke ({stroke_prob*100:.2f}%)")
        else:
            st.success(f"🧠 No Stroke Risk ({stroke_prob*100:.2f}%)")

    except:
        st.warning("Stroke prediction unavailable.")
