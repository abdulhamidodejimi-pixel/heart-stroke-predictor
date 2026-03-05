import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# LOAD MODELS
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.write("Enter patient details on the left and choose prediction type.")

# ---------------- SIDEBAR INPUTS ---------------- #

st.sidebar.header("Patient Details")

# General
age = st.sidebar.number_input("Age", 0, 120, 0)
sex_text = st.sidebar.selectbox("Sex", ["Male", "Female"])

# Heart Disease Features
cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.sidebar.slider("Rest ECG", 0,2)
thalach = st.sidebar.number_input("Max Heart Rate", 60,220,150)
exang = st.sidebar.selectbox("Exercise Angina", [0,1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0,6.0,1.0)
slope = st.sidebar.slider("Slope", 0,2)
ca = st.sidebar.slider("Number of vessels (ca)", 0,4)
thal = st.sidebar.slider("Thal", 0,3)

# Stroke Features
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Existing Heart Disease", [0,1])
glucose = st.sidebar.number_input("Glucose Level", 50,300,100)
bmi = st.sidebar.number_input("BMI", 10.0,50.0,25.0)

smoking_text = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked","formerly smoked","smokes"]
)

married_text = st.sidebar.selectbox(
    "Ever Married",
    ["Yes","No"]
)

work_text = st.sidebar.selectbox(
    "Work Type",
    ["Private","Self-employed","Govt_job","children"]
)

residence_text = st.sidebar.selectbox(
    "Residence Type",
    ["Urban","Rural"]
)

# ---------------- ENCODING ---------------- #

sex = 1 if sex_text=="Male" else 0

smoking_map={
    "never smoked":0,
    "formerly smoked":1,
    "smokes":2
}

work_map={
    "Private":0,
    "Self-employed":1,
    "Govt_job":2,
    "children":3
}

married = 1 if married_text=="Yes" else 0
residence = 1 if residence_text=="Urban" else 0

smoking = smoking_map[smoking_text]
work = work_map[work_text]

# ---------------- MODEL INPUTS ---------------- #

heart_input = np.array([[
    age,sex,cp,trestbps,chol,fbs,
    restecg,thalach,exang,oldpeak,
    slope,ca,thal
]])

stroke_input = pd.DataFrame({
    "gender":[sex_text],
    "age":[age],
    "hypertension":[hypertension],
    "heart_disease":[heart_disease],
    "ever_married":[married_text],
    "work_type":[work_text],
    "Residence_type":[residence_text],
    "avg_glucose_level":[glucose],
    "bmi":[bmi],
    "smoking_status":[smoking_text]
})

# ---------------- BUTTONS ---------------- #

col1,col2,col3 = st.columns(3)

with col1:
    heart_btn = st.button("Predict Heart Disease")

with col2:
    stroke_btn = st.button("Predict Stroke")

with col3:
    both_btn = st.button("Predict Both")

# ---------------- HEART PREDICTION ---------------- #

if heart_btn:

    st.subheader("Heart Disease Result")

    try:
        pred = heart_model.predict(heart_input)[0]
        prob = heart_model.predict_proba(heart_input)[0][1]
    except:
        pred = 0
        prob = 0

    if pred == 1:
        st.error(f"High Risk of Heart Disease ({prob*100:.1f}%)")
    else:
        st.success(f"Low Risk of Heart Disease ({prob*100:.1f}%)")


# ---------------- STROKE PREDICTION ---------------- #

if stroke_btn:

    st.subheader("Stroke Result")

    try:
        pred = stroke_model.predict(stroke_input)[0]
        prob = stroke_model.predict_proba(stroke_input)[0][1]
    except:
        pred = 0
        prob = 0

    if pred == 1:
        st.error(f"High Risk of Stroke ({prob*100:.1f}%)")
    else:
        st.success(f"No Stroke Risk ({prob*100:.1f}%)")


# ---------------- BOTH PREDICTION ---------------- #

if both_btn:

    st.subheader("Overall Diagnosis")

    # HEART
    try:
        heart_pred = heart_model.predict(heart_input)[0]
        heart_prob = heart_model.predict_proba(heart_input)[0][1]
    except:
        heart_pred = 0
        heart_prob = 0

    if heart_pred == 1:
        st.error(f"Heart Disease Risk: {heart_prob*100:.1f}%")
    else:
        st.success(f"No Heart Disease ({heart_prob*100:.1f}%)")

    # STROKE
    try:
        stroke_pred = stroke_model.predict(stroke_input)[0]
        stroke_prob = stroke_model.predict_proba(stroke_input)[0][1]
    except:
        stroke_pred = 0
        stroke_prob = 0

    if stroke_pred == 1:
        st.error(f"Stroke Risk: {stroke_prob*100:.1f}%")
    else:
        st.success(f"No Stroke Risk ({stroke_prob*100:.1f}%)")
