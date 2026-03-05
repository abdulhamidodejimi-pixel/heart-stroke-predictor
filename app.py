import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# ---------------- LOAD MODELS ---------------- #

heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.write("Enter patient details on the left and choose a prediction type.")

# ---------------- SIDEBAR INPUTS ---------------- #

st.sidebar.header("Patient Details")

# General
age = st.sidebar.slider("Age", 1, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

# Heart Disease Inputs
cp = st.sidebar.slider("Chest Pain Type", 0, 3)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.sidebar.slider("Rest ECG", 0,2)
thalach = st.sidebar.number_input("Max Heart Rate", 60,220,150)
exang = st.sidebar.selectbox("Exercise Angina", [0,1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0,6.0,1.0)
slope = st.sidebar.slider("Slope", 0,2)
ca = st.sidebar.slider("Number of Vessels (ca)", 0,4)
thal = st.sidebar.slider("Thal", 0,3)

# Stroke Inputs
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Existing Heart Disease", [0,1])
glucose = st.sidebar.number_input("Glucose Level", 50,300,100)
bmi = st.sidebar.number_input("BMI", 10.0,50.0,25.0)

smoking = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked","formerly smoked","smokes"]
)

married = st.sidebar.selectbox(
    "Ever Married",
    ["Yes","No"]
)

work = st.sidebar.selectbox(
    "Work Type",
    ["Private","Self-employed","Govt_job","children"]
)

residence = st.sidebar.selectbox(
    "Residence Type",
    ["Urban","Rural"]
)

# ---------------- ENCODING ---------------- #

sex_num = 1 if sex=="Male" else 0
married_num = 1 if married=="Yes" else 0
residence_num = 1 if residence=="Urban" else 0

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

smoking_num = smoking_map[smoking]
work_num = work_map[work]

# ---------------- HEART INPUT ---------------- #

heart_input = np.array([[
    age,sex_num,cp,trestbps,chol,fbs,
    restecg,thalach,exang,oldpeak,
    slope,ca,thal
]])

# ---------------- STROKE INPUT ---------------- #

stroke_data = {
    "gender":[sex],
    "age":[age],
    "hypertension":[hypertension],
    "heart_disease":[heart_disease],
    "ever_married":[married],
    "work_type":[work],
    "Residence_type":[residence],
    "avg_glucose_level":[glucose],
    "bmi":[bmi],
    "smoking_status":[smoking]
}

stroke_df = pd.DataFrame(stroke_data)

# convert categories to model format
stroke_df = pd.get_dummies(stroke_df)

# align columns with model
try:
    expected_cols = stroke_model.feature_names_in_
    stroke_df = stroke_df.reindex(columns=expected_cols, fill_value=0)
except:
    pass

# ---------------- BUTTONS ---------------- #

col1,col2,col3 = st.columns(3)

with col1:
    heart_btn = st.button("❤️ Predict Heart Disease")

with col2:
    stroke_btn = st.button("🧠 Predict Stroke")

with col3:
    both_btn = st.button("📊 Predict Both")

# ---------------- HEART PREDICTION ---------------- #

if heart_btn:

    st.subheader("Heart Disease Result")

    pred = heart_model.predict(heart_input)[0]
    prob = heart_model.predict_proba(heart_input)[0][1]

    if pred==1:
        st.error(f"High Risk of Heart Disease ({prob*100:.1f}%)")
    else:
        st.success(f"Low Risk of Heart Disease ({prob*100:.1f}%)")

# ---------------- STROKE PREDICTION ---------------- #

if stroke_btn:

    st.subheader("Stroke Result")

    pred = stroke_model.predict(stroke_df)[0]
    prob = stroke_model.predict_proba(stroke_df)[0][1]

    if pred==1:
        st.error(f"High Risk of Stroke ({prob*100:.1f}%)")
    else:
        st.success(f"Low Risk of Stroke ({prob*100:.1f}%)")

# ---------------- BOTH PREDICTION ---------------- #

if both_btn:

    st.subheader("Overall Diagnosis")

    # HEART
    heart_pred = heart_model.predict(heart_input)[0]
    heart_prob = heart_model.predict_proba(heart_input)[0][1]

    if heart_pred==1:
        st.error(f"Heart Disease Risk: {heart_prob*100:.1f}%")
    else:
        st.success(f"No Heart Disease ({heart_prob*100:.1f}%)")

    # STROKE
    stroke_pred = stroke_model.predict(stroke_df)[0]
    stroke_prob = stroke_model.predict_proba(stroke_df)[0][1]

    if stroke_pred==1:
        st.error(f"Stroke Risk: {stroke_prob*100:.1f}%")
    else:
        st.success(f"No Stroke Risk ({stroke_prob*100:.1f}%)")
