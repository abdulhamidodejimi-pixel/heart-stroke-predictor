import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# Load models safely
try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except:
    st.error("Model files not found. Please upload heart_model.pkl and stroke_model.pkl")
    st.stop()

st.title("🩺 AI Heart & Stroke Risk Prediction System")

st.write("Enter patient information and choose a prediction type.")

# ---------------------------
# LEFT SIDE INPUT
# ---------------------------

st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 1, 100, 40)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# HEART FEATURES
cp = st.sidebar.slider("Chest Pain Type", 0, 3)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.sidebar.slider("Rest ECG", 0,2)
thalach = st.sidebar.number_input("Max Heart Rate", 60,220,150)
exang = st.sidebar.selectbox("Exercise Angina", [0,1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0,6.0,1.0)
slope = st.sidebar.slider("Slope", 0,2)
ca = st.sidebar.slider("Number of vessels", 0,4)
thal = st.sidebar.slider("Thal", 0,3)

# STROKE FEATURES
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Existing Heart Disease", [0,1])
married = st.sidebar.selectbox("Ever Married", ["Yes","No"])

work = st.sidebar.selectbox(
"Work Type",
["Private","Self-employed","Govt_job","children"]
)

residence = st.sidebar.selectbox(
"Residence Type",
["Urban","Rural"]
)

glucose = st.sidebar.number_input("Average Glucose Level", 50,300,100)
bmi = st.sidebar.number_input("BMI", 10.0,50.0,25.0)

smoking = st.sidebar.selectbox(
"Smoking Status",
["never smoked","formerly smoked","smokes"]
)

# ---------------------------
# ENCODING
# ---------------------------

gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
residence = 1 if residence=="Urban" else 0

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

smoking=smoking_map[smoking]
work=work_map[work]

# ---------------------------
# MODEL INPUT
# ---------------------------

heart_input = np.array([[
age,gender,cp,trestbps,chol,fbs,restecg,
thalach,exang,oldpeak,slope,ca,thal
]])

stroke_input = np.array([[
gender,age,hypertension,heart_disease,
married,work,residence,glucose,bmi,smoking
]])

# ---------------------------
# BUTTONS
# ---------------------------

col1,col2,col3 = st.columns(3)

heart_btn = col1.button("❤️ Predict Heart Disease")
stroke_btn = col2.button("🧠 Predict Stroke")
both_btn = col3.button("🩺 Predict Both")

# ---------------------------
# HEART PREDICTION
# ---------------------------

if heart_btn:

    try:
        pred = heart_model.predict(heart_input)[0]
        prob = heart_model.predict_proba(heart_input)[0][1]

        st.subheader("Heart Disease Result")

        if pred==1:
            st.error(f"High Risk of Heart Disease ({prob*100:.1f}%)")
        else:
            st.success(f"No Heart Disease ({prob*100:.1f}%)")

    except:
        st.error("Heart prediction failed. Model input mismatch.")

# ---------------------------
# STROKE PREDICTION
# ---------------------------

if stroke_btn:

    try:
        pred = stroke_model.predict(stroke_input)[0]
        prob = stroke_model.predict_proba(stroke_input)[0][1]

        st.subheader("Stroke Result")

        if pred==1:
            st.error(f"High Risk of Stroke ({prob*100:.1f}%)")
        else:
            st.success(f"No Stroke Risk ({prob*100:.1f}%)")

    except:
        st.error("Stroke prediction failed. Model input mismatch.")

# ---------------------------
# BOTH PREDICTION
# ---------------------------

if both_btn:

    try:

        heart_pred = heart_model.predict(heart_input)[0]
        stroke_pred = stroke_model.predict(stroke_input)[0]

        st.subheader("Overall Diagnosis")

        if heart_pred==1 and stroke_pred==1:
            st.error("⚠ Patient has BOTH Heart Disease and Stroke Risk")

        elif heart_pred==1 and stroke_pred==0:
            st.warning("⚠ Patient has Heart Disease ONLY")

        elif heart_pred==0 and stroke_pred==1:
            st.warning("⚠ Patient has Stroke Risk ONLY")

        else:
            st.success("✅ Patient has NO Heart Disease or Stroke")

    except:
        st.error("Prediction failed. Check model inputs.")
