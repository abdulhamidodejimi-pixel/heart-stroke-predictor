import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart & Stroke Prediction", layout="wide")

# ---------------- LOAD MODELS ---------------- #

try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except Exception as e:
    st.error("Model loading failed.")
    st.stop()

# ---------------- TITLE ---------------- #

st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.write("Enter patient details and select prediction type.")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 0, 120, 0)
sex = st.sidebar.selectbox("Sex", ["Male","Female"])

# HEART FEATURES
cp = st.sidebar.slider("Chest Pain Type",0,3,0)
trestbps = st.sidebar.number_input("Resting Blood Pressure",80,200,120)
chol = st.sidebar.number_input("Cholesterol",100,400,200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120",[0,1])
restecg = st.sidebar.slider("Rest ECG",0,2,0)
thalach = st.sidebar.number_input("Max Heart Rate",60,220,150)
exang = st.sidebar.selectbox("Exercise Angina",[0,1])
oldpeak = st.sidebar.slider("Oldpeak",0.0,6.0,1.0)
slope = st.sidebar.slider("Slope",0,2,0)
ca = st.sidebar.slider("Number of vessels",0,4,0)
thal = st.sidebar.slider("Thal",0,3,0)

# STROKE FEATURES
hypertension = st.sidebar.selectbox("Hypertension",[0,1])
heart_disease = st.sidebar.selectbox("Existing Heart Disease",[0,1])
glucose = st.sidebar.number_input("Glucose Level",50,300,100)
bmi = st.sidebar.number_input("BMI",10.0,50.0,25.0)

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

sex = 1 if sex=="Male" else 0
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

# ---------------- MODEL INPUTS ---------------- #

heart_input = np.array([[age,sex,cp,trestbps,chol,fbs,
                         restecg,thalach,exang,oldpeak,
                         slope,ca,thal]])

stroke_input = np.array([[sex,age,hypertension,heart_disease,
                          married,work,residence,
                          glucose,bmi,smoking]])

# ---------------- BUTTONS ---------------- #

col1,col2,col3 = st.columns(3)

heart_btn = col1.button("Predict Heart Disease")
stroke_btn = col2.button("Predict Stroke")
both_btn = col3.button("Predict Both")

# ---------------- HEART ---------------- #

if heart_btn:

    pred = heart_model.predict(heart_input)[0]

    try:
        prob = heart_model.predict_proba(heart_input)[0][1]*100
    except:
        prob = 0

    st.subheader("Heart Disease Result")

    if pred==1:
        st.error(f"⚠ Patient likely has Heart Disease ({prob:.2f}%)")
    else:
        st.success(f"✅ Patient likely does NOT have Heart Disease ({prob:.2f}%)")


# ---------------- STROKE ---------------- #

if stroke_btn:

    try:

        pred = stroke_model.predict(stroke_input)[0]

        try:
            prob = stroke_model.predict_proba(stroke_input)[0][1]*100
        except:
            prob = 0

        st.subheader("Stroke Result")

        if pred==1:
            st.error(f"⚠ Patient likely has Stroke Risk ({prob:.2f}%)")
        else:
            st.success(f"✅ Patient likely does NOT have Stroke ({prob:.2f}%)")

    except Exception as e:
        st.error("Stroke model input mismatch. Model must be retrained with same features.")


# ---------------- BOTH ---------------- #

if both_btn:

    st.subheader("Overall Diagnosis")

    # HEART
    try:

        heart_pred = heart_model.predict(heart_input)[0]

        try:
            heart_prob = heart_model.predict_proba(heart_input)[0][1]*100
        except:
            heart_prob = 0

        if heart_pred==1:
            st.error(f"⚠ Heart Disease Risk: {heart_prob:.2f}%")
        else:
            st.success(f"✅ No Heart Disease Risk: {heart_prob:.2f}%")

    except:
        st.warning("Heart prediction failed.")

    # STROKE
    try:

        stroke_pred = stroke_model.predict(stroke_input)[0]

        try:
            stroke_prob = stroke_model.predict_proba(stroke_input)[0][1]*100
        except:
            stroke_prob = 0

        if stroke_pred==1:
            st.error(f"⚠ Stroke Risk: {stroke_prob:.2f}%")
        else:
            st.success(f"✅ No Stroke Risk: {stroke_prob:.2f}%")

    except:
        st.warning("Stroke prediction failed.")
