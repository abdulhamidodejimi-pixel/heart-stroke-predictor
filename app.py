import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Health Prediction System", layout="wide")

st.title("AI Health Prediction System")
st.write("Predict Heart Disease and Stroke Risk")

# Load Models
heart_model = pickle.load(open("heart_model.pkl", "rb"))
stroke_model = pickle.load(open("stroke_model.pkl", "rb"))

# -----------------------------
# Patient Input Section
# -----------------------------

st.header("Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=0)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

# Stroke Inputs
hypertension = st.selectbox("Hypertension", [0,1])
heart_disease = st.selectbox("Existing Heart Disease", [0,1])
ever_married = st.selectbox("Ever Married", ["Yes","No"])
work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
residence = st.selectbox("Residence Type", ["Urban","Rural"])
avg_glucose = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")
smoking = st.selectbox("Smoking Status", ["formerly smoked","never smoked","smokes","Unknown"])


# -----------------------------
# Convert Categorical Data
# -----------------------------

sex = 1 if sex == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
residence = 1 if residence == "Urban" else 0

work_map = {
"Private":0,
"Self-employed":1,
"Govt_job":2,
"children":3,
"Never_worked":4
}

smoke_map = {
"formerly smoked":0,
"never smoked":1,
"smokes":2,
"Unknown":3
}

work_type = work_map[work_type]
smoking = smoke_map[smoking]


# -----------------------------
# Prediction Buttons
# -----------------------------

col1, col2, col3 = st.columns(3)

# HEART PREDICTION
with col1:
    if st.button("Predict Heart Disease"):

        heart_features = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                                    thalach,exang,oldpeak,slope,ca,thal]])

        result = heart_model.predict(heart_features)

        if result[0] == 1:
            st.error("Patient HAS Heart Disease")
        else:
            st.success("Patient does NOT have Heart Disease")


# STROKE PREDICTION
with col2:
    if st.button("Predict Stroke"):

        stroke_features = np.array([[sex,age,hypertension,heart_disease,
                                     ever_married,work_type,residence,
                                     avg_glucose,bmi,smoking]])

        result = stroke_model.predict(stroke_features)

        if result[0] == 1:
            st.error("Patient HAS Stroke Risk")
        else:
            st.success("Patient does NOT have Stroke Risk")


# BOTH PREDICTION
with col3:
    if st.button("Predict Both"):

        heart_features = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                                    thalach,exang,oldpeak,slope,ca,thal]])

        stroke_features = np.array([[sex,age,hypertension,heart_disease,
                                     ever_married,work_type,residence,
                                     avg_glucose,bmi,smoking]])

        heart_result = heart_model.predict(heart_features)
        stroke_result = stroke_model.predict(stroke_features)

        st.subheader("Results")

        if heart_result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease")

        if stroke_result[0] == 1:
            st.error("Stroke Risk Detected")
        else:
            st.success("No Stroke Risk")
