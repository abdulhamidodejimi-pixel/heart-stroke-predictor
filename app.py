import streamlit as st
import numpy as np
import joblib

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="Health Risk Prediction System", layout="wide")

st.title("🩺 AI Health Risk Prediction System")
st.write("Predict Heart Disease and Stroke Risk")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Prediction",
    ["Heart Disease Prediction", "Stroke Prediction", "Full Health Check"]
)

# -----------------------------
# HEART DISEASE
# -----------------------------

if option == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("Old Peak")
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [0,1,2,3])

    if st.button("Predict Heart Disease"):

        input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                                thalach,exang,oldpeak,slope,ca,thal]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Patient is likely to have Heart Disease")
        else:
            st.success("✅ Patient is not likely to have Heart Disease")


# -----------------------------
# STROKE
# -----------------------------

elif option == "Stroke Prediction":

    st.header("Stroke Prediction")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    avg_glucose = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Predict Stroke"):

        input_data = np.array([[age,hypertension,heart_disease,avg_glucose,bmi]])

        prediction = stroke_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Patient is likely to have Stroke")
        else:
            st.success("✅ Patient is not likely to have Stroke")


# -----------------------------
# FULL HEALTH CHECK
# -----------------------------

else:

    st.header("Full Health Risk Check")

    st.subheader("Basic Information")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Existing Heart Disease", [0,1])
    glucose = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Run Full Diagnosis"):

        stroke_input = np.array([[age,hypertension,heart_disease,glucose,bmi]])
        stroke_pred = stroke_model.predict(stroke_input)

        heart_input = np.array([[age,1,1,120,200,0,1,150,0,1,1,0,2]])
        heart_pred = heart_model.predict(heart_input)

        if heart_pred[0] == 1 and stroke_pred[0] == 1:
            st.error("⚠️ Patient may have BOTH Stroke and Heart Disease")

        elif heart_pred[0] == 1:
            st.warning("⚠️ Risk of Heart Disease detected")

        elif stroke_pred[0] == 1:
            st.warning("⚠️ Risk of Stroke detected")

        else:
            st.success("✅ No major risk detected")
