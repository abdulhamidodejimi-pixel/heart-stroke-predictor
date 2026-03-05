import streamlit as st
import numpy as np
import pickle

# Load models
heart_model = pickle.load(open("heart_model.pkl", "rb"))
stroke_model = pickle.load(open("stroke_model.pkl", "rb"))

st.title("AI Health Prediction System")
st.write("Predict Heart Disease and Stroke Risk")

st.subheader("Enter Patient Details")

# HEART DATA INPUTS
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

heart_input = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                         thalach,exang,oldpeak,slope,ca,thal]])

# STROKE INPUTS
st.subheader("Stroke Related Information")

hypertension = st.selectbox("Hypertension", [0,1])
heart_disease = st.selectbox("Heart Disease History", [0,1])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

stroke_input = np.array([[age, hypertension, heart_disease,
                          avg_glucose_level, bmi]])

st.write("---")

# HEART PREDICTION
if st.button("Predict Heart Disease"):

    try:
        prediction = heart_model.predict(heart_input)[0]
        probability = heart_model.predict_proba(heart_input)[0][1] * 100

        if prediction == 1:
            st.error(f"High Risk of Heart Disease ({probability:.2f}%)")
        else:
            st.success(f"Low Risk of Heart Disease ({100-probability:.2f}%)")

    except:
        st.error("Heart prediction failed. Check model inputs.")


# STROKE PREDICTION
if st.button("Predict Stroke"):

    try:
        prediction = stroke_model.predict(stroke_input)[0]
        probability = stroke_model.predict_proba(stroke_input)[0][1] * 100

        if prediction == 1:
            st.error(f"Stroke Risk Detected ({probability:.2f}%)")
        else:
            st.success(f"No Stroke Detected ({100-probability:.2f}%)")

    except:
        st.error("Stroke prediction failed. Model input mismatch.")


# BOTH PREDICTION
if st.button("Predict Both"):

    try:
        # HEART
        heart_prediction = heart_model.predict(heart_input)[0]
        heart_prob = heart_model.predict_proba(heart_input)[0][1] * 100

        if heart_prediction == 1:
            heart_result = f"High Risk ({heart_prob:.2f}%)"
        else:
            heart_result = f"Low Risk ({100-heart_prob:.2f}%)"

        # STROKE
        stroke_prediction = stroke_model.predict(stroke_input)[0]
        stroke_prob = stroke_model.predict_proba(stroke_input)[0][1] * 100

        if stroke_prediction == 1:
            stroke_result = f"Stroke Risk ({stroke_prob:.2f}%)"
        else:
            stroke_result = f"No Stroke ({100-stroke_prob:.2f}%)"

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Heart Disease Risk", heart_result)

        with col2:
            st.metric("Stroke Risk", stroke_result)

    except:
        st.error("Prediction failed. Check model inputs.")
