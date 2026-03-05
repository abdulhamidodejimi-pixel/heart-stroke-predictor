import streamlit as st
import joblib
import numpy as np

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.title("❤️ Heart & Stroke Prediction App")

st.header("Enter Patient Details")

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol Level")
blood_pressure = st.number_input("Blood Pressure")
glucose = st.number_input("Glucose Level")

# Convert categorical values
sex = 1 if sex == "Male" else 0

if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])
    prediction = heart_model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

if st.button("Predict Stroke"):
    input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])
    prediction = stroke_model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Stroke")
    else:
        st.success("Low Risk of Stroke")
