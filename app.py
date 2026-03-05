import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart & Stroke Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart & Stroke Prediction App")
st.markdown("AI-powered medical risk assessment system")

# Load models safely
try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except:
    st.error("Model files not found.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol Level", 0.0, 600.0, 200.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0.0, 300.0, 120.0)
glucose = st.sidebar.number_input("Glucose Level", 0.0, 500.0, 100.0)

sex = 1 if sex == "Male" else 0

input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

st.subheader("Prediction Panel")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Predict Heart"):
        prediction = heart_model.predict(input_data)
        if prediction[0] == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

with col2:
    if st.button("Predict Stroke"):
        prediction = stroke_model.predict(input_data)
        if prediction[0] == 1:
            st.error("High Risk of Stroke")
        else:
            st.success("Low Risk of Stroke")

with col3:
    if st.button("Predict Both"):
        heart_pred = heart_model.predict(input_data)
        stroke_pred = stroke_model.predict(input_data)

        if heart_pred[0] == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        if stroke_pred[0] == 1:
            st.error("High Risk of Stroke")
        else:
            st.success("Low Risk of Stroke")
