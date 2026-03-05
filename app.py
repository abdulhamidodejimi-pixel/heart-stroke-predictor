import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Heart & Stroke Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart & Stroke Prediction App")

st.markdown("""
This application predicts the risk of **Heart Disease** and **Stroke**
using trained Machine Learning models.
""")

# ------------------------------
# Load Models (Safe Loading)
# ------------------------------
try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except Exception as e:
    st.error("Error loading model files.")
    st.stop()

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol Level", min_value=0.0, value=200.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0.0, value=120.0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0.0, value=100.0)

# Convert categorical values
sex = 1 if sex == "Male" else 0

# Prepare input
input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

st.subheader("Prediction Results")

col1, col2 = st.columns(2)

# ------------------------------
# Heart Disease Prediction
# ------------------------------
with col1:
    if st.button("Predict Heart Disease"):
        try:
            prediction = heart_model.predict(input_data)
            probability = heart_model.predict_proba(input_data)[0][1]

            st.metric("Heart Disease Risk (%)", f"{probability*100:.2f}%")

            if prediction[0] == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")
        except:
            st.error("Model prediction error.")

# ------------------------------
# Stroke Prediction
# ------------------------------
with col2:
    if st.button("Predict Stroke"):
        try:
            prediction = stroke_model.predict(input_data)
            probability = stroke_model.predict_proba(input_data)[0][1]

            st.metric("Stroke Risk (%)", f"{probability*100:.2f}%")

            if prediction[0] == 1:
                st.error("High Risk of Stroke")
            else:
                st.success("Low Risk of Stroke")
        except:
            st.error("Model prediction error.")
