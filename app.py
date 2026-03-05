import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
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

# -------------------------------
# Load Models Safely
# -------------------------------
try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except Exception as e:
    st.error("Error loading model files. Make sure .pkl files are uploaded.")
    st.stop()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol Level", min_value=0.0, value=200.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0.0, value=120.0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0.0, value=100.0)

# Convert categorical input
sex = 1 if sex == "Male" else 0

# Prepare input array
input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

st.subheader("Prediction Results")

# -------------------------------
# 3 Button Layout
# -------------------------------
col1, col2, col3 = st.columns(3)

# -------------------------------
# Heart Prediction
# -------------------------------
with col1:
    if st.button("❤️ Predict Heart"):
        try:
            heart_pred = heart_model.predict(input_data)
            heart_prob = heart_model.predict_proba(input_data)[0][1]

            st.metric("Heart Risk (%)", f"{heart_prob*100:.2f}%")

            if heart_pred[0] == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")
        except:
            st.error("Prediction error.")

# -------------------------------
# Stroke Prediction
# -------------------------------
with col2:
    if st.button("🧠 Predict Stroke"):
        try:
            stroke_pred = stroke_model.predict(input_data)
            stroke_prob = stroke_model.predict_proba(input_data)[0][1]

            st.metric("Stroke Risk (%)", f"{stroke_prob*100:.2f}%")

            if stroke_pred[0] == 1:
                st.error("High Risk of Stroke")
            else:
                st.success("Low Risk of Stroke")
        except:
            st.error("Prediction error.")

# -------------------------------
# Predict Both
# -------------------------------
with col3:
    if st.button("🔍 Predict Both"):
        try:
            # Heart
            heart_pred = heart_model.predict(input_data)
            heart_prob = heart_model.predict_proba(input_data)[0][1]

            # Stroke
            stroke_pred = stroke_model.predict(input_data)
            stroke_prob = stroke_model.predict_proba(input_data)[0][1]

            st.write("### Combined Results")

            col_h, col_s = st.columns(2)

            with col_h:
                st.metric("Heart Risk (%)", f"{heart_prob*100:.2f}%")
                if heart_pred[0] == 1:
                    st.error("High Risk of Heart Disease")
                else:
                    st.success("Low Risk of Heart Disease")

            with col_s:
                st.metric("Stroke Risk (%)", f"{stroke_prob*100:.2f}%")
                if stroke_pred[0] == 1:
                    st.error("High Risk of Stroke")
                else:
                    st.success("Low Risk of Stroke")

        except:
            st.error("Prediction error.")
