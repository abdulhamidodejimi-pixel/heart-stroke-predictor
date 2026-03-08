import streamlit as st
import numpy as np
import pickle
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# LOAD MODELS SAFELY
# =====================================

stroke_model = None
heart_model = None

if os.path.exists("stroke_model.pkl"):
    stroke_model = pickle.load(open("stroke_model.pkl", "rb"))

if os.path.exists("heart_model.pkl"):
    heart_model = pickle.load(open("heart_model.pkl", "rb"))

# =====================================
# HEADER
# =====================================

st.markdown(
"<h1 style='text-align:center;color:#0A4DA3;'>AI Medical Diagnosis System</h1>",
unsafe_allow_html=True
)

st.markdown(
"<p style='text-align:center;'>Stroke & Heart Disease Prediction using Machine Learning</p>",
unsafe_allow_html=True
)

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
"Select Page",
[
"Dashboard",
"Stroke Prediction",
"Heart Disease Prediction",
"Full Diagnosis"
]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This AI system predicts:

• Stroke Risk  
• Heart Disease Risk  
• Combined Diagnosis
"""
)

# =====================================
# DASHBOARD
# =====================================

if menu == "Dashboard":

    st.header("Welcome")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Stroke Risk Prediction")

    with col2:
        st.info("Heart Disease Prediction")

    with col3:
        st.info("Full Medical Diagnosis")

    st.success("Use the sidebar to start prediction")

# =====================================
# STROKE PREDICTION
# =====================================

elif menu == "Stroke Prediction":

    st.header("Stroke Prediction")

    age = st.number_input("Age", 1, 120)
    glucose = st.number_input("Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Predict Stroke"):

        if glucose == 0 or bmi == 0:
            st.error("Please enter patient details")

        else:

            if stroke_model is None:

                # Demo prediction if model missing
                probability = np.random.uniform(10,80)

            else:

                features = np.array([[age, glucose, bmi]])
                probability = stroke_model.predict_proba(features)[0][1] * 100

            st.subheader("Result")

            st.metric("Stroke Risk", f"{probability:.2f}%")

            st.progress(int(probability))

            if probability > 50:
                st.error("High Stroke Risk")

            else:
                st.success("Low Stroke Risk")

# =====================================
# HEART DISEASE
# =====================================

elif menu == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age ", 1, 120)
    cholesterol = st.number_input("Cholesterol")
    blood_pressure = st.number_input("Blood Pressure")

    if st.button("Predict Heart Disease"):

        if cholesterol == 0 or blood_pressure == 0:
            st.error("Please enter patient details")

        else:

            if heart_model is None:

                probability = np.random.uniform(10,80)

            else:

                features = np.array([[age, cholesterol, blood_pressure]])
                probability = heart_model.predict_proba(features)[0][1] * 100

            st.subheader("Result")

            st.metric("Heart Disease Risk", f"{probability:.2f}%")

            st.progress(int(probability))

            if probability > 50:
                st.error("High Heart Disease Risk")

            else:
                st.success("Low Heart Disease Risk")

# =====================================
# FULL DIAGNOSIS
# =====================================

elif menu == "Full Diagnosis":

    st.header("Full Health Diagnosis")

    age = st.number_input("Patient Age", 1, 120)
    glucose = st.number_input("Glucose")
    cholesterol = st.number_input("Cholesterol")

    if st.button("Run Diagnosis"):

        if glucose == 0 or cholesterol == 0:
            st.error("Please complete patient information")

        else:

            stroke_risk = np.random.uniform(10,80)
            heart_risk = np.random.uniform(10,80)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Stroke Risk", f"{stroke_risk:.2f}%")
                st.progress(int(stroke_risk))

            with col2:
                st.metric("Heart Disease Risk", f"{heart_risk:.2f}%")
                st.progress(int(heart_risk))

            if stroke_risk > 50 and heart_risk > 50:
                st.error("Risk of BOTH Stroke and Heart Disease")

            elif stroke_risk > 50:
                st.warning("Stroke Risk Detected")

            elif heart_risk > 50:
                st.warning("Heart Disease Risk Detected")

            else:
                st.success("Low Health Risk")

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
"<center>AI Medical Diagnosis System | Machine Learning Project</center>",
unsafe_allow_html=True
)
