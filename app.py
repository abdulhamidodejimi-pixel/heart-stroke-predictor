import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="Heart & Stroke Predictor", layout="centered")

st.title("🫀 Heart Disease & Stroke Prediction System")

st.write("Enter patient health details below to assess risk.")

# ----------------------------
# USER INPUTS
# ----------------------------

age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

chol = st.number_input("Cholesterol Level")

trestbps = st.number_input("Resting Blood Pressure")

glucose = st.number_input("Glucose Level")

# Convert to model input
input_data = np.array([[age, sex, chol, trestbps, glucose]])

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict Health Risk"):

    try:

        # HEART PREDICTION
        heart_prob = heart_model.predict_proba(input_data)[0][1]
        heart_percent = round(heart_prob * 100, 2)

        # STROKE PREDICTION
        stroke_prob = stroke_model.predict_proba(input_data)[0][1]
        stroke_percent = round(stroke_prob * 100, 2)

        # ----------------------------
        # HEART RESULT
        # ----------------------------

        st.subheader("Heart Disease Risk")

        heart_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=heart_percent,
            title={'text': "Heart Risk %"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(heart_fig)

        if heart_percent > 50:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        # ----------------------------
        # STROKE RESULT
        # ----------------------------

        st.subheader("Stroke Risk")

        stroke_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stroke_percent,
            title={'text': "Stroke Risk %"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(stroke_fig)

        if stroke_percent > 50:
            st.error("High Risk of Stroke")
        else:
            st.success("Low Risk of Stroke")

        # ----------------------------
        # BOTH CONDITION CHECK
        # ----------------------------

        if heart_percent > 50 and stroke_percent > 50:
            st.warning("Patient is at risk of BOTH Heart Disease and Stroke")

        elif heart_percent > 50:
            st.warning("Patient is mainly at risk of Heart Disease")

        elif stroke_percent > 50:
            st.warning("Patient is mainly at risk of Stroke")

        else:
            st.info("Patient shows low risk for both conditions")

    except Exception as e:
        st.error("⚠️ Input format does not match model requirements.")
