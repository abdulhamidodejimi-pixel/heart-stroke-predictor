import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

st.title("📊 Health Risk Dashboard")

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol", 0.0, 600.0, 200.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0.0, 300.0, 120.0)
glucose = st.sidebar.number_input("Glucose", 0.0, 500.0, 100.0)

sex = 1 if sex == "Male" else 0
input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

if st.button("🔍 Predict Risk"):
    heart_prob = heart_model.predict_proba(input_data)[0][1]
    stroke_prob = stroke_model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=heart_prob*100,
            title={'text': "Heart Risk (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stroke_prob*100,
            title={'text': "Stroke Risk (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🩺 Health Tips")

    if heart_prob > 0.5:
        st.warning("• Improve diet\n• Exercise regularly\n• Reduce salt\n• Regular checkups")
    else:
        st.success("Heart health looks stable. Maintain healthy habits.")

    if stroke_prob > 0.5:
        st.warning("• Monitor blood pressure\n• Avoid smoking\n• Control cholesterol")
    else:
        st.success("Stroke risk appears low. Continue preventive care.")
