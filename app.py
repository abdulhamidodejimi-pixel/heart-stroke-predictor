import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="AI Health Risk Predictor", layout="wide")

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

# Title
st.title("🧠 AI Health Risk Prediction System")
st.markdown("### Predict **Heart Disease**, **Stroke**, or **Both**")

st.divider()

# -------------------------
# INPUT SECTION
# -------------------------

st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120)

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])

with col3:
    cholesterol = st.number_input("Cholesterol Level")

col4, col5 = st.columns(2)

with col4:
    blood_pressure = st.number_input("Resting Blood Pressure")

with col5:
    glucose = st.number_input("Glucose Level")

# Convert sex
sex = 1 if sex == "Male" else 0

input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

st.divider()

# -------------------------
# BUTTON SECTION
# -------------------------

col6, col7, col8 = st.columns(3)

heart_button = col6.button("❤️ Predict Heart Disease")
stroke_button = col7.button("🧠 Predict Stroke")
both_button = col8.button("🩺 Predict Both")

st.divider()

# -------------------------
# GAUGE FUNCTION
# -------------------------

def show_gauge(title, value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.3}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# HEART PREDICTION
# -------------------------

if heart_button:

    try:
        prob = heart_model.predict_proba(input_data)[0][1]
        percent = round(prob * 100, 2)

        st.subheader("Heart Disease Result")

        show_gauge("Heart Disease Risk %", percent)

        if percent > 50:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except:
        st.error("Input does not match the heart model requirements.")

# -------------------------
# STROKE PREDICTION
# -------------------------

if stroke_button:

    try:
        prob = stroke_model.predict_proba(input_data)[0][1]
        percent = round(prob * 100, 2)

        st.subheader("Stroke Result")

        show_gauge("Stroke Risk %", percent)

        if percent > 50:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

    except:
        st.error("Input does not match the stroke model requirements.")

# -------------------------
# BOTH PREDICTION
# -------------------------

if both_button:

    try:

        heart_prob = heart_model.predict_proba(input_data)[0][1]
        stroke_prob = stroke_model.predict_proba(input_data)[0][1]

        heart_percent = round(heart_prob * 100, 2)
        stroke_percent = round(stroke_prob * 100, 2)

        col9, col10 = st.columns(2)

        with col9:
            st.subheader("Heart Risk")
            show_gauge("Heart Disease %", heart_percent)

        with col10:
            st.subheader("Stroke Risk")
            show_gauge("Stroke %", stroke_percent)

        st.divider()

        if heart_percent > 50 and stroke_percent > 50:
            st.warning("⚠️ Patient is at risk of BOTH Heart Disease and Stroke")

        elif heart_percent > 50:
            st.warning("⚠️ Patient is mainly at risk of Heart Disease")

        elif stroke_percent > 50:
            st.warning("⚠️ Patient is mainly at risk of Stroke")

        else:
            st.success("✅ Patient shows low risk for both conditions")

    except:
        st.error("Input format does not match the models.")
