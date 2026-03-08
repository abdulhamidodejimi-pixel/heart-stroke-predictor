import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import os

st.set_page_config(page_title="AI Heart & Stroke Predictor", layout="wide")

# -----------------------------
# LOAD MODELS SAFELY
# -----------------------------
def load_model(file):
    if not os.path.exists(file):
        st.error(f"Model file '{file}' not found. Upload it to your GitHub repository.")
        return None
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return None

heart_model = load_model("heart_model.pkl")
stroke_model = load_model("stroke_model.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 AI Heart & Stroke Risk Prediction System")
st.markdown("Professional AI medical risk assessment system.")

# -----------------------------
# PATIENT DETAILS
# -----------------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.number_input("Chest Pain Type", min_value=0, max_value=3)
trestbps = st.sidebar.number_input("Resting Blood Pressure")
chol = st.sidebar.number_input("Cholesterol")
fbs = st.sidebar.number_input("Fasting Blood Sugar")
thalach = st.sidebar.number_input("Max Heart Rate")
oldpeak = st.sidebar.number_input("ST Depression")

predict_btn = st.sidebar.button("Predict Risk")

# -----------------------------
# RISK GAUGE
# -----------------------------
def risk_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PDF REPORT
# -----------------------------
def generate_pdf(heart_result, stroke_result):
    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("AI Medical Prediction Report", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Heart Disease Risk: {heart_result}", styles['Normal']))
    story.append(Paragraph(f"Stroke Risk: {stroke_result}", styles['Normal']))

    doc = SimpleDocTemplate(buffer)
    doc.build(story)

    buffer.seek(0)
    return buffer

# -----------------------------
# PREDICTION
# -----------------------------
if predict_btn:

    if heart_model is None or stroke_model is None:
        st.error("Models not loaded. Check GitHub files.")
        st.stop()

    if age == 0 or trestbps == 0 or chol == 0:
        st.warning("Please fill patient details before prediction.")
        st.stop()

    sex_val = 1 if sex == "Male" else 0

    heart_features = np.array([[age, sex_val, cp, trestbps, chol, fbs, thalach, oldpeak]])

    try:
        heart_pred = heart_model.predict(heart_features)[0]
        heart_prob = heart_model.predict_proba(heart_features)[0][1]*100
    except:
        heart_pred = 0
        heart_prob = 0

    try:
        stroke_pred = stroke_model.predict(heart_features)[0]
        stroke_prob = stroke_model.predict_proba(heart_features)[0][1]*100
    except:
        stroke_pred = 0
        stroke_prob = 0

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        risk_gauge(heart_prob, "Heart Disease Risk %")

    with col2:
        risk_gauge(stroke_prob, "Stroke Risk %")

    # AI explanation
    st.subheader("AI Explanation")

    if heart_pred == 1:
        st.error("The model detected signs associated with heart disease risk.")
    else:
        st.success("Low heart disease risk detected.")

    if stroke_pred == 1:
        st.error("The model detected stroke risk factors.")
    else:
        st.success("Low stroke risk detected.")

    # PDF download
    pdf = generate_pdf(heart_prob, stroke_prob)

    st.download_button(
        label="Download Medical Report (PDF)",
        data=pdf,
        file_name="medical_report.pdf",
        mime="application/pdf"
    )
