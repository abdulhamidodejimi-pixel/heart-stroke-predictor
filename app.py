import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🧠 AI Stroke & Heart Disease Prediction System")
st.markdown("### Intelligent Medical Risk Assessment Platform")

# -----------------------------
# LOAD MODELS
# -----------------------------
stroke_model = pickle.load(open("stroke_model.pkl", "rb"))
heart_model = pickle.load(open("heart_model.pkl", "rb"))

# -----------------------------
# PATIENT DETAILS
# -----------------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120)
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease_input = st.sidebar.selectbox("Previous Heart Disease", [0,1])
glucose = st.sidebar.number_input("Glucose Level")
bmi = st.sidebar.number_input("BMI")

sex = st.sidebar.selectbox("Sex", [0,1])
cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
chol = st.sidebar.number_input("Cholesterol")
thalach = st.sidebar.number_input("Max Heart Rate")

predict_btn = st.sidebar.button("Predict Disease")

# -----------------------------
# FUNCTIONS
# -----------------------------

def gauge_chart(value, title):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color':"red"},
            'steps':[
                {'range':[0,30],'color':"green"},
                {'range':[30,60],'color':"yellow"},
                {'range':[60,100],'color':"red"}
            ],
        }
    ))

    return fig


def generate_explanation(age, bmi, glucose):

    explanation = []

    if age > 55:
        explanation.append("Advanced age increases cardiovascular risk.")

    if bmi > 30:
        explanation.append("High BMI indicates obesity risk.")

    if glucose > 140:
        explanation.append("High glucose suggests diabetes risk.")

    if len(explanation) == 0:
        explanation.append("No major risk factors detected.")

    return explanation


def generate_pdf(stroke_risk, heart_risk):

    styles = getSampleStyleSheet()

    temp = tempfile.NamedTemporaryFile(delete=False)

    doc = SimpleDocTemplate(temp.name)

    elements = []

    elements.append(Paragraph("AI Medical Risk Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Stroke Risk: {stroke_risk:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"Heart Disease Risk: {heart_risk:.2f}%", styles['Normal']))

    doc.build(elements)

    return temp.name

# -----------------------------
# PREDICTION
# -----------------------------

if predict_btn:

    stroke_data = np.array([[age, hypertension, heart_disease_input, glucose, bmi]])
    heart_data = np.array([[age, sex, cp, chol, thalach]])

    stroke_prob = stroke_model.predict_proba(stroke_data)[0][1] * 100
    heart_prob = heart_model.predict_proba(heart_data)[0][1] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stroke Risk")
        st.plotly_chart(gauge_chart(stroke_prob,"Stroke Risk %"))

    with col2:
        st.subheader("Heart Disease Risk")
        st.plotly_chart(gauge_chart(heart_prob,"Heart Disease Risk %"))

    # -----------------------------
    # FINAL RESULT
    # -----------------------------

    st.header("Diagnosis")

    if stroke_prob > 50 and heart_prob > 50:
        st.error("⚠ Patient is at risk of BOTH Stroke and Heart Disease")

    elif stroke_prob > 50:
        st.warning("⚠ Patient is at risk of Stroke")

    elif heart_prob > 50:
        st.warning("⚠ Patient is at risk of Heart Disease")

    else:
        st.success("✅ Patient has low risk")

    # -----------------------------
    # AI EXPLANATION
    # -----------------------------

    st.header("AI Explanation")

    explanation = generate_explanation(age,bmi,glucose)

    for e in explanation:
        st.write("•", e)

    # -----------------------------
    # PDF REPORT
    # -----------------------------

    st.header("Download Medical Report")

    pdf_file = generate_pdf(stroke_prob, heart_prob)

    with open(pdf_file,"rb") as f:

        st.download_button(
            label="Download Patient Report",
            data=f,
            file_name="medical_report.pdf",
            mime="application/pdf"
        )
