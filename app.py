import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Medical Diagnosis System", page_icon="🩺", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("🩺 AI Medical Diagnosis System")
st.write("Machine Learning System for Stroke & Heart Disease Risk Prediction")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Patient Menu")

menu = st.sidebar.radio(
"Select Section",
[
"Patient Details",
"Stroke Prediction",
"Heart Disease Prediction",
"Full Diagnosis"
]
)

# -----------------------------
# PATIENT DETAILS
# -----------------------------
if menu == "Patient Details":

    st.header("Patient Information")

    col1,col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age",1,120)

    with col2:
        gender = st.selectbox("Gender",["Male","Female"])
        bmi = st.number_input("BMI")

    st.success("Patient information recorded")

# -----------------------------
# RISK GAUGE FUNCTION
# -----------------------------
def gauge_chart(probability):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Risk Level (%)"},
        gauge={
            'axis': {'range': [0,100]},
            'steps':[
                {'range':[0,30],'color':"green"},
                {'range':[30,60],'color':"orange"},
                {'range':[60,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# AI EXPLANATION
# -----------------------------
def explain_prediction(model,feature_names,input_data):

    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":importances
    })

    df = df.sort_values("Importance",ascending=False)

    st.subheader("AI Explanation (Top Influencing Factors)")
    st.bar_chart(df.set_index("Feature"))

# =====================================================
# STROKE PREDICTION
# =====================================================
elif menu == "Stroke Prediction":

    st.header("Stroke Risk Prediction")

    age = st.number_input("Age",1,120)
    hypertension = st.selectbox("Hypertension",[0,1])
    heart_disease = st.selectbox("Heart Disease",[0,1])
    glucose = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Predict Stroke"):

        data = np.array([[age,hypertension,heart_disease,glucose,bmi]])

        pred = stroke_model.predict(data)
        prob = stroke_model.predict_proba(data)[0][1]*100

        st.subheader("Stroke Risk Gauge")
        gauge_chart(prob)

        if pred[0]==1:
            st.error("Patient likely at risk of Stroke")
        else:
            st.success("Low Stroke Risk")

        explain_prediction(
            stroke_model,
            ["Age","Hypertension","Heart Disease","Glucose","BMI"],
            data
        )

# =====================================================
# HEART DISEASE PREDICTION
# =====================================================
elif menu == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex",[0,1])
    cp = st.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar",[0,1])
    restecg = st.selectbox("Rest ECG",[0,1,2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Angina",[0,1])
    oldpeak = st.number_input("Old Peak")
    slope = st.selectbox("Slope",[0,1,2])
    ca = st.selectbox("Major Vessels",[0,1,2,3])
    thal = st.selectbox("Thal",[0,1,2,3])

    if st.button("Predict Heart Disease"):

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]])

        pred = heart_model.predict(data)
        prob = heart_model.predict_proba(data)[0][1]*100

        st.subheader("Heart Disease Risk Gauge")
        gauge_chart(prob)

        if pred[0]==1:
            st.error("Patient likely has Heart Disease")
        else:
            st.success("Low Heart Disease Risk")

        explain_prediction(
            heart_model,
            ["Age","Sex","ChestPain","BP","Cholesterol","FBS",
             "ECG","MaxHR","Angina","OldPeak","Slope","Vessels","Thal"],
            data
        )

# =====================================================
# FULL DIAGNOSIS
# =====================================================
elif menu == "Full Diagnosis":

    st.header("Complete Health Diagnosis")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension",[0,1])
    glucose = st.number_input("Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Run Diagnosis"):

        stroke_data = np.array([[age,hypertension,0,glucose,bmi]])
        stroke_prob = stroke_model.predict_proba(stroke_data)[0][1]*100
        stroke_pred = stroke_model.predict(stroke_data)

        heart_data = np.array([[age,1,1,120,200,0,1,150,0,1,1,0,2]])
        heart_prob = heart_model.predict_proba(heart_data)[0][1]*100
        heart_pred = heart_model.predict(heart_data)

        st.subheader("Stroke Risk")
        gauge_chart(stroke_prob)

        st.subheader("Heart Disease Risk")
        gauge_chart(heart_prob)

        if stroke_pred[0]==1 and heart_pred[0]==1:
            diagnosis="Both Stroke and Heart Disease Risk"
        elif stroke_pred[0]==1:
            diagnosis="Stroke Risk"
        elif heart_pred[0]==1:
            diagnosis="Heart Disease Risk"
        else:
            diagnosis="Low Risk"

        st.success(diagnosis)

        # -----------------------------
        # PDF REPORT
        # -----------------------------
        if st.button("Download Medical Report"):

            styles = getSampleStyleSheet()

            report = SimpleDocTemplate("medical_report.pdf")

            story = []

            story.append(Paragraph("AI Medical Diagnosis Report",styles['Title']))
            story.append(Spacer(1,20))
            story.append(Paragraph(f"Stroke Risk: {stroke_prob:.2f}%",styles['Normal']))
            story.append(Paragraph(f"Heart Disease Risk: {heart_prob:.2f}%",styles['Normal']))
            story.append(Paragraph(f"Final Diagnosis: {diagnosis}",styles['Normal']))

            report.build(story)

            with open("medical_report.pdf","rb") as file:
                st.download_button(
                    label="Download PDF Report",
                    data=file,
                    file_name="patient_diagnosis_report.pdf"
                )
