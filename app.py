import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="AI Health Risk Predictor", layout="wide")

st.title("AI Heart Disease & Stroke Prediction System")

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

# Layout
left, right = st.columns([1,2])

with left:

    st.header("Patient Details")

    age = st.number_input("Age",1,120)

    sex = st.selectbox("Sex",["Male","Female"])
    sex = 1 if sex=="Male" else 0

    cholesterol = st.number_input("Cholesterol")

    blood_pressure = st.number_input("Blood Pressure")

    glucose = st.number_input("Glucose Level")

    st.divider()

    heart_btn = st.button("Predict Heart Disease")
    stroke_btn = st.button("Predict Stroke")
    both_btn = st.button("Predict Both")

# Gauge function
def gauge(title,value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text':title},
        gauge={
            'axis':{'range':[0,100]},
            'steps':[
                {'range':[0,30],'color':"green"},
                {'range':[30,60],'color':"yellow"},
                {'range':[60,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

with right:

    st.header("Prediction Results")

    # HEART INPUT (13 features)
    heart_input = np.array([[

        age,
        sex,
        0,              # cp
        blood_pressure,
        cholesterol,
        0,              # fbs
        0,              # restecg
        150,            # thalach
        0,              # exang
        1.0,            # oldpeak
        1,              # slope
        0,              # ca
        2               # thal

    ]])

    # STROKE INPUT
    stroke_input = np.array([[

        sex,
        age,
        0,              # hypertension
        0,              # heart_disease
        1,              # married
        2,              # work_type
        1,              # residence
        glucose,
        25,             # bmi
        1               # smoking

    ]])

    if heart_btn:

        prob = heart_model.predict_proba(heart_input)[0][1]
        percent = round(prob*100,2)

        gauge("Heart Disease Risk %",percent)

        if percent > 50:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

    if stroke_btn:

        prob = stroke_model.predict_proba(stroke_input)[0][1]
        percent = round(prob*100,2)

        gauge("Stroke Risk %",percent)

        if percent > 50:
            st.error("High Risk of Stroke")
        else:
            st.success("Low Risk of Stroke")

    if both_btn:

        heart_prob = heart_model.predict_proba(heart_input)[0][1]
        stroke_prob = stroke_model.predict_proba(stroke_input)[0][1]

        heart_percent = round(heart_prob*100,2)
        stroke_percent = round(stroke_prob*100,2)

        col1,col2 = st.columns(2)

        with col1:
            gauge("Heart Risk %",heart_percent)

        with col2:
            gauge("Stroke Risk %",stroke_percent)

        if heart_percent>50 and stroke_percent>50:
            st.error("Patient at risk of BOTH Heart Disease and Stroke")

        elif heart_percent>50:
            st.warning("Patient likely has Heart Disease but not Stroke")

        elif stroke_percent>50:
            st.warning("Patient likely has Stroke but not Heart Disease")

        else:
            st.success("Patient shows low risk for both diseases")
