import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:bold;
color:#0f172a;
}

.subtitle{
font-size:18px;
color:gray;
}

.card{
background-color:#f8fafc;
padding:20px;
border-radius:12px;
box-shadow:0px 2px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<p class="main-title">🧠 AI Heart & Stroke Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning system to detect Heart Disease and Stroke Risk</p>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([1,2])

# -----------------------------
# LEFT SIDE (PATIENT INPUT)
# -----------------------------
with left:

    st.markdown("### Patient Information")

    age = st.number_input("Age", 1, 120)

    sex = st.selectbox("Sex", ["Male","Female"])
    sex = 1 if sex=="Male" else 0

    cholesterol = st.number_input("Cholesterol Level")

    blood_pressure = st.number_input("Resting Blood Pressure")

    glucose = st.number_input("Glucose Level")

    st.divider()

    st.markdown("### Prediction")

    heart_btn = st.button("❤️ Heart Disease")

    stroke_btn = st.button("🧠 Stroke")

    both_btn = st.button("🩺 Both")

# -----------------------------
# INPUT ARRAY
# -----------------------------
input_data = np.array([[age,sex,cholesterol,blood_pressure,glucose]])

# -----------------------------
# GAUGE FUNCTION
# -----------------------------
def risk_gauge(title,value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color':"#ef4444"},
            'steps':[
                {'range':[0,30],'color':"#22c55e"},
                {'range':[30,60],'color':"#facc15"},
                {'range':[60,100],'color':"#ef4444"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# RIGHT SIDE (RESULTS)
# -----------------------------
with right:

    st.markdown("### Prediction Results")

    if heart_btn:

        try:

            prob = heart_model.predict_proba(input_data)[0][1]
            percent = round(prob*100,2)

            risk_gauge("Heart Disease Risk %",percent)

            if percent > 50:
                st.error("⚠ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

        except:
            st.warning("Input format does not match the heart model.")


    if stroke_btn:

        try:

            prob = stroke_model.predict_proba(input_data)[0][1]
            percent = round(prob*100,2)

            risk_gauge("Stroke Risk %",percent)

            if percent > 50:
                st.error("⚠ High Risk of Stroke")
            else:
                st.success("✅ Low Risk of Stroke")

        except:
            st.warning("Input format does not match the stroke model.")


    if both_btn:

        try:

            heart_prob = heart_model.predict_proba(input_data)[0][1]
            stroke_prob = stroke_model.predict_proba(input_data)[0][1]

            heart_percent = round(heart_prob*100,2)
            stroke_percent = round(stroke_prob*100,2)

            col1,col2 = st.columns(2)

            with col1:
                risk_gauge("Heart Disease Risk %",heart_percent)

            with col2:
                risk_gauge("Stroke Risk %",stroke_percent)

            st.divider()

            if heart_percent>50 and stroke_percent>50:
                st.error("⚠ Patient is at risk of BOTH Heart Disease and Stroke")

            elif heart_percent>50:
                st.warning("⚠ Patient mainly at risk of Heart Disease")

            elif stroke_percent>50:
                st.warning("⚠ Patient mainly at risk of Stroke")

            else:
                st.success("✅ Patient shows LOW risk for both conditions")

        except:
            st.warning("Input format does not match the models.")
