import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Heart & Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            font-size:40px;
            font-weight:bold;
            color:#d62828;
        }
        .subtitle {
            font-size:18px;
            color:#444;
        }
        .footer {
            font-size:14px;
            text-align:center;
            margin-top:50px;
            color:gray;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<p class="title">🩺 Heart & Stroke Risk Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered risk assessment using Machine Learning</p>', unsafe_allow_html=True)
st.divider()

# -------------------------------
# Load Models
# -------------------------------
try:
    heart_model = joblib.load("heart_model.pkl")
    stroke_model = joblib.load("stroke_model.pkl")
except:
    st.error("⚠️ Model files not found.")
    st.stop()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("👤 Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol Level", 0.0, 600.0, 200.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0.0, 300.0, 120.0)
glucose = st.sidebar.number_input("Glucose Level", 0.0, 500.0, 100.0)

st.sidebar.markdown("---")
st.sidebar.info("Enter patient details and select prediction type.")

# Convert sex
sex = 1 if sex == "Male" else 0

input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔍 Prediction Panel")

col1, col2, col3 = st.columns(3)

# HEART
with col1:
    if st.button("❤️ Predict Heart"):
        heart_pred = heart_model.predict(input_data)
        heart_prob = heart_model.predict_proba(input_data)[0][1]

        st.metric("Heart Risk", f"{heart_prob*100:.2f}%")

        if heart_pred[0] == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

# STROKE
with col2:
    if st.button("🧠 Predict Stroke"):
        stroke_pred = stroke_model.predict(input_data)
        stroke_prob = stroke_model.predict_proba(input_data)[0][1]

        st.metric("Stroke Risk", f"{stroke_prob*100:.2f}%")

        if stroke_pred[0] == 1:
            st.error("High Risk of Stroke")
        else:
            st.success("Low Risk of Stroke")

# BOTH
with col3:
    if st.button("📊 Predict Both"):
        heart_pred = heart_model.predict(input_data)
        heart_prob = heart_model.predict_proba(input_data)[0][1]

        stroke_pred = stroke_model.predict(input_data)
        stroke_prob = stroke_model.predict_proba(input_data)[0][1]

        st.markdown("### Combined Risk Assessment")

        colh, cols = st.columns(2)

        with colh:
            st.metric("Heart Risk", f"{heart_prob*100:.2f}%")
            if heart_pred[0] == 1:
                st.error("High Risk")
            else:
                st.success("Low Risk")

        with cols:
            st.metric("Stroke Risk", f"{stroke_prob*100:.2f}%")
            if stroke_pred[0] == 1:
                st.error("High Risk")
            else:
                st.success("Low Risk")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    '<div class="footer">Built with Streamlit & Scikit-Learn | Machine Learning Health Risk Predictor</div>',
    unsafe_allow_html=True
)
