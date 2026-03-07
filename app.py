import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Load Models
# ----------------------------

heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="AI Health Diagnosis System", layout="wide")

# ----------------------------
# Custom CSS (Professional UI)
# ----------------------------

st.markdown("""
<style>

.main {
    background-color:#f5f7fb;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#0a4da3;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:white;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">AI Health Risk Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Stroke & Heart Disease Diagnosis</p>', unsafe_allow_html=True)

st.write("")

# ----------------------------
# Navigation
# ----------------------------

option = st.sidebar.selectbox(
    "Select Diagnosis",
    ["Stroke Prediction", "Heart Disease Prediction", "Full Health Diagnosis"]
)

# ----------------------------
# STROKE PREDICTION
# ----------------------------

if option == "Stroke Prediction":

    st.header("Stroke Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        hypertension = st.selectbox("Hypertension", [0,1])
        heart_disease = st.selectbox("Existing Heart Disease", [0,1])

    with col2:
        glucose = st.number_input("Average Glucose Level")
        bmi = st.number_input("BMI")

    if st.button("Predict Stroke Risk"):

        data = np.array([[age, hypertension, heart_disease, glucose, bmi]])

        prediction = stroke_model.predict(data)
        probability = stroke_model.predict_proba(data)[0][1] * 100

        st.subheader("Diagnosis Result")

        st.progress(int(probability))

        if prediction[0] == 1:
            st.error(f"Patient is likely to have Stroke (Risk: {probability:.2f}%)")
        else:
            st.success(f"No Stroke Detected (Risk: {probability:.2f}%)")

        st.subheader("Possible Risk Factors")

        causes = []

        if hypertension == 1:
            causes.append("High Blood Pressure")

        if glucose > 140:
            causes.append("High Blood Glucose")

        if bmi > 30:
            causes.append("Obesity")

        if age > 55:
            causes.append("Advanced Age")

        if causes:
            for c in causes:
                st.write("•", c)
        else:
            st.write("No major risk factors detected.")

# ----------------------------
# HEART DISEASE
# ----------------------------

elif option == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
        sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:
        chol = st.number_input("Cholesterol")
        fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
        restecg = st.selectbox("Rest ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate")

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0,1])
        oldpeak = st.number_input("Old Peak")
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Major Vessels", [0,1,2,3])
        thal = st.selectbox("Thal", [0,1,2,3])

    if st.button("Predict Heart Disease"):

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]])

        prediction = heart_model.predict(data)
        probability = heart_model.predict_proba(data)[0][1] * 100

        st.subheader("Diagnosis Result")

        st.progress(int(probability))

        if prediction[0] == 1:
            st.error(f"Patient likely has Heart Disease (Risk: {probability:.2f}%)")
        else:
            st.success(f"No Heart Disease Detected (Risk: {probability:.2f}%)")

        st.subheader("Possible Risk Factors")

        causes = []

        if chol > 240:
            causes.append("High Cholesterol")

        if trestbps > 140:
            causes.append("High Blood Pressure")

        if exang == 1:
            causes.append("Exercise Induced Angina")

        if age > 55:
            causes.append("Advanced Age")

        if causes:
            for c in causes:
                st.write("•", c)

# ----------------------------
# FULL DIAGNOSIS
# ----------------------------

else:

    st.header("Complete Health Diagnosis")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension", [0,1])
    glucose = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Run Full Diagnosis"):

        stroke_data = np.array([[age,hypertension,0,glucose,bmi]])
        stroke_prob = stroke_model.predict_proba(stroke_data)[0][1]*100
        stroke_pred = stroke_model.predict(stroke_data)

        heart_data = np.array([[age,1,1,120,200,0,1,150,0,1,1,0,2]])
        heart_prob = heart_model.predict_proba(heart_data)[0][1]*100
        heart_pred = heart_model.predict(heart_data)

        st.subheader("Final Diagnosis")

        if stroke_pred[0]==1 and heart_pred[0]==1:

            st.error("Patient may have BOTH Stroke and Heart Disease")
            st.write(f"Stroke Risk: {stroke_prob:.2f}%")
            st.write(f"Heart Disease Risk: {heart_prob:.2f}%")

        elif stroke_pred[0]==1:

            st.warning("Patient may have Stroke Only")
            st.write(f"Stroke Risk: {stroke_prob:.2f}%")

        elif heart_pred[0]==1:

            st.warning("Patient may have Heart Disease Only")
            st.write(f"Heart Disease Risk: {heart_prob:.2f}%")

        else:

            st.success("No Stroke or Heart Disease Detected")
            st.write(f"Stroke Risk: {stroke_prob:.2f}%")
            st.write(f"Heart Disease Risk: {heart_prob:.2f}%")
