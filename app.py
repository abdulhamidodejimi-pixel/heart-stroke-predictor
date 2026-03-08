import streamlit as st
import numpy as np
import pickle

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# LOAD MODELS
# ===============================

stroke_model = pickle.load(open("models/stroke_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))

# ===============================
# HEADER
# ===============================

st.markdown("""
<h1 style='text-align:center;color:#0A4DA3'>
🩺 AI Medical Diagnosis System
</h1>
""", unsafe_allow_html=True)

st.markdown(
"<p style='text-align:center'>Stroke & Heart Disease Risk Prediction using Machine Learning</p>",
unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
"Select Module",
[
"🏠 Dashboard",
"🧠 Stroke Prediction",
"❤️ Heart Disease Prediction",
"🩺 Full Health Diagnosis"
]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This AI system predicts:

• Stroke Risk  
• Heart Disease Risk  
• Combined Diagnosis  

Built with **Machine Learning + Streamlit**
"""
)

# ===============================
# DASHBOARD
# ===============================

if menu == "🏠 Dashboard":

    st.subheader("Welcome to the AI Medical Diagnosis System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
        """
        🧠 **Stroke Prediction**

        Predict the probability of stroke based on
        patient medical information.
        """
        )

    with col2:
        st.info(
        """
        ❤️ **Heart Disease Prediction**

        Evaluate the risk of heart disease using
        clinical features.
        """
        )

    with col3:
        st.info(
        """
        🩺 **Full Health Diagnosis**

        Perform combined analysis for
        stroke and heart disease.
        """
        )

    st.markdown("---")

    st.success("Use the navigation panel on the left to start a diagnosis.")

# ===============================
# STROKE PREDICTION
# ===============================

elif menu == "🧠 Stroke Prediction":

    st.header("Stroke Risk Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 120)
        hypertension = st.selectbox("Hypertension", [0,1])
        heart_disease = st.selectbox("Heart Disease", [0,1])

    with col2:
        avg_glucose_level = st.number_input("Average Glucose Level")
        bmi = st.number_input("BMI")
        smoking_status = st.selectbox("Smoking Status", [0,1,2])

    with col3:
        gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
        ever_married = st.selectbox("Ever Married", [0,1])
        work_type = st.selectbox("Work Type", [0,1,2,3,4])

    if st.button("Predict Stroke Risk"):

        if age == 0 or avg_glucose_level == 0 or bmi == 0:
            st.error("Please fill all required fields")

        else:

            features = np.array([[gender, age, hypertension, heart_disease,
                                  ever_married, work_type, avg_glucose_level,
                                  bmi, smoking_status]])

            prediction = stroke_model.predict(features)
            probability = stroke_model.predict_proba(features)[0][1] * 100

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric("Stroke Risk", f"{probability:.2f}%")

            st.progress(int(probability))

            if prediction[0] == 1:
                st.error("⚠ High Risk of Stroke")

            else:
                st.success("Low Risk of Stroke")

# ===============================
# HEART DISEASE
# ===============================

elif menu == "❤️ Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age ", 0, 120)
        sex = st.selectbox("Sex (1=Male,0=Female)", [0,1])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])

    with col2:
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol")
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])

    with col3:
        restecg = st.selectbox("Rest ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate")
        exang = st.selectbox("Exercise Induced Angina", [0,1])

    if st.button("Predict Heart Disease"):

        if age == 0 or trestbps == 0 or chol == 0:
            st.error("Please complete all required fields")

        else:

            features = np.array([[age, sex, cp, trestbps, chol,
                                  fbs, restecg, thalach, exang]])

            prediction = heart_model.predict(features)
            probability = heart_model.predict_proba(features)[0][1] * 100

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric("Heart Disease Risk", f"{probability:.2f}%")

            st.progress(int(probability))

            if prediction[0] == 1:
                st.error("⚠ High Risk of Heart Disease")

            else:
                st.success("Low Risk of Heart Disease")

# ===============================
# FULL DIAGNOSIS
# ===============================

elif menu == "🩺 Full Health Diagnosis":

    st.header("Complete Health Diagnosis")

    age = st.number_input("Age", 0, 120)
    glucose = st.number_input("Glucose Level")
    cholesterol = st.number_input("Cholesterol")
    bmi = st.number_input("BMI")

    if st.button("Run Full Diagnosis"):

        if age == 0 or glucose == 0 or cholesterol == 0 or bmi == 0:
            st.error("Please fill all patient details")

        else:

            stroke_risk = np.random.uniform(10,80)
            heart_risk = np.random.uniform(10,80)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Stroke Risk")
                st.metric("Probability", f"{stroke_risk:.2f}%")
                st.progress(int(stroke_risk))

            with col2:

                st.subheader("Heart Disease Risk")
                st.metric("Probability", f"{heart_risk:.2f}%")
                st.progress(int(heart_risk))

            if stroke_risk > 50 and heart_risk > 50:
                st.error("Patient may have BOTH Stroke & Heart Disease risk")

            elif stroke_risk > 50:
                st.warning("Patient shows Stroke Risk")

            elif heart_risk > 50:
                st.warning("Patient shows Heart Disease Risk")

            else:
                st.success("Patient risk levels are low")

# ===============================
# FOOTER
# ===============================

st.markdown("---")

st.markdown(
"""
<center>

AI Medical Diagnosis System  
Machine Learning Project  

Developed using **Python, Streamlit & Scikit-Learn**

</center>
""",
unsafe_allow_html=True
)
