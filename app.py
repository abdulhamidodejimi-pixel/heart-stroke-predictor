import streamlit as st
import numpy as np
import joblib

# Load Models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="AI Medical Diagnosis System", layout="wide")

# ---------------------------
# Custom UI Styling
# ---------------------------

st.markdown("""
<style>

body {
background-color:#f5f7fb;
}

.main-title{
text-align:center;
font-size:42px;
font-weight:700;
color:#0a4da3;
}

.sub-title{
text-align:center;
font-size:18px;
color:#6c757d;
margin-bottom:25px;
}

.card{
background:white;
padding:25px;
border-radius:12px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);
margin-bottom:20px;
}

.metric{
font-size:30px;
font-weight:bold;
}

.low-risk{
color:#2ecc71;
font-weight:bold;
}

.medium-risk{
color:#f39c12;
font-weight:bold;
}

.high-risk{
color:#e74c3c;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------

st.markdown('<p class="main-title">🩺 AI Medical Diagnosis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Stroke & Heart Disease Prediction using Machine Learning</p>', unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("Medical AI System")

option = st.sidebar.radio(
"Select Diagnosis",
[
"Stroke Prediction",
"Heart Disease Prediction",
"Full Health Diagnosis"
]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This system predicts:

• Stroke Risk  
• Heart Disease Risk  
• Combined Diagnosis  

Built using Machine Learning.
"""
)

# ---------------------------
# Risk Level Function
# ---------------------------

def risk_level(prob):

    if prob < 30:
        return "Low Risk", "low-risk"
    elif prob < 60:
        return "Moderate Risk", "medium-risk"
    else:
        return "High Risk", "high-risk"


# =====================================================
# STROKE PREDICTION
# =====================================================

if option == "Stroke Prediction":

    st.header("Stroke Risk Prediction")

    with st.form("stroke_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120)
            hypertension = st.selectbox("Hypertension", ["Select",0,1])
            heart_disease = st.selectbox("Existing Heart Disease", ["Select",0,1])

        with col2:
            glucose = st.number_input("Average Glucose Level")
            bmi = st.number_input("BMI")

        submit = st.form_submit_button("Predict Stroke")

    if submit:

        if hypertension == "Select" or heart_disease == "Select" or glucose == 0 or bmi == 0:

            st.warning("Please enter all patient details before prediction.")

        else:

            data = np.array([[age, hypertension, heart_disease, glucose, bmi]])

            prediction = stroke_model.predict(data)
            probability = stroke_model.predict_proba(data)[0][1] * 100

            risk, style = risk_level(probability)

            st.markdown("### Diagnosis Result")

            st.progress(int(probability))

            st.markdown(f"""
            <div class="card">
            <p class="metric">{probability:.2f}%</p>
            <p class="{style}">{risk}</p>
            </div>
            """, unsafe_allow_html=True)

            if prediction[0] == 1:
                st.error("Patient is likely to have Stroke")
            else:
                st.success("No Stroke Detected")

            st.markdown("### Patient Summary")

            c1,c2 = st.columns(2)

            with c1:
                st.write("Age:",age)
                st.write("BMI:",bmi)

            with c2:
                st.write("Glucose Level:",glucose)
                st.write("Hypertension:",hypertension)


# =====================================================
# HEART DISEASE PREDICTION
# =====================================================

elif option == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    with st.form("heart_form"):

        col1,col2,col3 = st.columns(3)

        with col1:
            age = st.number_input("Age")
            sex = st.selectbox("Sex", ["Select",0,1])
            cp = st.selectbox("Chest Pain Type", ["Select",0,1,2,3])
            trestbps = st.number_input("Resting Blood Pressure")

        with col2:
            chol = st.number_input("Cholesterol")
            fbs = st.selectbox("Fasting Blood Sugar >120", ["Select",0,1])
            restecg = st.selectbox("Rest ECG", ["Select",0,1,2])
            thalach = st.number_input("Maximum Heart Rate")

        with col3:
            exang = st.selectbox("Exercise Induced Angina", ["Select",0,1])
            oldpeak = st.number_input("Old Peak")
            slope = st.selectbox("Slope", ["Select",0,1,2])
            ca = st.selectbox("Major Vessels", ["Select",0,1,2,3])
            thal = st.selectbox("Thal", ["Select",0,1,2,3])

        submit = st.form_submit_button("Predict Heart Disease")

    if submit:

        if "Select" in [sex,cp,fbs,restecg,exang,slope,ca,thal] or chol==0 or trestbps==0:

            st.warning("Please fill all patient details.")

        else:

            data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                              thalach,exang,oldpeak,slope,ca,thal]])

            prediction = heart_model.predict(data)
            probability = heart_model.predict_proba(data)[0][1]*100

            risk, style = risk_level(probability)

            st.markdown("### Diagnosis Result")

            st.progress(int(probability))

            st.markdown(f"""
            <div class="card">
            <p class="metric">{probability:.2f}%</p>
            <p class="{style}">{risk}</p>
            </div>
            """, unsafe_allow_html=True)

            if prediction[0]==1:
                st.error("Patient likely has Heart Disease")
            else:
                st.success("No Heart Disease Detected")


# =====================================================
# FULL HEALTH DIAGNOSIS
# =====================================================

else:

    st.header("Complete Health Diagnosis")

    with st.form("full_form"):

        age = st.number_input("Age")
        hypertension = st.selectbox("Hypertension", ["Select",0,1])
        glucose = st.number_input("Average Glucose Level")
        bmi = st.number_input("BMI")

        submit = st.form_submit_button("Run Diagnosis")

    if submit:

        if hypertension=="Select" or glucose==0 or bmi==0:

            st.warning("Please fill all patient details.")

        else:

            stroke_data = np.array([[age,hypertension,0,glucose,bmi]])

            stroke_prob = stroke_model.predict_proba(stroke_data)[0][1]*100
            stroke_pred = stroke_model.predict(stroke_data)

            heart_data = np.array([[age,1,1,120,200,0,1,150,0,1,1,0,2]])

            heart_prob = heart_model.predict_proba(heart_data)[0][1]*100
            heart_pred = heart_model.predict(heart_data)

            st.markdown("### Final Diagnosis")

            st.write(f"Stroke Risk: {stroke_prob:.2f}%")
            st.write(f"Heart Disease Risk: {heart_prob:.2f}%")

            if stroke_pred[0]==1 and heart_pred[0]==1:

                st.error("Patient may have BOTH Stroke and Heart Disease")

            elif stroke_pred[0]==1:

                st.warning("Patient may have Stroke Only")

            elif heart_pred[0]==1:

                st.warning("Patient may have Heart Disease Only")

            else:

                st.success("No Stroke or Heart Disease Detected")


# ---------------------------
# Footer
# ---------------------------

st.markdown("---")

st.markdown(
"""
<center>

AI Medical Diagnosis System  
Machine Learning Project  

Built with Streamlit & Scikit-Learn

</center>
""",
unsafe_allow_html=True
)
