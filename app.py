import streamlit as st
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# LOAD MODELS
# -----------------------------

heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
color:#0A4DA3;
text-align:center;
}

.sub-title{
text-align:center;
color:gray;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;
border-radius:12px;
box-shadow:0px 4px 15px rgba(0,0,0,0.08);
margin-top:15px;
}

.low{
color:green;
font-weight:600;
}

.medium{
color:orange;
font-weight:600;
}

.high{
color:red;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.markdown('<p class="main-title">AI Medical Diagnosis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Stroke & Heart Disease Risk Prediction using Machine Learning</p>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
"Select Service",
[
"Stroke Prediction",
"Heart Disease Prediction",
"Full Health Diagnosis"
]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
AI Medical System

This system predicts:

• Stroke Risk  
• Heart Disease Risk  
• Combined Diagnosis  

Using trained Machine Learning models.
"""
)

# -----------------------------
# RISK LEVEL FUNCTION
# -----------------------------

def risk_level(prob):

    if prob < 30:
        return "Low Risk","low"

    elif prob < 60:
        return "Moderate Risk","medium"

    else:
        return "High Risk","high"


# =====================================================
# STROKE PREDICTION
# =====================================================

if menu == "Stroke Prediction":

    st.header("Stroke Risk Prediction")

    with st.form("stroke_form"):

        col1,col2 = st.columns(2)

        with col1:
            age = st.number_input("Age",1,120)
            hypertension = st.selectbox("Hypertension",[None,0,1])
            heart_disease = st.selectbox("Existing Heart Disease",[None,0,1])

        with col2:
            glucose = st.number_input("Average Glucose Level")
            bmi = st.number_input("BMI")

        submit = st.form_submit_button("Predict Stroke Risk")

    if submit:

        if None in [hypertension,heart_disease] or glucose==0 or bmi==0:

            st.warning("Please fill all patient information.")

        else:

            data = np.array([[age,hypertension,heart_disease,glucose,bmi]])

            prediction = stroke_model.predict(data)
            probability = stroke_model.predict_proba(data)[0][1]*100

            level,style = risk_level(probability)

            st.subheader("Diagnosis Result")

            st.progress(int(probability))

            st.markdown(f"""
            <div class="card">
            <h3>Stroke Risk Probability</h3>
            <h1>{probability:.2f}%</h1>
            <p class="{style}">{level}</p>
            </div>
            """,unsafe_allow_html=True)

            if prediction[0]==1:
                st.error("Patient is likely to have Stroke")
            else:
                st.success("No Stroke Detected")

            st.subheader("Patient Summary")

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

elif menu == "Heart Disease Prediction":

    st.header("Heart Disease Risk Prediction")

    with st.form("heart_form"):

        col1,col2,col3 = st.columns(3)

        with col1:
            age = st.number_input("Age")
            sex = st.selectbox("Sex",[None,0,1])
            cp = st.selectbox("Chest Pain Type",[None,0,1,2,3])
            trestbps = st.number_input("Resting Blood Pressure")

        with col2:
            chol = st.number_input("Cholesterol")
            fbs = st.selectbox("Fasting Blood Sugar",[None,0,1])
            restecg = st.selectbox("Rest ECG",[None,0,1,2])
            thalach = st.number_input("Max Heart Rate")

        with col3:
            exang = st.selectbox("Exercise Induced Angina",[None,0,1])
            oldpeak = st.number_input("Old Peak")
            slope = st.selectbox("Slope",[None,0,1,2])
            ca = st.selectbox("Major Vessels",[None,0,1,2,3])
            thal = st.selectbox("Thal",[None,0,1,2,3])

        submit = st.form_submit_button("Predict Heart Disease")

    if submit:

        if None in [sex,cp,fbs,restecg,exang,slope,ca,thal] or chol==0 or trestbps==0:

            st.warning("Please fill all patient information.")

        else:

            data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                              thalach,exang,oldpeak,slope,ca,thal]])

            prediction = heart_model.predict(data)
            probability = heart_model.predict_proba(data)[0][1]*100

            level,style = risk_level(probability)

            st.subheader("Diagnosis Result")

            st.progress(int(probability))

            st.markdown(f"""
            <div class="card">
            <h3>Heart Disease Risk</h3>
            <h1>{probability:.2f}%</h1>
            <p class="{style}">{level}</p>
            </div>
            """,unsafe_allow_html=True)

            if prediction[0]==1:
                st.error("Patient likely has Heart Disease")
            else:
                st.success("No Heart Disease Detected")


# =====================================================
# FULL DIAGNOSIS
# =====================================================

else:

    st.header("Complete Health Diagnosis")

    with st.form("full_form"):

        age = st.number_input("Age")
        hypertension = st.selectbox("Hypertension",[None,0,1])
        glucose = st.number_input("Average Glucose Level")
        bmi = st.number_input("BMI")

        submit = st.form_submit_button("Run Full Diagnosis")

    if submit:

        if None in [hypertension] or glucose==0 or bmi==0:

            st.warning("Please complete patient details.")

        else:

            stroke_data = np.array([[age,hypertension,0,glucose,bmi]])

            stroke_prob = stroke_model.predict_proba(stroke_data)[0][1]*100
            stroke_pred = stroke_model.predict(stroke_data)

            heart_data = np.array([[age,1,1,120,200,0,1,150,0,1,1,0,2]])

            heart_prob = heart_model.predict_proba(heart_data)[0][1]*100
            heart_pred = heart_model.predict(heart_data)

            st.subheader("Final Diagnosis")

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


# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
"""
<center>

AI Medical Diagnosis System  
Machine Learning Project  

Built with Streamlit

</center>
""",
unsafe_allow_html=True
)
