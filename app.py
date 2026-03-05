import streamlit as st
import numpy as np
import joblib

# Load models
heart_model = joblib.load("heart_model.pkl")
stroke_model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="Medical Risk Prediction", layout="wide")

st.title("🩺 Heart Disease & Stroke Prediction System")

st.write("Enter patient information to predict heart disease, stroke, or both.")

# Layout (Left = Inputs, Right = Results)
col1, col2 = st.columns([1,1])

with col1:

    st.header("Patient Details")

    age = st.number_input("Age", 1, 120)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

    cholesterol = st.number_input("Cholesterol Level")

    blood_pressure = st.number_input("Blood Pressure")

    glucose = st.number_input("Glucose Level")

    st.write("")

    heart_btn = st.button("Predict Heart Disease ❤️")
    stroke_btn = st.button("Predict Stroke 🧠")
    both_btn = st.button("Predict Both")

with col2:

    st.header("Prediction Result")

    try:

        input_data = np.array([[age, sex, cholesterol, blood_pressure, glucose]])

        # HEART PREDICTION
        if heart_btn:

            pred = heart_model.predict(input_data)[0]
            prob = heart_model.predict_proba(input_data)[0][1]

            st.subheader("Heart Disease Result")

            if pred == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")

            st.progress(int(prob * 100))
            st.write(f"Risk Probability: **{prob*100:.2f}%**")


        # STROKE PREDICTION
        if stroke_btn:

            pred = stroke_model.predict(input_data)[0]
            prob = stroke_model.predict_proba(input_data)[0][1]

            st.subheader("Stroke Result")

            if pred == 1:
                st.error("High Risk of Stroke")
            else:
                st.success("Low Risk of Stroke")

            st.progress(int(prob * 100))
            st.write(f"Risk Probability: **{prob*100:.2f}%**")


        # BOTH PREDICTION
        if both_btn:

            st.subheader("Combined Prediction")

            heart_pred = heart_model.predict(input_data)[0]
            heart_prob = heart_model.predict_proba(input_data)[0][1]

            stroke_pred = stroke_model.predict(input_data)[0]
            stroke_prob = stroke_model.predict_proba(input_data)[0][1]

            st.write("### Heart Disease")

            if heart_pred == 1:
                st.error("High Risk")
            else:
                st.success("Low Risk")

            st.progress(int(heart_prob * 100))
            st.write(f"Probability: **{heart_prob*100:.2f}%**")

            st.write("---")

            st.write("### Stroke")

            if stroke_pred == 1:
                st.error("High Risk")
            else:
                st.success("Low Risk")

            st.progress(int(stroke_prob * 100))
            st.write(f"Probability: **{stroke_prob*100:.2f}%**")

    except:
        st.warning("Input format does not match the models. Please check the data.")
