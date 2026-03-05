if st.button("Predict Both"):

    try:
        # HEART PREDICTION
        heart_prediction = heart_model.predict(heart_input)[0]
        heart_prob = heart_model.predict_proba(heart_input)[0][1] * 100

        if heart_prediction == 1:
            heart_result = f"High Risk of Heart Disease ({heart_prob:.2f}%)"
        else:
            heart_result = f"Low Risk of Heart Disease ({100 - heart_prob:.2f}%)"

        # STROKE PREDICTION
        stroke_prediction = stroke_model.predict(stroke_input)[0]
        stroke_prob = stroke_model.predict_proba(stroke_input)[0][1] * 100

        if stroke_prediction == 1:
            stroke_result = f"Stroke Risk Detected ({stroke_prob:.2f}%)"
        else:
            stroke_result = f"No Stroke Risk ({100 - stroke_prob:.2f}%)"

        st.success("Prediction Results")

        st.write("Heart Disease Result:")
        st.write(heart_result)

        st.write("Stroke Result:")
        st.write(stroke_result)

    except Exception as e:
        st.error("Prediction failed. Please check inputs.")
