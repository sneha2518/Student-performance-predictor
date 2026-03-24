import streamlit as st
import numpy as np
import pickle
import os

# Load model safely
model = pickle.load(open(os.path.join(os.getcwd(), "model.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(os.getcwd(), "encoders.pkl"), "rb"))

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")

st.header("📥 Enter Student Details")

gender = st.selectbox("Gender", ["male", "female"])

race = st.selectbox("Student Category", ["group A", "group B", "group C", "group D", "group E"])

parent_edu = st.selectbox(
    "Parental Level of Education",
    ["bachelor's degree", "some college", "master's degree",
     "associate's degree", "high school", "some high school"]
)

lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

if st.button("🔮 Predict"):

    try:
        input_data = [
            encoders['gender'].transform([gender])[0],
            encoders['race/ethnicity'].transform([race])[0],
            encoders['parental level of education'].transform([parent_edu])[0],
            encoders['lunch'].transform([lunch])[0],
            encoders['test preparation course'].transform([test_prep])[0]
        ]

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]

        if prediction >= 70:
            result = "Good"
        elif prediction >= 40:
            result = "Average"
        else:
            result = "Poor"

        st.success(f"🎯 Performance: {result}")
        st.write(f"📊 Score: {round(prediction, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")