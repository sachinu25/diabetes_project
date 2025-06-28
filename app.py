import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.slider("Glucose", 0, 200)
bp = st.slider("Blood Pressure", 0, 150)
skin = st.slider("Skin Thickness", 0, 100)
insulin = st.slider("Insulin", 0, 900)
bmi = st.slider("BMI", 0.0, 70.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5)
age = st.slider("Age", 15, 100)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Likely Diabetic (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Not Diabetic (Probability: {1 - prob:.2f})")
