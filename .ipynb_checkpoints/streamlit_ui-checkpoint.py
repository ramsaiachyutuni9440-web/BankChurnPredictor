import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

model = keras.models.load_model("BankChurnPredictor.keras")
scaler = joblib.load("scaler.pkl")

st.title("Bank Customer Churn Predictor")

st.header("Enter Customer Details")

CreditScore = st.slider("Credit Score", 300, 850, 650)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age", 18, 92, 38)
Tenure = st.slider("Tenure (years)", 0, 10, 4)
Balance = st.number_input("Balance", 0.0, 250000.0, 75000.0, step=1000.0)
NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
HasCrCard = st.selectbox("Has Credit Card", ["Yes", "No"])
IsActiveMember = st.selectbox("Is Active Member", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 65000.0, step=1000.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": 1 if HasCrCard == "Yes" else 0,
        "IsActiveMember": 1 if IsActiveMember == "Yes" else 0,
        "EstimatedSalary": EstimatedSalary,
        "Geography_Germany": 1 if Geography == 'Germany' else 0,
        "Geography_Spain": 2 if Geography == 'Spain' else 0
    }])

    # Encode exactly as done during training
    input_df["Gender"] = input_df["Gender"].map({"Male": 0, "Female": 1})

    input_scaled = scaler.transform(input_df)

    prob = model.predict(input_scaled)[0][0] * 100

    st.write(prob)

    if prob >= 50:
        st.error(f"🚨 Customer likely to churn — Probability: {prob:.1f}%")
    else:
        st.success(f"✅ Customer likely to stay — Probability: {prob:.1f}%")