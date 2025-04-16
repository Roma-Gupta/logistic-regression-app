import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

model_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age (years)", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare (ticket price)", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])

# Prepare input dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'Fare': [fare],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Sex': [sex]
})

# Encode 'Sex'
input_data = pd.get_dummies(input_data)

# Ensure all model features are present
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Arrange columns
input_data = input_data[model_features]

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "🎉 Survived" if prediction == 1 else "❌ Did Not Survive"
    st.success(f"Prediction: {result}")
