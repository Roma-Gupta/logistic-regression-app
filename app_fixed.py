import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("logistic_model_titanic.pkl", "rb") as file:
    model = pickle.load(file)

# Expected features based on model training
model_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']

# App UI
st.title("Titanic Survival Predictor 🚢")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
age = st.slider("Age", 0, 100, 25)
fare = st.slider("Fare", 0, 500, 50)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])

# Convert input to DataFrame
input_dict = {
    'Pclass': [pclass],
    'Age': [age],
    'Fare': [fare],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Sex': [sex]
}
input_df = pd.DataFrame(input_dict)

# One-hot encoding for Sex
input_df = pd.get_dummies(input_df)

# Add any missing columns from model_features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[model_features]

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    result = "Survived ✅" if prediction == 1 else "Did not survive ❌"
    st.success(f"Prediction: {result}")
