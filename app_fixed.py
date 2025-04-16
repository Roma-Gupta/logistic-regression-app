import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

# Features used in model
model_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']

# App UI
st.title("Titanic Survival Predictor 🚢")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age (leave blank if unknown)", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare (leave blank if unknown)", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])

# Prepare input data
input_dict = {
    'Pclass': [pclass],
    'Age': [age if age > 0 else np.nan],
    'Fare': [fare if fare > 0 else np.nan],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Sex': [sex]
}

input_df = pd.DataFrame(input_dict)

# Handle missing values: Fill with mean or default
input_df['Age'].fillna(30, inplace=True)    # average age
input_df['Fare'].fillna(32, inplace=True)   # average fare

# Encode categorical
input_df = pd.get_dummies(input_df)

# Add missing model features (in case Sex_female or male is missing)
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns
input_df = input_df[model_features]

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    result = "🎉 Survived" if prediction == 1 else "❌ Did Not Survive"
    st.success(f"Prediction: {result}")
