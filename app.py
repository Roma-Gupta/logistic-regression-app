import streamlit as st
import pandas as pd
import pickle

# Title
st.title("Titanic Survival Prediction - Logistic Regression")

# Load the model
with open("logistic_model_titanic.pkl", "rb") as file:
    model = pickle.load(file)

# Upload CSV or show sample
uploaded_file = st.file_uploader("Upload test CSV file (like Titanic_test.csv)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    try:
        predictions = model.predict(data)
        data["Prediction"] = predictions
        st.subheader("Prediction Output")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin prediction.")
