
import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

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
        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Predict using imputed data
        predictions = model.predict(data_imputed)
        data["Prediction"] = predictions

        st.subheader("Prediction Output")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin prediction.")
