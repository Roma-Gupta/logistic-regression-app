import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

st.title("Titanic Survival Prediction - Logistic Regression")

# Load full pipeline model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload Titanic CSV (like test set)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    try:
        # Use original raw features used in training
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']
        input_data = data[features].copy()

        # Impute missing values (only)
        imputer = SimpleImputer(strategy='most_frequent')
        input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=features)

        # ✅ Do NOT convert to NumPy if it's a pipeline with encoders
        predictions = model.predict(input_data)

        data['Prediction'] = predictions
        st.subheader("Prediction Output")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
