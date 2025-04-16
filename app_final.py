import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

st.title("Titanic Survival Prediction - Logistic Regression")

# Load model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload Titanic CSV (like test set)", type=["csv"])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(raw_data)

    try:
        # Use exactly the features used during training
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']
        df = raw_data[features].copy()

        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')  # in case Embarked/Sex has missing
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=features)

        # Predict
        predictions = model.predict(df_imputed)
        raw_data['Prediction'] = predictions

        st.subheader("Prediction Output")
        st.dataframe(raw_data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
