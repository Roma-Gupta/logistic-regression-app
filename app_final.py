
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
        # === Preprocessing: Use only trained features ===
        X = raw_data[['Pclass', 'Sex', 'Age', 'Fare']].copy()

        # Encode 'Sex'
        X['Sex'] = X['Sex'].map({'male': 1, 'female': 0})

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Strip column names if needed
        X_array = X_imputed.to_numpy()

        # Predict
        predictions = model.predict(X_array)

        raw_data['Prediction'] = predictions
        st.subheader("Prediction Output")
        st.dataframe(raw_data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
