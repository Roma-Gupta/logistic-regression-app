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
        # === Preprocessing ===
        df = raw_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].copy()

        # Encode 'Sex'
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        # One-hot encode 'Embarked'
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

        # Handle missing dummy columns
        expected_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Ensure order of columns
        df = df[expected_cols]

        # Handle NaNs
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=expected_cols)

        # Predict
        predictions = model.predict(df_imputed)
        raw_data['Prediction'] = predictions

        st.subheader("Prediction Output")
        st.dataframe(raw_data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
