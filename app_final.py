import streamlit as st
import pandas as pd
import pickle

st.title("🚢 Titanic Survival Prediction App")

# Load trained model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # ✅ Select only features used in training
        required_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        input_data = data[required_features]

        # ✅ Drop rows with missing values
        input_data = input_data.dropna()

        # Predict
        predictions = model.predict(input_data)

        # Merge predictions with original data (only non-NaN rows)
        data_clean = data.loc[input_data.index]
        data_clean["Survived Prediction"] = predictions

        st.subheader("Prediction Output")
        st.dataframe(data_clean)

        # Download option
        st.download_button("Download Results", data_clean.to_csv(index=False), "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
