import streamlit as st
import pandas as pd
import pickle

# Title
st.title("🚢 Titanic Survival Prediction App")

# Load the trained model
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # ✅ Load and prepare data
        data = pd.read_csv(uploaded_file)
        required_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        input_data = data[required_features]

        # ✅ Predict only on required features
        predictions = model.predict(input_data)

        # ✅ Add predictions back to original data
        data["Survived Prediction"] = predictions

        # Show prediction
        st.subheader("Prediction Output")
        st.dataframe(data)

        # Download predictions
        st.download_button("Download Results", data.to_csv(index=False), "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
