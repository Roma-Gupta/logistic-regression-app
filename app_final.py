import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

st.title("Titanic Survival Prediction - Logistic Regression")

# Load the full pipeline (model + preprocessing)
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload Titanic CSV (like test set)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    try:
        # Keep only the raw columns used before training
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']
        input_data = data[features].copy()

        # Impute missing values (if needed)
        imputer = SimpleImputer(strategy='most_frequent')
        input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=features)

        # Convert to numpy array to remove column names (this is the KEY fix!)
        input_array = input_data_imputed.to_numpy()

        # Predict using the pipeline
        predictions = model.predict(input_array)

        data['Prediction'] = predictions
        st.subheader("Prediction Output")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
