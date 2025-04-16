import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

st.title("Titanic Survival Prediction - Logistic Regression")

# Load the model pipeline (which includes preprocessor)
with open("logistic_model_titanic_v2.pkl", "rb") as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload Titanic CSV (like test set)", type=["csv"])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(raw_data)

    try:
        # Use original columns before training
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']
        df = raw_data[features].copy()

        # Impute missing values (only if pipeline doesn’t already)
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=features)

        # Convert to NumPy array to avoid feature name mismatch
        X_input = df_imputed.to_numpy()

        # Predict
        predictions = model.predict(X_input)
        raw_data['Prediction'] = predictions

        st.subheader("Prediction Output")
        st.dataframe(raw_data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
