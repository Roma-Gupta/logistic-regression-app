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
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    try:
        # Step 1: Required columns
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']
        df = data[features].copy()

        # Step 2: Encode 'Sex'
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        # Step 3: One-hot encode 'Embarked'
        df = pd.get_dummies(df, columns=['Embarked'])

        # Step 4: Ensure all dummy columns exist
        for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
            if col not in df.columns:
                df[col] = 0

        # Step 5: Set final column order
        final_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        df = df[final_cols]

        # Step 6: Impute missing values
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(df)  # ✅ Output is NumPy array — NO column names

        # Step 7: Predict
        predictions = model.predict(input_data)
        data['Prediction'] = predictions

        st.subheader("Prediction Output")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to begin.")
