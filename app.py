import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("rf_model.pkl", "rb"))

st.title("🛍️ Theft Risk Classifier")

price = st.slider("Price ($)", 0, 300, 50)
popularity = st.slider("Popularity (1–10)", 1, 10, 5)
security = st.selectbox("Security Level", [1, 2, 3])
category = st.selectbox("Category", ['Electronics', 'Clothing', 'Cosmetics', 'Furniture', 'Toys', 'Groceries'])

user_input = pd.DataFrame([{
    'Price': price,
    'Popularity': popularity,
    'Security_Level': security,
    'Category': category
}])

user_input_encoded = pd.get_dummies(user_input, columns=['Category'])

model_features = model.feature_names_in_  

for col in model_features:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0  

user_input_encoded = user_input_encoded[model_features] 

if st.button("🔍 Predict Theft Risk"):
    risk_prob = model.predict_proba(user_input_encoded)[0][1]
    prediction = model.predict(user_input_encoded)[0]

    st.subheader("🧾 Prediction Result:")
    st.write(f"Theft Risk Score: **{risk_prob * 100:.1f}%**")

    if prediction == 1:
        st.error("⚠️ High Risk of Theft")
    else:
        st.success("✅ Low Risk of Theft")