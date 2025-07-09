import streamlit as st
import pandas as pd
import pickle

def run():
    st.title("🛍️ Theft Risk Classifier")

    model = pickle.load(open("rf_model.pkl", "rb"))

    # user inputs
    price = st.slider("Price ($)", 0, 300, 50)
    popularity = st.slider("Popularity (1–10)", 1, 10, 5)
    security = st.selectbox("Security Level", [1, 2, 3])
    category = st.selectbox("Category", ['Electronics', 'Clothing', 'Cosmetics', 'Furniture', 'Toys', 'Groceries'])
    borough = st.selectbox("Borough", ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND'])
    stock = st.slider("Stock Level", 1, 100, 50)
    discount = st.slider("Discount Percent (%)", 0, 50, 0, step=10)
    traffic = st.slider("Foot Traffic Score (0-10)", 0.0, 10.0, 5.0, 0.1)

    user_input = pd.DataFrame([{
        'Price': price,
        'Popularity': popularity,
        'Security_Level': security,
        'Category': category,
        'Borough': borough,
        'Stock_Level': stock,
        'Discount_Percent': discount,
        'Foot_Traffic_Score': traffic
    }])

    user_input_encoded = pd.get_dummies(user_input, columns=['Category', 'Borough'])
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

        st.markdown("### 💡 Risk Reasoning:")
        reasons = []
        if price > 50: reasons.append("• High Price")
        if popularity > 7: reasons.append("• Popular Product")
        if security == 1: reasons.append("• Low Security")
        if discount >= 30: reasons.append("• Deep Discount")
        if traffic > 7: reasons.append("• High Foot Traffic")
        if borough in ['MANHATTAN', 'BROOKLYN']: reasons.append("• High-Risk Borough")
        for r in reasons:
            st.markdown(r)