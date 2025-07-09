import streamlit as st

st.set_page_config(page_title="Theft Risk App", layout="centered")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a page:", ["🧠 Theft Classifier", "📊 Model Evaluation"])

if page == "🧠 Theft Classifier":
    import classifier_ui
    classifier_ui.run()

elif page == "📊 Model Evaluation":
    import model_dashboard
    model_dashboard.run()