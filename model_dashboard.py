import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def run():
    df = pd.read_pickle("model_training_data.pkl")
    df_encoded = pd.get_dummies(df, columns=['Category', 'Borough'], drop_first=False)

    X = df_encoded.drop(['Product_ID', 'Theft_Reported'], axis=1)
    y = df_encoded['Theft_Reported']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.title("📊 Model Evaluation Dashboard")

    with st.spinner("Running GridSearchCV..."):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # confusion matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Low Risk", "High Risk"], 
                yticklabels=["Low Risk", "High Risk"], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # gridsearch results
    st.subheader("🎯 Best GridSearchCV Parameters")
    st.json(grid_search.best_params_)
    st.write(f"🔍 Best Cross-Validation Accuracy: **{grid_search.best_score_:.4f}**")

    st.subheader("📋 All Grid Search Results")
    results_df = pd.DataFrame(grid_search.cv_results_)
    st.dataframe(results_df[['params', 'mean_test_score', 'rank_test_score']])