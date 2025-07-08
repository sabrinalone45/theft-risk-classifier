import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))


np.random.seed(42)

n = 300
categories = ['Electronics', 'Clothing', 'Cosmetics', 'Furniture', 'Toys', 'Groceries']
security_levels = [1, 2, 3] 

data = {
    'Product_ID': [f'P{i:03}' for i in range(n)],
    'Category': np.random.choice(categories, n),
    'Price': np.round(np.random.exponential(scale=50, size=n), 2),
    'Popularity': np.round(np.random.uniform(1, 10, n), 2),
    'Security_Level': np.random.choice(security_levels, n),
}

df = pd.DataFrame(data)

df['Theft_Reported'] = (
    (df['Price'] > 50).astype(int) +
    (df['Popularity'] > 7).astype(int) +
    (df['Security_Level'] == 1).astype(int)
)


df['Theft_Reported'] = (df['Theft_Reported'] >= 2).astype(int)

df.head()

import seaborn as sns
import matplotlib.pyplot as plt

print(df.describe())

print(df.groupby("Category")["Theft_Reported"].mean())

sns.boxplot(x='Theft_Reported', y='Price', data=df)
plt.title("Price vs Theft")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=True)

X = df_encoded.drop(['Product_ID', 'Theft_Reported'], axis=1)
y = df_encoded['Theft_Reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# predictions
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

print("🔍 Sample of the Dataset:")
print(df.head())

from sklearn.metrics import accuracy_score

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")

plt.figure(figsize=(8,6)) 
sns.heatmap(df.corr(numeric_only=True).round(2), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix (Rounded)")
plt.show()


risk_scores = clf.predict_proba(X_test)[:, 1]  
results_df = X_test.copy()
results_df['Actual_Theft'] = y_test.values
results_df['Predicted_Theft'] = y_pred
results_df['Theft_Risk_Score'] = np.round(risk_scores * 100, 1)

print(results_df.head())
print("✅ Model successfully trained and evaluated!")

X_test_reset = X_test.reset_index(drop=True)
top_risks = results_df.copy()
top_risks['Product_ID'] = df_encoded.loc[X_test.index, 'Product_ID'].values
top_risks['Price'] = df_encoded.loc[X_test.index, 'Price'].values
top_risks['Security_Level'] = df_encoded.loc[X_test.index, 'Security_Level'].values

top10 = top_risks.sort_values(by='Theft_Risk_Score', ascending=False).head(10)
print("🔒 Top 10 High-Risk Products:")
print(top10[['Product_ID', 'Price', 'Security_Level', 'Theft_Risk_Score']])

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(
    x='Theft_Risk_Score', 
    y='Product_ID', 
    data=top10, 
    palette='Reds_r'
)
plt.title('Top 10 High-Risk Products by Theft Risk Score')
plt.xlabel('Theft Risk Score (%)')
plt.ylabel('Product ID')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
