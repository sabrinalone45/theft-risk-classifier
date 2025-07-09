import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

theft_df = pd.read_csv("data/nyc_retail_theft.csv")
borough_counts = theft_df['BORO_NM'].value_counts()
borough_risk_score = (borough_counts / borough_counts.max()).to_dict()

if "(null)" in borough_risk_score:
    del borough_risk_score["(null)"]

np.random.seed(42)
n = 300
categories = ['Electronics', 'Clothing', 'Cosmetics', 'Furniture', 'Toys', 'Groceries']
security_levels = [1, 2, 3]
boroughs = list(borough_risk_score.keys())

data = {
    'Product_ID': [f'P{i:03}' for i in range(n)],
    'Category': np.random.choice(categories, n),
    'Price': np.round(np.random.exponential(scale=50, size=n), 2),
    'Popularity': np.round(np.random.uniform(1, 10, n), 2),
    'Security_Level': np.random.choice(security_levels, n),
    'Borough': np.random.choice(boroughs, n),
    'Stock_Level': np.random.randint(1, 100, n),
    'Discount_Percent': np.random.choice([0, 10, 20, 30, 40, 50], n),
    'Foot_Traffic_Score': np.round(np.random.normal(loc=5, scale=2, size=n), 2)

}

df = pd.DataFrame(data)

df['Borough_Risk_Score'] = df['Borough'].map(borough_risk_score)

df['Theft_Reported'] = (
    (df['Price'] > 50).astype(int) +
    (df['Popularity'] > 7).astype(int) +
    (df['Security_Level'] == 1).astype(int) +
    (df['Borough_Risk_Score'] > 0.4).astype(int) +
    (df['Discount_Percent'] >= 30).astype(int) +
    (df['Foot_Traffic_Score'] > 7).astype(int)
)
df['Theft_Reported'] = (df['Theft_Reported'] >= 3).astype(int)

df_encoded = pd.get_dummies(df, columns=['Category', 'Borough'], drop_first=False)

X = df_encoded.drop(['Product_ID', 'Theft_Reported'], axis=1)
y = df_encoded['Theft_Reported']

clf = RandomForestClassifier()
clf.fit(X, y)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved with borough-based theft risk.")

df['Theft_Reported'] = y
df.to_pickle("model_training_data.pkl")