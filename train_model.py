import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


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

df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=False)

X = df_encoded.drop(['Product_ID', 'Theft_Reported'], axis=1)
y = df_encoded['Theft_Reported']

clf = RandomForestClassifier()
clf.fit(X, y)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved as rf_model.pkl")
