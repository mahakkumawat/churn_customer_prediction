import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('Data.csv')

# 2. Preprocessing
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Handle potential empty strings in TotalCharges
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

# Convert text to numbers
X_encoded = pd.get_dummies(X)

# SAVE COLUMNS - This is what was missing in your error
model_columns = list(X_encoded.columns)
with open("columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 4. Train & Save
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Success! model.pkl, scaler.pkl, and columns.pkl generated.")