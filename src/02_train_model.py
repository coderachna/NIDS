import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the cleaned dataset (fast — no reprocessing needed)
DATA_PATH = r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\data\cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded shape:", df.shape)
print(df['Label'].value_counts())
# Separate features (X) from target label (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Encode text labels into numbers: Normal=0, DoS=1, etc. (exact mapping printed below)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nLabel encoding mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"  {class_name} -> {i}")

print("\nFeature matrix shape:", X.shape)
print("Encoded labels shape:", y_encoded.shape)

# Train/test split — stratified so class proportions are preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Confirm stratification worked — class proportions should closely match the full dataset
import numpy as np
print("\nTrain class distribution:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print("\nTest class distribution:")
print(pd.Series(y_test).value_counts(normalize=True).sort_index())

# Feature scaling — fit scaler on training data only, then apply to both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling complete.")
print("X_train_scaled shape:", X_train_scaled.shape)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

# ---- Model 1: Decision Tree (baseline) ----
print("\n" + "="*50)
print("Training Decision Tree...")
start = time.time()

dt_model = DecisionTreeClassifier(random_state=42, max_depth=15)
dt_model.fit(X_train_scaled, y_train)

print(f"Decision Tree trained in {time.time() - start:.2f} seconds")

dt_preds = dt_model.predict(X_test_scaled)

print("\nDecision Tree - Classification Report:")
print(classification_report(y_test, dt_preds, target_names=le.classes_))

print("\nDecision Tree - Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))

# ---- Model 2: Random Forest ----
print("\n" + "="*50)
print("Training Random Forest...")
start = time.time()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

print(f"Random Forest trained in {time.time() - start:.2f} seconds")

rf_preds = rf_model.predict(X_test_scaled)

print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, rf_preds, target_names=le.classes_))

print("\nRandom Forest - Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

import joblib

# Save the trained Random Forest model and the scaler + label encoder together
# (we need all three later to make predictions on new data in the Flask app)
joblib.dump(rf_model, r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\rf_model.pkl")
joblib.dump(scaler, r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\scaler.pkl")
joblib.dump(le, r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\label_encoder.pkl")
print("\nModel, scaler, and label encoder saved to disk.")

# Feature importances — which of the 78 features drove the Random Forest's decisions
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nTop 10 most important features:")
print(importances.sort_values(ascending=False).head(10))