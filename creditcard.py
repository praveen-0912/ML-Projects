import streamlit as st
import numpy as np
import pandas as pd

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# For imbalance handling
from imblearn.over_sampling import SMOTE

# Title
st.title("💳 Credit Card Fraud Detection")

# ---- DATA GENERATION (IMBALANCED) ----
np.random.seed(42)
n = 5000

# Features
amount = np.round(np.random.uniform(1, 5000, n), 2)
transactions_per_day = np.random.randint(1, 50, n)
location_change = np.random.randint(0, 2, n)  # 0 = same, 1 = different location

# Imbalanced target (Fraud = very rare)
fraud = ((amount > 3000) & (transactions_per_day > 30) | (location_change == 1)).astype(int)

# Make it highly imbalanced
fraud[:4500] = 0   # 90% non-fraud
fraud[4500:] = 1   # 10% fraud

df = pd.DataFrame({
    "amount": amount,
    "transactions_per_day": transactions_per_day,
    "location_change": location_change,
    "fraud": fraud
})

# Split
X = df.drop("fraud", axis=1)
y = df["fraud"]

# ---- HANDLE IMBALANCE (SMOTE) ----
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ---- MODEL SELECTION ----
model_name = st.selectbox("Choose Model", [
    "Logistic Regression",
    "XGBoost"
])

# Initialize model
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {acc:.2f}")

# ---- USER INPUT ----
st.subheader("Enter Transaction Details")

amount_input = st.slider("Transaction Amount", 1, 5000, 100)
txn_input = st.slider("Transactions per Day", 1, 50, 5)
location_input = st.selectbox("Location Changed?", ["No", "Yes"])

location_val = 1 if location_input == "Yes" else 0

# Prediction
if st.button("Check Fraud"):

    user_data = np.array([[amount_input, txn_input, location_val]])
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe")
        st.balloons()