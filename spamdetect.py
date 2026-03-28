import streamlit as st
import numpy as np
import pandas as pd

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 Spam Email Classifier")

# ---- SAMPLE DATASET ----
data = {
    "email": [
        "Win a free lottery now",
        "Hello, how are you?",
        "Claim your prize money",
        "Meeting at 10 AM",
        "Get free coupons now",
        "Let's catch up tomorrow",
        "You won cash reward",
        "Project deadline reminder"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# ---- TEXT VECTORIZATION ----
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---- MODEL SELECTION ----
model_name = st.selectbox("Choose Model", [
    "Logistic Regression",
    "Naive Bayes"
])

# Initialize model
if model_name == "Logistic Regression":
    model = LogisticRegression()
else:
    model = MultinomialNB()

# Train
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {acc:.2f}")

# ---- USER INPUT ----
st.subheader("Enter Email Text")

user_input = st.text_area("Type your email here")

# Prediction
if st.button("Predict Spam"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM Email")
        else:
            st.success("✅ This is NOT Spam")
            st.balloons()