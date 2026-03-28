import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("ML Project App")

np.random.seed(42)
n = 1000

# 👉 Create features (CHANGE THIS)
f1 = np.random.randint(1, 100, n)
f2 = np.random.randint(1, 100, n)

# 👉 Target logic (CHANGE THIS)
target = ((f1 > 50) & (f2 > 50)).astype(int)

df = pd.DataFrame({
    "f1": f1,
    "f2": f2,
    "target": target
})

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 👉 Model (CHANGE THIS)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"Accuracy: {acc:.2f}")

# UI
val1 = st.slider("Feature 1", 1, 100, 50)
val2 = st.slider("Feature 2", 1, 100, 50)

if st.button("Predict"):
    pred = model.predict([[val1, val2]])
    st.write("Prediction:", pred[0])
    st.balloons()