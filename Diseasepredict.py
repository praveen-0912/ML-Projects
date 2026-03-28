# import streamlit as st
# import numpy as np
# import pandas as pd

# # ML Models
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# # Title
# st.title("🏥 Disease Prediction System")

# # ---- DATA GENERATION ----
# np.random.seed(42)
# n = 1000

# # Symptoms (0 = No, 1 = Yes)
# fever = np.random.randint(0, 2, n)
# cough = np.random.randint(0, 2, n)
# fatigue = np.random.randint(0, 2, n)
# headache = np.random.randint(0, 2, n)

# # Rule-based disease logic
# # 0 = No Disease, 1 = Flu, 2 = Viral Fever
# disease = []

# for i in range(n):
#     if fever[i] == 1 and cough[i] == 1:
#         disease.append(1)  # Flu
#     elif fever[i] == 1 and fatigue[i] == 1:
#         disease.append(2)  # Viral Fever
#     else:
#         disease.append(0)  # No Disease

# df = pd.DataFrame({
#     "fever": fever,
#     "cough": cough,
#     "fatigue": fatigue,
#     "headache": headache,
#     "disease": disease
# })

# # Split
# X = df.drop("disease", axis=1)
# y = df["disease"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # ---- MODEL SELECTION ----
# model_name = st.selectbox("Choose Model", [
#     "Logistic Regression",
#     "Decision Tree",
#     "Random Forest",
#     "SVM",
#     "KNN",
#     "Naive Bayes",
#     "Gradient Boosting"
# ])

# # Initialize model
# if model_name == "Logistic Regression":
#     model = LogisticRegression()
# elif model_name == "Decision Tree":
#     model = DecisionTreeClassifier()
# elif model_name == "Random Forest":
#     model = RandomForestClassifier(n_estimators=100)
# elif model_name == "SVM":
#     model = SVC()
# elif model_name == "KNN":
#     model = KNeighborsClassifier()
# elif model_name == "Naive Bayes":
#     model = GaussianNB()
# elif model_name == "Gradient Boosting":
#     model = GradientBoostingClassifier()

# # Train
# model.fit(X_train, y_train)

# # Accuracy
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)

# st.write(f"### Model Accuracy: {acc:.2f}")

# # ---- USER INPUT ----
# st.subheader("Enter Symptoms")

# fever_input = st.selectbox("Fever", [0, 1])
# cough_input = st.selectbox("Cough", [0, 1])
# fatigue_input = st.selectbox("Fatigue", [0, 1])
# headache_input = st.selectbox("Headache", [0, 1])

# # Prediction
# if st.button("Predict Disease"):
    
#     user_data = np.array([[fever_input, cough_input, fatigue_input, headache_input]])
#     prediction = model.predict(user_data)

#     if prediction[0] == 1:
#         st.success("🦠 Predicted Disease: Flu")
#     elif prediction[0] == 2:
#         st.warning("🤒 Predicted Disease: Viral Fever")
#     else:
#         st.info("✅ No Disease Detected")
#         st.balloons()


import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("💬 Disease Prediction Chatbot")

# ---- DATA ----
np.random.seed(42)
n = 1000

fever = np.random.randint(0, 2, n)
cough = np.random.randint(0, 2, n)
fatigue = np.random.randint(0, 2, n)
headache = np.random.randint(0, 2, n)

disease = []

for i in range(n):
    if fever[i] == 1 and cough[i] == 1:
        disease.append(1)
    elif fever[i] == 1 and fatigue[i] == 1:
        disease.append(2)
    else:
        disease.append(0)

df = pd.DataFrame({
    "fever": fever,
    "cough": cough,
    "fatigue": fatigue,
    "headache": headache,
    "disease": disease
})

X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---- CHAT INPUT ----
user_text = st.text_input("Describe your symptoms (e.g., I have fever and cough)")

# ---- NLP FUNCTION ----
def extract_symptoms(text):
    text = text.lower()

    symptoms = {
        "fever": 0,
        "cough": 0,
        "fatigue": 0,
        "headache": 0
    }

    for symptom in symptoms:
        if symptom in text:
            symptoms[symptom] = 1

    return list(symptoms.values())

# ---- PREDICTION ----
if st.button("Predict"):

    user_data = np.array([extract_symptoms(user_text)])
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.success("🦠 You may have Flu")
        st.balloons()
    elif prediction[0] == 2:
        st.warning("🤒 You may have Viral Fever")
    else:
        st.info("✅ You seem fine")