# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Title
# st.title("🎯 Student Placement Prediction")

# # Generate Dataset
# np.random.seed(42)
# n = 1000

# cgpa = np.round(np.random.uniform(5.0, 10.0, n), 2)
# skills = np.random.randint(1, 11, n)        # skill rating (1–10)
# internships = np.random.randint(0, 5, n)

# # Placement logic (rule-based for training)
# placed = ((cgpa > 7.0) & (skills > 6) | (internships >= 2)).astype(int)

# df = pd.DataFrame({
#     "cgpa": cgpa,
#     "skills": skills,
#     "internships": internships,
#     "placed": placed
# })

# # Split data
# X = df.drop("placed", axis=1)
# y = df["placed"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Model selection
# model_option = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])

# if model_option == "Logistic Regression":
#     model = LogisticRegression()
# else:
#     model = DecisionTreeClassifier(max_depth=5)

# # Train model
# model.fit(X_train, y_train)

# # Accuracy
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)

# st.write(f"### Model Accuracy: {acc:.2f}")

# # ---- USER INPUT ----
# st.subheader("Enter Student Details")

# cgpa_input = st.slider("CGPA", 5.0, 10.0, 7.5)
# skills_input = st.slider("Skills Rating (1-10)", 1, 10, 5)
# internships_input = st.slider("Number of Internships", 0, 5, 1)

# # Prediction button
# if st.button("Predict Placement"):

#     user_data = np.array([[cgpa_input, skills_input, internships_input]])
#     prediction = model.predict(user_data)

#     if prediction[0] == 1:
#         st.success("🎉 Student is Likely to be Placed")
#         st.balloons()
#     else:
#         st.error("❌ Student is Not Likely to be Placed")



import streamlit as st
import numpy as np
import pandas as pd

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Title
st.title("🎓 Student Placement Prediction")

# ---- DATA GENERATION ----
np.random.seed(42)
n = 1000

cgpa = np.round(np.random.uniform(5.0, 10.0, n), 2)
skills = np.random.randint(1, 10, n)  # skill rating
internships = np.random.randint(0, 5, n)

# Rule-based target
placed = ((cgpa > 7.0) & (skills > 5) | (internships >= 2)).astype(int)

df = pd.DataFrame({
    "cgpa": cgpa,
    "skills": skills,
    "internships": internships,
    "placed": placed
})

# Split
X = df.drop("placed", axis=1)
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---- MODEL SELECTION ----
model_name = st.selectbox("Choose Model", [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN",
    "Naive Bayes",
    "Gradient Boosting"
])

# Initialize model
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)
elif model_name == "SVM":
    model = SVC()
elif model_name == "KNN":
    model = KNeighborsClassifier()
elif model_name == "Naive Bayes":
    model = GaussianNB()
elif model_name == "Gradient Boosting":
    model = GradientBoostingClassifier()

# Train
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {acc:.2f}")

# ---- USER INPUT ----
st.subheader("Enter Student Details")

cgpa_input = st.slider("CGPA", 5.0, 10.0, 7.0)
skills_input = st.slider("Skill Level (1-10)", 1, 10, 5)
internships_input = st.slider("Internships", 0, 5, 1)

# Prediction
if st.button("Predict Placement"):
    
    user_data = np.array([[cgpa_input, skills_input, internships_input]])
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.success(" Student is Likely to be Placed")
        st.balloons()

    else:
        st.error(" Student is Not Likely to be Placed")
    

    
   
    