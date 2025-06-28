import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVC(probability=True, kernel='rbf', C=1)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

y_pred = model.predict(X_test_scaled)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[:10])

st.subheader("SHAP Summary Plot (First 10 Test Samples)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test.iloc[:10], show=False)
st.pyplot(fig)
