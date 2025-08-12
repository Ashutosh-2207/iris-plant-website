import streamlit as st
import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Iris class names
iris_classes = ['Setosa', 'Versicolor', 'Virginica']

# App title
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the measurements and predict the Iris flower type.")

# User input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌼 Predicted Class: {iris_classes[prediction]}")
