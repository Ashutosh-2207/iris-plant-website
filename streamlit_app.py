import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model_path = "iris_random_forest_model.joblib"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

st.title("🌸 Iris Species Predictor")

st.write("""
Enter the measurements of the iris flower below and click **Predict** to see the species.
""")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted species: **{prediction}**")

st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** and **scikit-learn**")
