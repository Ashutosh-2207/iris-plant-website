import joblib
import numpy as np

# Load the model
model = joblib.load('iris_random_forest_model.joblib')

# Example new sample (sepal_length, sepal_width, petal_length, petal_width)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict species
prediction = model.predict(sample)
print(f"Predicted species: {prediction[0]}")
