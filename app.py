
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.title("Iris Flower Species Prediction App")

# Load model & feature names
with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)["feature_columns"]

model = joblib.load("iris_model.pkl")

st.header("Enter Feature Values")

inputs = []
for col in feature_cols:
    val = st.number_input(col, min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    inputs.append(val)

if st.button("Predict"):
    arr = np.array(inputs).reshape(1, -1)
    pred = model.predict(arr)[0]
    species = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}
    st.success(f"Predicted Species: {species[pred]}")
