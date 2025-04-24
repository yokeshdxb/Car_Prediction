import streamlit as st
import numpy as np
import pickle
import pandas as pd
import datetime


# Load the model
with open('Car_Regressor.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚗 Used Car Price Predictor")

# User Inputs
brand = st.text_input("Brand", "Toyota")
model_name = st.text_input("Model", "Corolla")
year = st.slider("Year of Manufacture", 1990, datetime.datetime.now().year, 2015)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
owner_count = st.selectbox("Previous Owners", [0, 1, 2, 3, 4])

# Calculate car age
car_age = datetime.datetime.now().year - year

# Prepare input DataFrame
input_dict = {
    "Brand": [brand],
    "Model": [model_name],
    "Engine_Size": [engine_size],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Mileage": [mileage],
    "Doors": [doors],
    "Owner_Count": [owner_count],
    "Car_Age": [car_age]
}

input_df = pd.DataFrame(input_dict)

# 🔄 Apply same encoding and preprocessing used during training
# Example placeholder:
# input_df = your_preprocessing_function(input_df)

# ⚠️ Replace above line with your actual preprocessing steps!

# Predict button
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


