import streamlit as st
import numpy as np
import pickle
import pandas as pd
import datetime


# Load the model
with open('Car_Regressor_New.pkl', 'rb') as f:
    model = pickle.load(f)

st.write("Loaded model type:", type(model))

# Encoded label mappings
brand_to_models = {
    'Audi': ['Q5', 'A3', 'A4'],
    'BMW': ['5 Series', '3 Series', 'X5'],
    'Chevrolet': ['Malibu', 'Equinox', 'Impala'],
    'Ford': ['Explorer', 'Fiesta', 'Focus'],
    'Honda': ['Civic', 'CR-V', 'Accord'],
    'Hyundai': ['Elantra', 'Tucson', 'Sonata'],
    'Kia': ['Rio', 'Sportage', 'Optima'],
    'Mercedes': ['GLA', 'E-Class', 'C-Class'],
    'Toyota': ['Camry', 'RAV4', 'Corolla'],
    'Volkswagen': ['Golf', 'Tiguan', 'Passat']
}

brand_map = {
    'Audi': 0, 'BMW': 1, 'Chevrolet': 2, 'Ford': 3, 'Honda': 4,
    'Hyundai': 5, 'Kia': 6, 'Mercedes': 7, 'Toyota': 8, 'Volkswagen': 9
}

model_map = {
    'Q5': 22, 'A3': 2, 'A4': 3, '5 Series': 1, '3 Series': 0, 'X5': 29,
    'Malibu': 19, 'Equinox': 12, 'Impala': 18, 'Explorer': 13, 'Fiesta': 14, 'Focus': 15,
    'Civic': 8, 'CR-V': 6, 'Accord': 4, 'Elantra': 11, 'Tucson': 28, 'Sonata': 25,
    'Rio': 24, 'Sportage': 26, 'Optima': 20, 'GLA': 16, 'E-Class': 10, 'C-Class': 5,
    'Camry': 7, 'RAV4': 23, 'Corolla': 9, 'Golf': 17, 'Tiguan': 27, 'Passat': 21
}

fuel_map = {
    'Diesel': 0, 'Electric': 1, 'Hybrid': 2, 'Petrol': 3, 'Petrol (\\n)': 4
}

transmission_map = {
    'Automatic': 0, 'Manual': 1, 'Semi-Automatic': 2
}

# Streamlit UI
st.title("🚗 Used Car Price Predictor")

# Brand selection
brand = st.selectbox("Select Brand", list(brand_to_models.keys()))
model = st.selectbox("Select Model", brand_to_models[brand])
fuel = st.selectbox("Select Fuel Type", list(fuel_map.keys()))
transmission = st.selectbox("Select Transmission", list(transmission_map.keys()))
mileage = st.number_input("Mileage (in km/l)", value=15.0)
engine = st.number_input("Engine Capacity (in CC)", value=1500)
power = st.number_input("Power (in bhp)", value=100.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
kms = st.number_input("Kilometers Driven", value=50000)

# Predict button
if st.button("Predict Price"):
    try:
        input_data = np.array([
            brand_map[brand],
            model_map[model],
            fuel_map[fuel],
            transmission_map[transmission],
            mileage,
            engine,
            power,
            seats,
            year,
            kms
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Car Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

