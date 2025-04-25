import streamlit as st
import numpy as np
import pickle
import pandas as pd
import datetime


# Load the model
with open('Car_Regressor_New.pkl', 'rb') as f:
    model_X = pickle.load(f)

st.write("Loaded model type:", type(model_X))

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


## Brand selection
brand = st.selectbox("Select Brand", list(brand_to_models.keys()))
model_1 = st.selectbox("Select Model", brand_to_models[brand])
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
fuel = st.selectbox("Select Fuel Type", list(fuel_map.keys()))
transmission = st.selectbox("Select Transmission", list(transmission_map.keys()))
mileage = st.number_input("Mileage (km)", min_value=0, max_value=50000, value=5000)
doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
owner_count = st.selectbox("Previous Owners", [0, 1, 2, 3, 4])
year = st.slider("Year of Manufacture", 1990, datetime.datetime.now().year, 2015)


# Calculate car age
car_age = datetime.datetime.now().year - year

# Predict button
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([{
            "Brand": brand_map[brand],
            "Model": model_map[model_1],
            "Engine_Size": engine_size,
            "Fuel_Type": fuel_map[fuel],
            "Transmission": transmission_map[transmission],
            "Mileage": mileage,
            "Doors": doors,
            "Owner_Count": owner_count,
            "Car_Age": car_age
        }])

        prediction = model_X.predict(input_df)[0]
        st.success(f"Estimated Car Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

