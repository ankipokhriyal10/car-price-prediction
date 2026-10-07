import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
# Load trained model
model = pk.load(open('model.pkl', 'rb'))
st.header('Car Price Prediction ML Model')
# Load dataset
cars_data = pd.read_csv('Car details.csv')
# Function to get car brand
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)
# Encoding dictionaries
brand_dict = {
    'Maruti': 1,
    'Skoda': 2,
    'Honda': 3,
    'Hyundai': 4,
    'Toyota': 5,
    'Ford': 6,
    'Renault': 7,
    'Mahindra': 8,
    'Tata': 9,
    'Chevrolet': 10,
    'Datsun': 11,
    'Jeep': 12,
    'Mercedes-Benz': 13,
    'Mitsubishi': 14,
    'Audi': 15,
    'Volkswagen': 16,
    'BMW': 17,
    'Nissan': 18,
    'Lexus': 19,
    'Jaguar': 20,
    'Land': 21,
    'MG': 22,
    'Volvo': 23,
    'Daewoo': 24,
    'Kia': 25,
    'Fiat': 26,
    'Force': 27,
    'Ambassador': 28,
    'Ashok': 29,
    'Isuzu': 30,
    'Opel': 31
}
fuel_dict = {
    'Diesel': 1,
    'Petrol': 2,
    'LPG': 3,
    'CNG': 4
}
seller_dict = {
    'Individual': 1,
    'Dealer': 2,
    'Trustmark Dealer': 3
}
transmission_dict = {
    'Manual': 1,
    'Automatic': 2
}
owner_dict = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 5
}
#user input
name = st.selectbox(
    'Select Car Brand',
    cars_data['name'].unique()
)
year = st.slider(
    'Car Manufactured Year',
    1994,
    2024
)
km_driven = st.slider(
    'No of kms Driven',
    11,
    200000
)
fuel = st.selectbox(
    'Fuel type',
    cars_data['fuel'].unique()
)
seller_type = st.selectbox(
    'Seller type',
    cars_data['seller_type'].unique()
)
transmission = st.selectbox(
    'Transmission type',
    cars_data['transmission'].unique()
)
owner = st.selectbox(
    'Owner type',
    cars_data['owner'].unique()
)
mileage = st.slider(
    'Car Mileage',
    10.0,
    40.0
)
engine = st.slider(
    'Engine CC',
    700.0,
    5000.0
)
max_power = st.slider(
    'Max Power',
    0.0,
    200.0
)
seats = st.slider(
    'No of Seats',
    5.0,
    10.0
)
# Prediction
if st.button("Predict"):
# Convert categorical values into numbers
    name_encoded = brand_dict.get(name)
    fuel_encoded = fuel_dict.get(fuel)
    seller_encoded = seller_dict.get(seller_type)
    transmission_encoded = transmission_dict.get(transmission)
    owner_encoded = owner_dict.get(owner)
# Create input DataFrame
    input_data_model = pd.DataFrame(
        [[
            name_encoded,
            year,
            km_driven,
            fuel_encoded,
            seller_encoded,
            transmission_encoded,
            owner_encoded,
            mileage,
            engine,
            max_power,
            seats
        ]],
        columns=[
            'name',
            'year',
            'km_driven',
            'fuel',
            'seller_type',
            'transmission',
            'owner',
            'mileage',
            'engine',
            'max_power',
            'seats'
        ]
    )
 # Make prediction
    car_price = model.predict(input_data_model)

    st.success(
        'Car Price is going to be ₹' + str(round(car_price[0], 2))
    )
