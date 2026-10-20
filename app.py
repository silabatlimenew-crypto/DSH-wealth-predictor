import streamlit as st
import pandas as pd
import joblib

# Load the saved model and feature columns
model = joblib.load('best_wealth_model.pkl')
selected_features = joblib.load('selected_features.pkl')

st.title("DHS Household Wealth Index Predictor")
st.write("Predict the wealth tier (**Poor**, **Middle**, or **Rich**) of a household based on assets, housing materials, and geographic subbasin location.")

# --- Sidebar Inputs ---
st.sidebar.header("Household & Location Inputs")

# Geographic & Location Features
selected_subbasin = st.sidebar.selectbox("Subbasin", ['Abbay Basin Subbasin 1', 'Abbay Basin Subbasin 2', 'Abbay Basin Subbasin 3'])
urban_rural = st.sidebar.selectbox("Urban / Rural", ['Urban', 'Rural'])

# Household Assets & Characteristics
electricity = st.sidebar.selectbox("Electricity Access", ['Yes', 'No'])
television = st.sidebar.selectbox("Television", ['Yes', 'No'])
drinking_water = st.sidebar.selectbox("Drinking Water Source", ['Piped water', 'Tube well / borehole', 'Protected well', 'Surface water'])
toilet_facility = st.sidebar.selectbox("Toilet Facility", ['Flush toilet', 'Pit latrine', 'No facility'])

# --- Prediction Logic ---
if st.button("Predict Wealth Category"):
    # Create an input dataframe initialized to zero with exact training columns
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # Make prediction using the trained model
    prediction = model.predict(input_data)
    
    # Display results
    st.success(f"Predicted Household Wealth Category: **{prediction[0]}**")
    st.info(f"Contextual Location: **{selected_subbasin}** ({urban_rural})")
