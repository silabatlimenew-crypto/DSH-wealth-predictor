import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="DHS Wealth Predictor", page_icon="🇪🇹", layout="wide")

# Load model and features safely
@st.cache_resource
def load_assets():
    model = joblib.load('best_wealth_model.pkl')
    # Retrieve exact feature names from model if available, else load from pkl
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        features = list(joblib.load('selected_features.pkl'))
    return model, features

model, expected_features = load_assets()

st.title("🇪🇹 Household Wealth Index Predictor")
st.write("Select household characteristics to predict the wealth tier (**Poor**, **Middle**, or **Rich**).")

# Extract Subbasin location options from features dynamically
subbasin_cols = [f for f in expected_features if f.startswith('Subbasin_')]
subbasin_names = [f.replace('Subbasin_', '') for f in subbasin_cols]

# Main Page Form Layout (Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Geographic Location")
    selected_subbasin = None
    if subbasin_names:
        selected_subbasin = st.selectbox("Subbasin Location", subbasin_names)
    
    st.subheader("🏠 Residency Type")
    residency = st.selectbox("Residency Type", ["Urban", "Rural"])

    st.subheader("🚰 Water & Sanitation")
    water_source = st.selectbox(
        "Drinking Water Source",
        ["Piped water", "Tube well / borehole", "Protected well", "Surface water", "Other"]
    )
    toilet_type = st.selectbox(
        "Toilet Facility Type",
        ["Flush toilet", "Pit latrine", "No facility / Bush", "Other"]
    )

with col2:
    st.subheader("💡 Household Assets")
    electricity = st.selectbox("Electricity Access (hv206)", ["No", "Yes"])
    radio = st.selectbox("Radio Ownership (hv207)", ["No", "Yes"])
    tv = st.selectbox("Television Ownership (hv208)", ["No", "Yes"])
    refrigerator = st.selectbox("Refrigerator Ownership (hv209)", ["No", "Yes"])

st.markdown("---")

# Predict Action Button
if st.button("🚀 Predict Household Wealth Category", use_container_width=True):
    # Initialize a 1-row DataFrame filled with 0s using EXACT model feature names and order
    input_data = pd.DataFrame(0, index=[0], columns=expected_features)

    # Map selected Subbasin
    if selected_subbasin:
        sub_col = f"Subbasin_{selected_subbasin}"
        if sub_col in input_data.columns:
            input_data.at[0, sub_col] = 1

    # Map Residency Type
    if 'hv025' in input_data.columns:
        input_data.at[0, 'hv025'] = 1 if residency == "Urban" else 0

    # Map Household Assets
    binary_map = {
        'hv206': 1 if electricity == "Yes" else 0,
        'hv207': 1 if radio == "Yes" else 0,
        'hv208': 1 if tv == "Yes" else 0,
        'hv209': 1 if refrigerator == "Yes" else 0,
    }
    for col_name, val in binary_map.items():
        if col_name in input_data.columns:
            input_data.at[0, col_name] = val

    # Map Water and Toilet Facilities
    for col in input_data.columns:
        if water_source.split()[0].lower() in col.lower():
            input_data.at[0, col] = 1
        if toilet_type.split()[0].lower() in col.lower():
            input_data.at[0, col] = 1

    # Execute Model Prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"### 📊 Prediction Result: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
