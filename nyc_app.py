import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="NYC Property Predictor", layout="wide")
st.title("🏙️ NYC Property Sale Price Predictor")

# Load model and feature names
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # returns (model, expected_features)

model, expected_features = load_model()

# User Input Form
st.header("📝 Enter Property Details")

with st.form("property_form"):
    borough = st.selectbox("Borough", ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"])
    building_class_category = st.selectbox("Building Class Category", ["01 ONE FAMILY DWELLINGS", "02 TWO FAMILY DWELLINGS"])
    tax_class_present = st.selectbox("Tax Class at Present", ["1", "2", "4"])
    tax_class_sale = st.selectbox("Tax Class at Time of Sale", ["1", "2", "4"])
    building_class_present = st.selectbox("Building Class at Present", ["A0", "A1", "A2"])
    building_class_sale = st.selectbox("Building Class at Time of Sale", ["A0", "A1", "A2"])
    season = st.selectbox("Season of Sale", ["Winter", "Spring", "Summer", "Fall"])

    gross_sqft = st.number_input("Gross Square Feet", min_value=100)
    land_sqft = st.number_input("Land Square Feet", min_value=100)
    residential_units = st.number_input("Residential Units", min_value=1, step=1)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
    sale_year = st.number_input("Sale Year", min_value=2000, max_value=2025, step=1)

    submit = st.form_submit_button("Predict Sale Price")

if submit:
    # Create feature vector
    input_dict = {
        "GROSS SQUARE FEET": gross_sqft,
        "LAND SQUARE FEET": land_sqft,
        "RESIDENTIAL UNITS": residential_units,
        "YEAR BUILT": year_built,
        "SALE YEAR": sale_year,
        "SALE MONTH": 1,  # not used here
        "BUILDING AGE": sale_year - year_built,
        "PRICE PER SQFT": 0  # not used for prediction, model predicts price
    }

    df_input = pd.DataFrame([input_dict])

    # One-hot encoding
    df_input["SEASON_" + season] = 1
    df_input["BOROUGH_" + borough] = 1
    df_input["BUILDING CLASS CATEGORY_" + building_class_category] = 1
    df_input["TAX CLASS AT PRESENT_" + tax_class_present] = 1
    df_input["TAX CLASS AT TIME OF SALE_" + tax_class_sale] = 1
    df_input["BUILDING CLASS AT PRESENT_" + building_class_present] = 1
    df_input["BUILDING CLASS AT TIME OF SALE_" + building_class_sale] = 1

    # Add missing expected columns
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns to match model
    df_input = df_input[expected_features]

    # Prediction
    y_pred = model.predict(df_input)[0]
    st.success(f"💰 Predicted Sale Price: ${y_pred:,.2f}")
