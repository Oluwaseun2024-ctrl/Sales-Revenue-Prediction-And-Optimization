import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODEL & FEATURES
# ==============================
model = joblib.load('random_forest_model.pkl')
model_features = joblib.load('model_features.pkl')


# ==============================
# PREPROCESS FUNCTION (FROM YOUR NOTEBOOK)
# ==============================
def preprocess_input(df):
    # One-hot encoding (same as training)
    df = pd.get_dummies(
        df,
        columns=['product_category', 'store_location', 'season'],
        drop_first=True
    )

    df = pd.get_dummies(df, columns=['store_id'], drop_first=True)

    # Drop date if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Align with training features
    df = df.reindex(columns=model_features, fill_value=0)

    return df


# ==============================
# STREAMLIT UI
# ==============================
st.title("📊 Sales Revenue Prediction App")

st.write("Enter product and sales details to predict revenue")

# --- INPUTS ---
units_sold = st.number_input("Units Sold", min_value=0, value=100)
price_per_unit = st.number_input("Price per Unit ($)", min_value=0.0, value=10.0)
discount = st.slider("Discount (%)", 0, 100, 0)
holiday_flag = st.selectbox("Holiday?", [0, 1])
month = st.selectbox("Month", list(range(1, 13)))

product_category = st.selectbox(
    "Product Category",
    ["Electronics", "Clothing", "Groceries"]
)

store_location = st.selectbox(
    "Store Location",
    ["Lagos", "Abuja", "Port Harcourt"]
)

season = st.selectbox(
    "Season",
    ["Dry", "Rainy", "Harmattan"]
)

store_id = st.selectbox(
    "Store ID",
    list(range(1, 11))
)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Revenue"):
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'units_sold': units_sold,
        'price_per_unit': price_per_unit,
        'discount': discount,
        'holiday_flag': holiday_flag,
        'month': month,
        'product_category': product_category,
        'store_location': store_location,
        'season': season,
        'store_id': store_id
    }])

    # Preprocess
    processed_input = preprocess_input(input_df)

    # Predict
    prediction = model.predict(processed_input)

    # Display
    st.success(f"💰 Predicted Revenue: ${prediction[0]:,.2f}")