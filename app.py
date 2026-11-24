import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Page settings
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction System")
st.write("Enter the property details to estimate its sale price.")

# Select housing market
source = st.selectbox(
    "Select Housing Market",
    ["Ames", "King County"]
)

# Property details
living_area = st.number_input(
    "Living Area (sq ft)", min_value=100, value=1500
)

lot_area = st.number_input(
    "Lot Area (sq ft)", min_value=100, value=5000
)

bedrooms = st.number_input(
    "Number of Bedrooms", min_value=0, max_value=20, value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms", min_value=0.0, max_value=20.0, value=2.0
)

basement_area = st.number_input(
    "Basement Area (sq ft)", min_value=0, value=500
)

# Keep the sale year within the dataset's historical period
if source == "Ames":
    year_built = st.number_input(
        "Year Built", min_value=1800, max_value=2010, value=2000
    )
    sale_year = st.number_input(
        "Sale Year", min_value=2006, max_value=2010, value=2008
    )
else:
    year_built = st.number_input(
        "Year Built", min_value=1800, max_value=2015, value=2000
    )
    sale_year = st.number_input(
        "Sale Year", min_value=2014, max_value=2015, value=2014
    )

is_renovated = st.selectbox(
    "Has the House Been Renovated?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Calculate house age
house_age = sale_year - year_built

# Prediction button
if st.button("Predict House Price"):

    if year_built > sale_year:
        st.error("Year built cannot be later than the sale year.")
    else:
        input_data = pd.DataFrame([{
            "living_area": living_area,
            "lot_area": lot_area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "basement_area": basement_area,
            "year_built": year_built,
            "house_age": house_age,
            "is_renovated": is_renovated,
            "sale_year": sale_year,
            "dataset_source": source
        }])

        prediction = model.predict(input_data)[0]

        st.success(
            f"Estimated House Price: ${prediction:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate based on historical "
            "housing data, not a guaranteed market price."
        )