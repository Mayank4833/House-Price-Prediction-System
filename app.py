import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# --------------------------------------------------
# LIGHT / DARK MODE
# --------------------------------------------------

with st.sidebar:
    theme_mode = st.radio(
        "🎨 Choose Appearance",
        ["☀️ Light Mode", "🌙 Dark Mode"],
        index=0,
        horizontal=True
    )

if theme_mode == "🌙 Dark Mode":
    bg_color = "#101827"
    text_color = "#F1F5F9"
    card_color = "#1E293B"
    subtitle_color = "#CBD5E1"
    border_color = "#334155"
    section_color = "#93C5FD"
    input_bg = "#1E293B"
else:
    bg_color = "#F4F7FB"
    text_color = "#142B49"
    card_color = "#FFFFFF"
    subtitle_color = "#718096"
    border_color = "#E3EAF3"
    section_color = "#17365D"
    input_bg = "#FFFFFF"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

[data-testid="stSidebar"] {{
    background-color: #10243e;
}}

[data-testid="stSidebar"] * {{
    color: white;
}}

[data-testid="stSidebar"] label {{
    color: white !important;
}}

/* Sidebar housing market dropdown */

[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border-color: #D1D5DB !important;
}}

[data-testid="stSidebar"] div[data-baseweb="select"] * {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}}

/* Dropdown menu options */

[data-baseweb="menu"] {{
    background-color: #FFFFFF !important;
}}

[data-baseweb="menu"] * {{
    color: #000000 !important;
}}

/* Main headings */

.main-title {{
    font-size: 34px;
    font-weight: 800;
    color: {text_color};
    margin-bottom: 5px;
}}

.subtitle {{
    color: {subtitle_color};
    font-size: 16px;
    margin-bottom: 25px;
}}

.section-title {{
    font-size: 21px;
    font-weight: 700;
    color: {section_color};
    margin-top: 10px;
    margin-bottom: 15px;
}}

/* Information cards */

.info-card {{
    background-color: {card_color};
    padding: 22px;
    border-radius: 14px;
    border: 1px solid {border_color};
    margin-bottom: 18px;
}}

/* Prediction result */

.result-card {{
    background: linear-gradient(120deg, #12345a, #2463a6);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-top: 20px;
    margin-bottom: 20px;
}}

.result-label {{
    font-size: 17px;
    color: #DBEAFE;
}}

.result-price {{
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-top: 10px;
}}

/* Input fields */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {{
    background-color: {input_bg};
    border-color: {border_color};
}}

div[data-baseweb="input"] input,
div[data-baseweb="select"] input {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* Metrics */

[data-testid="stMetric"] {{
    background-color: {card_color};
    padding: 15px;
    border-radius: 12px;
    border: 1px solid {border_color};
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
    color: {text_color} !important;
}}

/* Prediction button */

div.stButton > button,
div.stFormSubmitButton > button {{
    background-color: #1769AA;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
    min-height: 48px;
}}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {{
    background-color: #104C80;
    color: white;
}}

/* Footer */

.footer {{
    text-align: center;
    color: {subtitle_color};
    font-size: 13px;
    padding: 25px;
}}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🏡 HouseValue")

    st.markdown("---")

    st.markdown("### Housing Market")

    source = st.selectbox(
        "Select your market",
        ["Ames", "King County"]
    )

    st.markdown("---")

    st.markdown("### About the Project")

    st.write(
        "This application uses machine learning "
        "to estimate residential property prices "
        "based on historical housing data."
    )

    st.markdown("---")

    st.caption("Machine Learning Project")
    st.caption("Ames & King County Housing Markets")

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🏠 House Price Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover the estimated value of a property '
    'using machine learning.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MARKET INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📍 **Selected Market**\n\n" + source)

with col2:
    st.info("🤖 **Technology**\n\nMachine Learning")

with col3:
    st.info("📊 **Prediction Type**\n\nHouse Sale Price")

st.markdown("---")

# --------------------------------------------------
# PROPERTY DETAILS FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🏡 Enter Property Details'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Provide the following information to estimate "
    "the property's sale price."
)

# Dataset-specific year limits

if source == "Ames":
    max_built = 2010
    min_sale = 2006
    max_sale = 2010
    default_sale = 2008
else:
    max_built = 2015
    min_sale = 2014
    max_sale = 2015
    default_sale = 2014

with st.form("house_prediction_form"):

    st.markdown("#### 📐 Property Size")

    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=100,
            value=1500
        )

    with col2:
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=100,
            value=5000
        )

    st.markdown("#### 🛏️ Rooms and Facilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

    with col3:
        basement_area = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            value=500
        )

    st.markdown("#### 📅 Property History")

    col1, col2 = st.columns(2)

    with col1:
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=max_built,
            value=2000
        )

    with col2:
        sale_year = st.number_input(
            "Sale Year",
            min_value=min_sale,
            max_value=max_sale,
            value=default_sale
        )

    renovation = st.selectbox(
        "Has the House Been Renovated?",
        ["No", "Yes"]
    )

    is_renovated = 1 if renovation == "Yes" else 0

    st.markdown("---")

    predict_button = st.form_submit_button(
        "🔍 Predict House Price"
    )

# --------------------------------------------------
# HOUSE PRICE PREDICTION
# --------------------------------------------------

if predict_button:

    if year_built > sale_year:

        st.error(
            "Year built cannot be later than the sale year."
        )

    else:

        house_age = sale_year - year_built

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

        try:

            prediction = model.predict(input_data)[0]

            st.markdown("---")

            st.markdown(
                '<div class="section-title">'
                '✨ Your Prediction Result'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated House Price
                    </div>
                    <div class="result-price">
                        ${prediction:,.2f}
                    </div>
                    <div class="result-label">
                        {source} Housing Market
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📋 Property Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Bedrooms", bedrooms)
            col2.metric("Bathrooms", bathrooms)
            col3.metric("House Age", f"{house_age} years")

            col1, col2 = st.columns(2)

            col1.metric(
                "Living Area",
                f"{living_area:,} sq ft"
            )

            col2.metric(
                "Lot Area",
                f"{lot_area:,} sq ft"
            )

            st.success(
                "Your house price prediction was generated successfully!"
            )

            st.caption(
                "Disclaimer: This is a machine-learning estimate "
                "based on historical housing data. It is not a "
                "guaranteed market price or professional valuation."
            )

        except Exception as e:

            st.error(
                f"Prediction failed. Please check the model and "
                f"input compatibility. Details: {e}"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🏡 House Price Prediction System<br>
        Developed using Python, Streamlit and Machine Learning<br>
        Ames & King County Housing Markets
    </div>
    """,
    unsafe_allow_html=True
)
