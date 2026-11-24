import streamlit as st
import pandas as pd
import numpy as np
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

MODEL_FILE = "house_price_model.pkl"
DATA_FILE = "Combined_House_Price_Dataset.csv"
COMPARISON_FILE = "Model_Comparison_Results.csv"


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_housing_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_comparison_data():
    return pd.read_csv(COMPARISON_FILE)


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
/* Keep the selected sidebar dropdown value and its options black */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border-color: #D1D5DB !important;
}}
[data-testid="stSidebar"] div[data-baseweb="select"] * {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}}
[data-baseweb="menu"] {{
    background-color: #FFFFFF !important;
}}
[data-baseweb="menu"] * {{
    color: #000000 !important;
}}
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
.info-card {{
    background-color: {card_color};
    padding: 22px;
    border-radius: 14px;
    border: 1px solid {border_color};
    margin-bottom: 18px;
}}
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
.footer {{
    text-align: center;
    color: {subtitle_color};
    font-size: 13px;
    padding: 25px;
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD THE DATA AND MODEL
# --------------------------------------------------

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

try:
    housing_data = load_housing_data()
except FileNotFoundError:
    st.error(
        f"Dataset '{DATA_FILE}' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the housing dataset: {error}")
    st.stop()

required_columns = {"sale_price", "sale_year", "dataset_source"}
missing_columns = required_columns.difference(housing_data.columns)
if missing_columns:
    st.error(
        "The combined dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

housing_data["sale_price"] = pd.to_numeric(
    housing_data["sale_price"], errors="coerce"
)
housing_data["sale_year"] = pd.to_numeric(
    housing_data["sale_year"], errors="coerce"
)
housing_data = housing_data.dropna(
    subset=["sale_price", "sale_year", "dataset_source"]
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🏡 HouseValue")
    st.markdown("---")
    st.markdown("### Housing Market")

    available_markets = sorted(
        housing_data["dataset_source"].dropna().unique().tolist()
    )
    preferred_order = ["Ames", "King County"]
    market_options = [
        market for market in preferred_order if market in available_markets
    ]
    market_options += [
        market for market in available_markets
        if market not in market_options
    ]

    source = st.selectbox(
        "Select your market",
        market_options
    )

    st.markdown("---")
    st.markdown("### About the Project")
    st.write(
        "This application uses machine learning and historical "
        "housing records to estimate and explore residential property prices."
    )
    st.markdown("---")
    st.caption("Machine Learning Project")
    st.caption("Ames & King County Housing Markets")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏠 House Price Prediction System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Discover estimated property values and explore '
    'historical housing prices using the available dataset.</div>',
    unsafe_allow_html=True
)

market_data = housing_data[
    housing_data["dataset_source"].astype(str) == str(source)
].copy()

if market_data.empty:
    st.warning(f"No housing records were found for {source}.")
    st.stop()

market_min_price = float(market_data["sale_price"].min())
market_max_price = float(market_data["sale_price"].max())

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📍 **Selected Market**\n\n{source}")
with col2:
    st.info(f"🏘️ **Historical Records**\n\n{len(market_data):,}")
with col3:
    first_year = int(market_data["sale_year"].min())
    last_year = int(market_data["sale_year"].max())
    st.info(f"📅 **Dataset Years**\n\n{first_year}–{last_year}")

st.markdown("---")


# --------------------------------------------------
# PROPERTY DETAILS FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 Enter Property Details</div>',
    unsafe_allow_html=True
)
st.write(
    "Provide the following information to estimate the property's sale price."
)

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
            "Living Area (sq ft)", min_value=100, value=1500
        )
    with col2:
        lot_area = st.number_input(
            "Lot Area (sq ft)", min_value=100, value=5000
        )

    st.markdown("#### 🛏️ Rooms and Facilities")
    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms", min_value=0, max_value=20, value=3
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
            "Basement Area (sq ft)", min_value=0, value=500
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
    predict_button = st.form_submit_button("🔍 Predict House Price")


# --------------------------------------------------
# HOUSE PRICE PREDICTION
# --------------------------------------------------

if predict_button:
    if year_built > sale_year:
        st.error("Year built cannot be later than the sale year.")
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
                '<div class="section-title">✨ Your Prediction Result</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated House Price</div>
                    <div class="result-price">${prediction:,.2f}</div>
                    <div class="result-label">{source} Housing Market</div>
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
            col1.metric("Living Area", f"{living_area:,} sq ft")
            col2.metric("Lot Area", f"{lot_area:,} sq ft")

            st.success(
                "Your house price prediction was generated successfully!"
            )
            st.caption(
                "Disclaimer: This is a machine-learning estimate based on "
                "historical housing data. It is not a guaranteed market price "
                "or professional valuation."
            )
        except Exception as error:
            st.error(
                "Prediction failed. Please check the model and input "
                f"compatibility. Details: {error}"
            )


# --------------------------------------------------
# HOUSE PRICE TREND ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="section-title">📊 House Price Trend Analysis</div>',
    unsafe_allow_html=True
)
st.write(
    "Choose a minimum and maximum sale price to explore actual properties "
    "recorded in the selected housing market."
)

price_col1, price_col2 = st.columns(2)

with price_col1:
    min_price = st.number_input(
        "Minimum house price ($)",
        min_value=0.0,
        max_value=market_max_price,
        value=market_min_price,
        step=10000.0,
        format="%.0f",
        key=f"trend_min_{source}"
    )

with price_col2:
    max_price = st.number_input(
        "Maximum house price ($)",
        min_value=0.0,
        max_value=max(market_max_price, 1.0),
        value=market_max_price,
        step=10000.0,
        format="%.0f",
        key=f"trend_max_{source}"
    )

if min_price > max_price:
    st.error("Minimum price cannot be greater than maximum price.")
else:
    filtered_data = market_data[
        market_data["sale_price"].between(min_price, max_price, inclusive="both")
    ].copy()

    st.markdown("#### Results for your selected price range")
    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Matching Houses", f"{len(filtered_data):,}")

    if not filtered_data.empty:
        median_price = filtered_data["sale_price"].median()
        average_price = filtered_data["sale_price"].mean()
        metric2.metric("Median Sale Price", f"${median_price:,.0f}")
        metric3.metric("Average Sale Price", f"${average_price:,.0f}")

        st.markdown("#### 1. Price Distribution")
        st.caption(
            "Number of actual dataset records in each price band. "
            "The bands are calculated from the selected range."
        )

        if min_price == max_price:
            distribution = pd.DataFrame({
                "Price Band": [f"${min_price:,.0f}"],
                "Number of Houses": [len(filtered_data)]
            })
        else:
            bin_edges = np.linspace(min_price, max_price, 9)
            bin_edges = np.unique(bin_edges)

            if len(bin_edges) < 2:
                distribution = pd.DataFrame({
                    "Price Band": [f"${min_price:,.0f}–${max_price:,.0f}"],
                    "Number of Houses": [len(filtered_data)]
                })
            else:
                labels = [
                    f"${bin_edges[i]:,.0f} – ${bin_edges[i + 1]:,.0f}"
                    for i in range(len(bin_edges) - 1)
                ]
                price_bands = pd.cut(
                    filtered_data["sale_price"],
                    bins=bin_edges,
                    labels=labels,
                    include_lowest=True
                )
                counts = (
                    price_bands.value_counts(sort=False)
                    .reindex(labels, fill_value=0)
                )
                distribution = pd.DataFrame({
                    "Price Band": labels,
                    "Number of Houses": counts.values
                })

        st.bar_chart(
            distribution.set_index("Price Band"),
            y="Number of Houses",
            use_container_width=True
        )

        st.markdown("#### 2. Sale Price by Year")
        yearly_trend = (
            filtered_data.groupby("sale_year")
            .agg(
                median_sale_price=("sale_price", "median"),
                average_sale_price=("sale_price", "mean"),
                houses_sold=("sale_price", "size")
            )
            .reset_index()
            .sort_values("sale_year")
        )

        if len(yearly_trend) >= 1:
            chart_data = yearly_trend.set_index("sale_year")[
                ["median_sale_price", "average_sale_price"]
            ]
            st.line_chart(chart_data, use_container_width=True)
            st.caption(
                "The chart shows median and average sale prices for matching "
                "records in each available sale year. It is based on the "
                "years present in this dataset, not a forecast."
            )

            with st.expander("View year-wise figures"):
                display_yearly = yearly_trend.copy()
                display_yearly["median_sale_price"] = (
                    display_yearly["median_sale_price"].map(lambda x: f"${x:,.0f}")
                )
                display_yearly["average_sale_price"] = (
                    display_yearly["average_sale_price"].map(lambda x: f"${x:,.0f}")
                )
                display_yearly = display_yearly.rename(columns={
                    "sale_year": "Sale Year",
                    "median_sale_price": "Median Sale Price",
                    "average_sale_price": "Average Sale Price",
                    "houses_sold": "Matching Records"
                })
                st.dataframe(display_yearly, use_container_width=True, hide_index=True)

        if yearly_trend["sale_year"].nunique() < 3:
            st.info(
                f"The {source} dataset contains only "
                f"{yearly_trend['sale_year'].nunique()} sale year(s) "
                "for this selected price range. Treat the year-wise chart "
                "as a limited historical comparison, not a long-term trend."
            )
    else:
        st.warning(
            "No houses in this market were found within the selected price "
            "range. Try widening the minimum and maximum prices."
        )

st.caption(
    "Trend analysis uses the uploaded Combined_House_Price_Dataset.csv. "
    "It summarizes historical records and does not predict future prices."
)


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="section-title">🤖 Model Performance Comparison</div>',
    unsafe_allow_html=True
)

try:
    comparison_data = load_comparison_data()
    expected_metrics = {"Model", "MAE", "RMSE", "R2 Score"}
    missing_metrics = expected_metrics.difference(comparison_data.columns)

    if missing_metrics:
        st.warning(
            "The model comparison file is missing columns: "
            + ", ".join(sorted(missing_metrics))
        )
    else:
        st.write(
            "These are the model evaluation results recorded in "
            "Model_Comparison_Results.csv."
        )

        comparison_sorted = comparison_data.sort_values(
            "R2 Score", ascending=False
        )

        best_model = comparison_sorted.iloc[0]
        st.info(
            f"Highest R² score in this comparison: **{best_model['Model']}** "
            f"(R² = {best_model['R2 Score']:.3f})."
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**R² Score (higher is better)**")
            r2_chart = comparison_data.set_index("Model")[["R2 Score"]]
            st.bar_chart(r2_chart, use_container_width=True)

        with chart_col2:
            st.markdown("**RMSE (lower is better)**")
            rmse_chart = comparison_data.set_index("Model")[["RMSE"]]
            st.bar_chart(rmse_chart, use_container_width=True)

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True
        )
except FileNotFoundError:
    st.warning(
        f"'{COMPARISON_FILE}' was not found. Add it beside app.py "
        "to display model comparison results."
    )
except Exception as error:
    st.warning(f"Could not display model comparison results: {error}")


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
