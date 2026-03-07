import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models"

def resolve_existing_path(*relative_options: str) -> str:
    """Return the first existing path from provided repo-relative options."""
    for rel in relative_options:
        candidate = BASE_DIR / rel
        if candidate.exists():
            return str(candidate)
    # Fall back to the first option so downstream errors are explicit
    return str(BASE_DIR / relative_options[0])

tfidf = TfidfVectorizer(stop_words='english')

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Smart E-Commerce Analytics Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM STYLE (Modern Dashboard Theme)
# -------------------------------------------------------
st.markdown("""
<style>

/* -------------------------------------------------
MAIN APP BACKGROUND
------------------------------------------------- */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 20%, rgba(59,130,246,0.18), transparent 45%),
        radial-gradient(circle at 85% 25%, rgba(236,72,153,0.15), transparent 45%),
        radial-gradient(circle at 50% 85%, rgba(34,197,94,0.12), transparent 45%),
        linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    background-attachment: fixed;
}


/* -------------------------------------------------
SIDEBAR
------------------------------------------------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
}


/* -------------------------------------------------
MAIN CONTENT CONTAINER (GLASS EFFECT)
------------------------------------------------- */

.block-container {
    background: rgba(15,23,42,0.55);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}


/* -------------------------------------------------
METRIC CARDS
------------------------------------------------- */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1e293b, #020617);
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}


/* Hover animation for metrics */

[data-testid="stMetric"]:hover {
    transform: translateY(-6px);
    box-shadow: 0px 18px 35px rgba(0,0,0,0.45);
}


/* Metric number styling */

[data-testid="stMetricValue"] {
    font-size: 32px;
    font-weight: 700;
}


/* -------------------------------------------------
PLOTLY CHART CONTAINER
------------------------------------------------- */

[data-testid="stPlotlyChart"] {
    background: rgba(2,6,23,0.45);
    border-radius: 12px;
    padding: 12px;
}


/* -------------------------------------------------
DATA TABLES
------------------------------------------------- */

[data-testid="stDataFrame"] {
    background: rgba(2,6,23,0.45);
    border-radius: 10px;
    padding: 8px;
}


/* -------------------------------------------------
HEADERS
------------------------------------------------- */

h1, h2, h3 {
    font-weight: 600;
}


/* -------------------------------------------------
DIVIDER STYLE
------------------------------------------------- */

hr {
    border: 1px solid rgba(148,163,184,0.15);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    customers = pd.read_csv(
        resolve_existing_path("Data/cleaned_customer.csv", "cleaned_customer.csv")
    )
    products = pd.read_csv(
        resolve_existing_path("Data/cleaned_product.csv", "cleaned_product.csv")
    )
    transactions = pd.read_csv(
        resolve_existing_path("Data/cleaned_transaction.csv", "cleaned_transaction.csv")
    )
    return customers, products, transactions

customers, products, transactions = load_data()

@st.cache_resource
def build_tfidf(products):

    product_list = (
        products.iloc[:,0]
        .astype(str)
        .str.lower()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    tfidf = TfidfVectorizer(stop_words="english")

    product_vectors = tfidf.fit_transform(product_list)

    return tfidf, product_vectors, product_list


tfidf, product_vectors, product_list = build_tfidf(products)

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
# Segmentation
kmeans = joblib.load(resolve_existing_path("Models/kmeans_model.pkl", "kmeans_model.pkl"))
seg_scaler = joblib.load(resolve_existing_path("Models/seg_scaler.pkl", "seg_scaler.pkl"))
seg_feature_cols = joblib.load(resolve_existing_path("Models/seg_feature_columns.pkl", "seg_feature_columns.pkl"))

# Churn
churn_model = joblib.load(resolve_existing_path("Models/churn_model.pkl", "churn_model.pkl"))
encoders = joblib.load(resolve_existing_path("Models/churn_encoders.pkl", "churn_encoders.pkl"))
feature_cols = joblib.load(resolve_existing_path("Models/feature_columns.pkl", "feature_columns.pkl"))
lstm_model = load_model(resolve_existing_path("Models/sales_lstm_model.h5", "sales_lstm_model.h5"),compile=False,safe_mode=False)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Select Business Module",
    [
        "Executive Overview",
        "Customer Segmentation",
        "Churn Prediction",
        "Sales Forecasting",
        "Product Recommendation"
    ]
)

# =======================================================
# 1️⃣ EXECUTIVE OVERVIEW
# =======================================================
if menu == "Executive Overview":

    # ---------------------------------------------------
    # HERO HEADER (Fashion Banner + Title)
    # ---------------------------------------------------

    st.markdown(
        """
        <div style="
            background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.75)),
            url('https://images.unsplash.com/photo-1441986300917-64674bd600d8');
            background-size: cover;
            background-position: center;
            padding: 60px 30px;
            border-radius: 14px;
            margin-bottom: 20px;
        ">
            <h1 style="color:white;">🛍️ Smart Fashion E-Commerce Analytics</h1>
            <p style="color:#d1d5db;font-size:18px;">
            Business intelligence dashboard for customers, products, and transactions
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------------------------------------------
    # CLEAN COLUMN NAMES
    # ---------------------------------------------------

    customers.columns = customers.columns.str.strip()
    products.columns = products.columns.str.strip()
    transactions.columns = transactions.columns.str.strip()

    # ---------------------------------------------------
    # KPI METRICS
    # ---------------------------------------------------

    total_customers = len(customers)
    total_products = len(products)
    total_transactions = len(transactions)

    st.markdown("### 📊 Platform Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("👥 Total Customers", f"{total_customers:,}")
    c2.metric("👗 Total Products", f"{total_products:,}")
    c3.metric("🧾 Total Transactions", f"{total_transactions:,}")

    st.divider()

    # ---------------------------------------------------
    # REVENUE DISTRIBUTION (INTERACTIVE)
    # ---------------------------------------------------

    st.subheader("💰 Revenue Distribution")

    if "Amount" in transactions.columns:

        fig = px.histogram(
            transactions,
            x="Amount",
            nbins=40,
            template="plotly_dark",
            title="Order Amount Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Revenue column not available.")

    # ---------------------------------------------------
    # CUSTOMER OVERVIEW
    # ---------------------------------------------------

    st.subheader("👥 Customer Overview")

    col1, col2 = st.columns([1.2,1])

    with col1:
        st.write("### Sample Customers")
        st.dataframe(customers.head(), use_container_width=True)

    with col2:

        if "Churn" in customers.columns:

            churn_df = customers["Churn"].value_counts().reset_index()
            churn_df.columns = ["Churn", "Count"]

            fig2 = px.bar(
                churn_df,
                x="Churn",
                y="Count",
                color="Churn",
                template="plotly_dark",
                title="Customer Churn Distribution"
            )

            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("Churn column not available.")

    # ---------------------------------------------------
    # PRODUCT OVERVIEW
    # ---------------------------------------------------

    st.subheader("👗 Product Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Sample Products")

        st.dataframe(
            products.head(),
            use_container_width=True
        )

    with col2:

        if "prod_name" in products.columns:

            st.write("### 🔥 Trending Products")

            top_products = (
                products["prod_name"]
                .value_counts()
                .head(5)
                .reset_index()
            )

            top_products.columns = ["Product Name", "Count"]

            fig3 = px.bar(
                top_products,
                x="Product Name",
                y="Count",
                color="Count",
                template="plotly_dark",
                title="Top Fashion Items"
            )

            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("Column 'prod_name' not found.")

    # ---------------------------------------------------
    # TRANSACTION OVERVIEW
    # ---------------------------------------------------

    st.divider()

    st.subheader("🧾 Transaction Overview")

    st.dataframe(
        transactions.head(),
        use_container_width=True
    )

# =======================================================
# 2️⃣ CUSTOMER SEGMENTATION
# =======================================================

elif menu == "Customer Segmentation":

    st.header("Customer Segmentation (RFM + K-Means)")
    st.markdown(
        "Segment customers based on **Recency**, **Frequency**, and **Monetary value**."
    )

    df = customers

    # -----------------------------------
    # Dataset Ranges
    # -----------------------------------

    recency_min = int(df["DaySinceLastOrder"].min())
    recency_max = int(df["DaySinceLastOrder"].max())

    freq_min = int(df["OrderCount"].min())
    freq_max = int(df["OrderCount"].max())

    mon_min = float(df["CashbackAmount"].min())
    mon_max = float(df["CashbackAmount"].max())

    st.divider()

    st.subheader("Enter RFM Values")

    col1, col2 = st.columns(2)

    # -----------------------------------
    # Recency + Monetary
    # -----------------------------------

    with col1:

        st.caption("Recommended Dataset Range : 150 — 220 days")

        recency = st.number_input(
            "Recency (Days Since Last Order)",
            value=20,
            step=1
        )

        st.caption("Recommended Dataset Range : ₹5000 — ₹20000")

        monetary = st.number_input(
            "Monetary (Cashback Amount)",
            value=150.0,
            step=1.0
        )

    # -----------------------------------
    # Frequency
    # -----------------------------------

    with col2:

        st.caption("Recommended Dataset Range : 1 — 20 orders")

        frequency = st.number_input(
            "Frequency (Total Orders)",
            value=2,
            step=1
        )

    st.divider()

    # -----------------------------------
    # Predict Button
    # -----------------------------------

    if st.button("Predict Customer Segment", width="stretch"):

        # Prepare input using same feature names as training
        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        # Scale features
        scaled_input = seg_scaler.transform(input_df)

        # Predict cluster
        cluster = kmeans.predict(scaled_input)[0]

        st.write("Predicted Cluster:", cluster)

        # -----------------------------------
        # Cluster Business Labels   
        # (based on cluster analysis)
        # -----------------------------------

        segment_labels = {

            2: "Low Value Customers",
            1: "Medium Value Customers",
            3: "High Value Customers",
            0: "Premium Loyal Customers"

        }

        segment_name = segment_labels.get(cluster, "Unknown Segment")

        # -----------------------------------
        # Display Result
        # -----------------------------------

        st.success("Customer Segment Identified")

        st.markdown(f"### 🎯 {segment_name}")

        # Optional explanation for recruiters
        st.info(
            """
            **Segment Meaning**

            • Premium Loyal Customers → frequent purchases and high spending  
            • High Value Customers → high spending customers  
            • Medium Value Customers → moderate engagement and spending  
            • Low Value Customers → low spending and infrequent purchases
            """
        )


# =======================================================
# 3️⃣ CHURN PREDICTION
# =======================================================
elif menu == "Churn Prediction":

    st.header("Customer Churn Risk Analysis")

    # --------------------------------------------------
    # COPY & CLEAN DATA
    # --------------------------------------------------

    df = customers.copy()
    df.columns = df.columns.str.strip()

    # --------------------------------------------------
    # TRAINING MIN / MAX
    # --------------------------------------------------

    numeric_columns = df.select_dtypes(include="number").columns

    min_max_dict = {
        col: (df[col].min(), df[col].max())
        for col in numeric_columns
    }

    # --------------------------------------------------
    # PREPARE INPUT DF
    # --------------------------------------------------

    input_df = pd.DataFrame(columns=feature_cols)
    input_df.loc[0] = 0

    st.divider()

    out_of_range_features = []

    # --------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------

    for col in feature_cols:

        # ==========================
        # CATEGORICAL FEATURES
        # ==========================

        if col in encoders:

            encoder = encoders[col]

            # Only allow trained values
            options = list(encoder.classes_)
            selected_value = st.selectbox(col, sorted(options))

            encoded_value = encoder.transform([selected_value])[0]
            input_df.loc[0, col] = encoded_value

        # ==========================
        # NUMERIC FEATURES
        # ==========================

        else:

            # ---- CityTier (ordinal category)
            if col == "CityTier":

                value = st.selectbox(
                    "CityTier",
                    options=[1, 2, 3],
                    index=1
                )

            # ---- Binary Features
            elif col in ["Complain"]:

                value = st.selectbox(
                    col,
                    options=[0, 1]
                )

            # ---- Integer Features
            elif col in [
                "SatisfactionScore",
                "NumberOfDeviceRegistered",
                "OrderCount",
                "CouponUsed"
            ]:

                min_val, max_val = min_max_dict[col]

                value = st.number_input(
                    f"{col} (Range: {int(min_val)} — {int(max_val)})",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(df[col].mean()),
                    step=1
                )

            # ---- Continuous Features
            else:

                min_val, max_val = min_max_dict[col]

                value = st.number_input(
                    f"{col} (Training Range: {min_val:.2f} — {max_val:.2f})",
                    value=float(df[col].mean()),
                    step=0.1
                )

                if value < min_val or value > max_val:
                    out_of_range_features.append(
                        f"{col}: Allowed {min_val:.2f} — {max_val:.2f}, Entered {value:.2f}"
                    )

            input_df.loc[0, col] = value

    st.divider()

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if st.button("Predict Churn Probability", use_container_width=True):

        if out_of_range_features:
            st.warning("⚠ The following inputs are outside training range:")
            for item in out_of_range_features:
                st.write("•", item)
            st.info("Prediction reliability may reduce.")

        probability = churn_model.predict_proba(input_df)[0][1]

        st.metric("Churn Probability", f"{probability:.2f}")

        if probability > 0.5:
            st.error("⚠ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

# =======================================================
# 4️⃣ SALES FORECASTING
# =======================================================

elif menu == "Sales Forecasting":

    import joblib
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import streamlit as st

    st.header("📈 AI Sales Forecasting")
    st.markdown(
        "Predict future revenue using a trained **LSTM deep learning model**."
    )

    st.divider()

    # --------------------------------------------------
    # LOAD MODEL & SCALER
    # --------------------------------------------------

    scaler = None

    # LOAD SCALER
    try:
        scaler = joblib.load(resolve_existing_path("Models/scaler.pkl", "scaler.pkl"))
    except Exception as e:
        scaler = None
        st.warning(f"⚠️ Scaler not found: {e}")

    # LOAD LSTM MODEL
    try:

        model_path = resolve_existing_path("Models/sales_lstm_model.h5", "sales_lstm_model.h5")

        lstm_model = load_model(model_path, compile=False)

    except Exception as e:

        st.error(f"❌ LSTM model loading failed: {e}")

        st.stop()

    # --------------------------------------------------
    # DETECT DATE COLUMN
    # --------------------------------------------------

    possible_dates = [
        "OrderDate",
        "InvoiceDate",
        "TransactionDate",
        "Date",
        "order_date"
    ]

    date_col = next(
        (c for c in possible_dates if c in transactions.columns),
        None
    )

    if date_col is None:
        st.error("❌ No valid date column found.")
        st.stop()

    if "Amount" not in transactions.columns:
        st.error("❌ Amount column missing.")
        st.stop()

    # --------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------

    sales_df = transactions.copy()

    sales_df.rename(
        columns={date_col: "OrderDate"},
        inplace=True
    )

    sales_df["OrderDate"] = pd.to_datetime(
        sales_df["OrderDate"],
        errors="coerce"
    )

    sales_df.dropna(
        subset=["OrderDate", "Amount"],
        inplace=True
    )

    # --------------------------------------------------
    # CREATE DAILY SALES
    # --------------------------------------------------

    daily_sales = (
        sales_df
        .groupby("OrderDate")["Amount"]
        .sum()
        .asfreq("D", fill_value=0)
        .reset_index()
    )

    st.subheader("📊 Last 90 Days Revenue Trend")

    fig = px.line(
        daily_sales.tail(90),
        x="OrderDate",
        y="Amount",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # FORECAST SETTINGS
    # --------------------------------------------------

    lookback = 30

    forecast_days = st.slider(
        "Select forecast horizon",
        7,
        60,
        30
    )

    if len(daily_sales) < lookback:
        st.warning("Not enough historical data.")
        st.stop()

    # --------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------

    last_values = daily_sales["Amount"].tail(lookback).values

    if scaler:
        dummy = np.zeros((len(last_values),6))
        dummy[:,0] = last_values

        scaled = scaler.transform(dummy)
    else:
        scaled = last_values.reshape(-1, 1)

    sequence = scaled.reshape(1, lookback, 6)

    # --------------------------------------------------
    # FORECAST BUTTON
    # --------------------------------------------------

    if st.button("🚀 Generate Forecast", use_container_width=True):

        predictions = []

        current_seq = sequence.copy()

        for _ in range(forecast_days):

            pred = lstm_model.predict(current_seq, verbose=0)

            predictions.append(pred[0][0])

            new_row = np.zeros((1,1,6))
            new_row[0,0,0] = pred[0][0]

            current_seq = np.append(
                current_seq[:,1:,:],
                new_row,
                axis=1
            )

        predictions = np.array(predictions).reshape(-1, 1)

        if scaler:
            dummy = np.zeros((len(predictions), 6))
            dummy[:,0] = predictions[:,0]

            dummy = scaler.inverse_transform(dummy)

            predictions = dummy[:,0].reshape(-1,1)

        predictions = predictions.flatten()

        # --------------------------------------------------
        # CREATE FORECAST DATAFRAME
        # --------------------------------------------------

        last_date = daily_sales["OrderDate"].max()

        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=forecast_days
        )

        forecast_df = pd.DataFrame({
            "OrderDate": future_dates,
            "Predicted": predictions
        })

        # --------------------------------------------------
        # DISPLAY METRICS
        # --------------------------------------------------

        st.subheader("📊 Forecast Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Avg Forecast",
            f"₹{forecast_df['Predicted'].mean():,.0f}"
        )

        col2.metric(
            "Max Forecast",
            f"₹{forecast_df['Predicted'].max():,.0f}"
        )

        col3.metric(
            "Min Forecast",
            f"₹{forecast_df['Predicted'].min():,.0f}"
        )

        st.divider()

        # --------------------------------------------------
        # FORECAST GRAPH
        # --------------------------------------------------

        st.subheader("🔮 Sales Forecast")

        fig2 = go.Figure()

        # Historical sales
        fig2.add_trace(
            go.Scatter(
                x=daily_sales.tail(60)["OrderDate"],
                y=daily_sales.tail(60)["Amount"],
                name="Historical Sales",
                mode="lines",
                line=dict(color="#4C78FF", width=3)
            )
        )

        # Forecast line
        fig2.add_trace(
            go.Scatter(
                x=forecast_df["OrderDate"],
                y=forecast_df["Predicted"],
                name="Forecast",
                mode="lines",
                line=dict(color="#FF4B4B", width=3)
            )
        )

        # Connect last actual point to first forecast
        fig2.add_trace(
            go.Scatter(
                x=[
                    daily_sales["OrderDate"].iloc[-1],
                    forecast_df["OrderDate"].iloc[0]
                ],
                y=[
                    daily_sales["Amount"].iloc[-1],
                    forecast_df["Predicted"].iloc[0]
                ],
                mode="lines",
                line=dict(color="orange", dash="dot"),
                showlegend=False
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            title="Next Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        fig2.update_layout(
            template="plotly_dark",
            title="Next Sales Forecast"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # ERROR ANALYSIS
        # --------------------------------------------------

        st.subheader("📉 Model Evaluation")

        y_true = daily_sales["Amount"].tail(30).values
        y_pred = np.repeat(forecast_df["Predicted"].mean(), len(y_true))

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        col1, col2 = st.columns(2)

        col1.metric("RMSE", f"₹{rmse:,.0f}")
        col2.metric("MAE", f"₹{mae:,.0f}")

        st.divider()

        # --------------------------------------------------
        # MODEL INSIGHT
        # --------------------------------------------------

        st.subheader("📌 Forecast Insight")

        avg_sales = daily_sales["Amount"].mean()

        if forecast_df["Predicted"].mean() > avg_sales:

            st.success("📈 Demand expected to be above normal.")

        else:

            st.warning("📉 Demand may stay below average.")


# =======================================================
# 5️⃣ PRODUCT RECOMMENDATION
# =======================================================
elif menu == "Product Recommendation":

    st.header("Content-Based Product Recommendation")

    @st.cache_resource
    def build_recommender(products_df):
        product_list = (
            products_df.iloc[:, 0]
            .astype(str)
            .str.lower()
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        product_vectors = vectorizer.fit_transform(product_list)

        return vectorizer, product_vectors, product_list


    # Build once
    tfidf, product_vectors, product_list = build_recommender(products)

    selected_product = st.selectbox(
        "Select a Product",
        product_list
    )

    if st.button("Recommend Products"):

        selected_vector = tfidf.transform([selected_product])

        similarity_scores = cosine_similarity(
            selected_vector,
            product_vectors
        )

        # Get top 5 similar products (excluding itself)
        top_indices = similarity_scores.argsort()[0][-6:-1][::-1]

        st.subheader("Recommended Products")

        for idx in top_indices:
            st.write(product_list[idx])