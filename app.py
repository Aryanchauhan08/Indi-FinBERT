import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from config import TICKER_LIST, CONFIDENCE_THRESHOLD

# Configure page settings
st.set_page_config(
    page_title="Indian Financial Sentiment MLOps Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Placeholder Github raw URL and local CSV details
GITHUB_RAW_URL = "https://raw.githubusercontent.com/username/FinBERT_Project/main/data/sentiment_log.csv"
LOCAL_CSV_PATH = os.path.join("data", "sentiment_log.csv")


def generate_mock_historical_data():
    """
    Generates realistic historical sentiment data for the tickers.
    Ensures the dashboard is immediately functional and visually complete
    before the workflow runs or if the URL is unreachable.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.date.today(), periods=15, freq="D")
    data = []
    
    classes = ["positive", "neutral", "negative"]
    
    mock_headlines = {
        "positive": [
            "Reliance retail segment expands rapidly in Q1.",
            "HDFC Bank profits surge past estimates with loan growth.",
            "TCS bags multi-million dollar digital transformation deal.",
            "Tata Motors EV bookings record strong momentum.",
            "SBI reports robust asset quality improvements."
        ],
        "neutral": [
            "Infosys schedules quarterly board meeting.",
            "Reliance AGM to highlight key digital expansions.",
            "TCS announces dividend payout for shareholders.",
            "SBI opens new branch networks in suburban areas.",
            "HDFC Bank holds standard interest rate projections."
        ],
        "negative": [
            "Infosys drops target forecast citing IT slowdown.",
            "Tata Motors reports minor domestic sales decline.",
            "SBI faces regulatory inspection inquiries.",
            "Reliance faces production downtime due to maintenance.",
            "HDFC Bank shares dip slightly on marginal pressure."
        ]
    }
    
    for date in dates:
        for ticker in TICKER_LIST:
            # 1-2 headlines per day per ticker
            num_headlines = np.random.randint(1, 3)
            for _ in range(num_headlines):
                pred_class = np.random.choice(classes, p=[0.45, 0.35, 0.20])
                headline = np.random.choice(mock_headlines[pred_class])
                confidence = float(np.random.uniform(0.50, 0.98))
                action = "AUTO_ACCEPTED" if confidence >= CONFIDENCE_THRESHOLD else "FLAGGED_FOR_HUMAN_REVIEW"
                
                data.append({
                    "Date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Ticker": ticker,
                    "Headline": f"[{ticker}] {headline}",
                    "Predicted_Class": pred_class,
                    "Confidence": round(confidence, 4),
                    "Action_Type": action
                })
                
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_data(url):
    """
    Loads daily sentiment results.
    Tries GitHub Raw URL first, falls back to local CSV, and generates mock data if neither exists.
    """
    data_source = ""
    try:
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"])
        data_source = "Loaded from GitHub Raw URL"
        return df, data_source
    except Exception:
        try:
            if os.path.exists(LOCAL_CSV_PATH):
                df = pd.read_csv(LOCAL_CSV_PATH)
                df["Date"] = pd.to_datetime(df["Date"])
                data_source = f"Loaded from local fallback ({LOCAL_CSV_PATH})"
                return df, data_source
            else:
                df = generate_mock_historical_data()
                data_source = "Loaded from generated mock historical data (fallback)"
                return df, data_source
        except Exception as local_err:
            st.error(f"Error loading local data: {local_err}")
            return pd.DataFrame(), "Failed to load"


# Load Sentiment Pipeline Data
df, source_info = load_data(GITHUB_RAW_URL)

# Title & Info
st.markdown('<div class="main-header">Indian Financial Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">MLOps Pipeline Dashboard &bull; <i>{source_info}</i></div>', unsafe_allow_html=True)

# -----------------
# Component 1: Sidebar multi-select for Tickers
# -----------------
st.sidebar.header("Control Panel")
selected_tickers = st.sidebar.multiselect(
    "Select Tickers to Display",
    options=TICKER_LIST,
    default=TICKER_LIST
)

# Apply global ticker filter (avoid errors if none are selected)
if not selected_tickers:
    st.warning("Please select at least one ticker from the sidebar.")
    st.stop()

# -----------------
# Component 2: 7-day rolling average sentiment score (Plotly Express)
# -----------------
st.subheader("7-Day Rolling Average Sentiment Trend")

# Map prediction classes to numeric sentiment values
sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
df_chart = df.copy()
df_chart["Sentiment_Score"] = df_chart["Predicted_Class"].str.lower().map(sentiment_map).fillna(0.0)

# Extract date part to calculate daily average sentiment per ticker
df_chart["Date_Only"] = df_chart["Date"].dt.date
daily_avg = df_chart.groupby(["Ticker", "Date_Only"])["Sentiment_Score"].mean().reset_index()

# Sort to perform correct rolling window operation
daily_avg = daily_avg.sort_values(by=["Ticker", "Date_Only"])

# Calculate 7-day rolling average
daily_avg["7-Day Rolling Avg"] = (
    daily_avg.groupby("Ticker")["Sentiment_Score"]
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

# Filter by selected tickers
filtered_chart_data = daily_avg[daily_avg["Ticker"].isin(selected_tickers)]

# Plot rolling average
if not filtered_chart_data.empty:
    fig = px.line(
        filtered_chart_data,
        x="Date_Only",
        y="7-Day Rolling Avg",
        color="Ticker",
        labels={
            "Date_Only": "Date",
            "7-Day Rolling Avg": "Rolling Sentiment Score (-1.0 to 1.0)"
        },
        height=400,
        template="plotly_white"
    )
    
    # Visual enhancements for Plotly line chart
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB", range=[-1.1, 1.1]),
        margin=dict(l=40, r=40, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data available for the selected tickers to render the trend line chart.")

# -----------------
# Component 3: Today's Headlines and Cascading HITL Status
# -----------------
st.subheader("Today's Headlines & HITL Gating Status")

# Identify records from the most recent run date
if not df.empty:
    latest_run_date = df["Date"].dt.date.max()
    todays_df = df[df["Date"].dt.date == latest_run_date].copy()
    
    # Filter by ticker selection
    todays_df = todays_df[todays_df["Ticker"].isin(selected_tickers)]
    
    # Format dates to string for clean display
    todays_df["Date"] = todays_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
else:
    latest_run_date = datetime.date.today()
    todays_df = pd.DataFrame(columns=["Date", "Ticker", "Headline", "Predicted_Class", "Confidence", "Action_Type"])

# Confidence threshold filter slider in main panel
min_confidence = st.slider(
    "Filter by Minimum Classification Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05
)

# Apply confidence filtering
filtered_headlines = todays_df[todays_df["Confidence"] >= min_confidence]

# KPI summary metrics for the day
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Gated Headlines Today", len(filtered_headlines))
with col2:
    auto_acc = len(filtered_headlines[filtered_headlines["Action_Type"] == "AUTO_ACCEPTED"])
    st.metric("Auto-Accepted (Clean Predictions)", auto_acc)
with col3:
    flagged = len(filtered_headlines[filtered_headlines["Action_Type"] == "FLAGGED_FOR_HUMAN_REVIEW"])
    st.metric("Flagged for Human Review (HITL)", flagged)

# Row highlight styling function
def highlight_action_rows(row):
    action = row["Action_Type"]
    if action == "AUTO_ACCEPTED":
        # Pastel light green background with darker green text
        return ["background-color: #e2f0d9; color: #276a3c"] * len(row)
    elif action == "FLAGGED_FOR_HUMAN_REVIEW":
        # Pastel amber background with darker amber text
        return ["background-color: #fff2cc; color: #8c6b00"] * len(row)
    return [""] * len(row)

# Render Styled Pandas DataFrame
if not filtered_headlines.empty:
    # Sort by timestamp (latest first)
    filtered_headlines = filtered_headlines.sort_values(by="Date", ascending=False)
    
    # Display table with formatting and highlight style applied
    styled_df = (
        filtered_headlines.style
        .apply(highlight_action_rows, axis=1)
        .format({"Confidence": "{:.4f}"})
    )
    
    st.dataframe(styled_df, use_container_width=True, height=350)
else:
    st.info(f"No headlines found matching selected tickers and confidence threshold on the latest run date ({latest_run_date}).")
