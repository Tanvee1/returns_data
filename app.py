import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

st.set_page_config(
    page_title="NSE Returns Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NSE Returns Analyzer")
st.markdown("Analyze stock returns, risk metrics, and distributions.")

# Sidebar
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="RELIANCE.NS"
)

period = st.sidebar.selectbox(
    "Time Period",
    ["6mo", "1y", "3y", "5y", "10y"],
    index=2
)

risk_free_rate = st.sidebar.number_input(
    "Risk Free Rate (%)",
    value=7.0,
    step=0.5
)

# Fetch Data
@st.cache_data
def load_data(ticker, period):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )
    return data

data = load_data(ticker, period)

if data.empty:
    st.error("Unable to fetch data for this ticker.")
    st.stop()

data.index = pd.to_datetime(data.index)

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"].iloc[:, 0]
    open_price = data["Open"].iloc[:, 0]
    high = data["High"].iloc[:, 0]
    low = data["Low"].iloc[:, 0]
else:
    close = data["Close"]
    open_price = data["Open"]
    high = data["High"]
    low = data["Low"]

# Returns
daily_returns = close.pct_change().dropna()

weekly_returns = (
    close.resample("W").last()
    .pct_change()
    .dropna()
)

monthly_returns = (
    close.resample("ME").last()
    .pct_change()
    .dropna()
)

# Metrics
annual_return = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * np.sqrt(252)

sharpe_ratio = (
    annual_return - risk_free_rate / 100
) / annual_volatility

var_95 = np.percentile(daily_returns, 5)

cumulative_returns = (
    1 + daily_returns
).cumprod()

rolling_max = cumulative_returns.cummax()
drawdown = (
    cumulative_returns - rolling_max
) / rolling_max

max_drawdown = drawdown.min()

# Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Annual Return",
    f"{annual_return*100:.2f}%"
)

col2.metric(
    "Volatility",
    f"{annual_volatility*100:.2f}%"
)

col3.metric(
    "Sharpe Ratio",
    f"{sharpe_ratio:.2f}"
)

col4.metric(
    "95% VaR",
    f"{var_95*100:.2f}%"
)

col5.metric(
    "Max Drawdown",
    f"{max_drawdown*100:.2f}%"
)

# Price Chart
st.subheader("Price Chart")

fig_price = go.Figure()

fig_price.add_trace(
    go.Scatter(
        x=close.index,
        y=close.values,
        mode="lines",
        name="Close Price"
    )
)

fig_price.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(
    fig_price,
    use_container_width=True
)

# Candlestick
st.subheader("Candlestick Chart")

fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=data.index,
            open=open_price,
            high=high,
            low=low,
            close=close
        )
    ]
)

fig_candle.update_layout(
    height=600
)

st.plotly_chart(
    fig_candle,
    use_container_width=True
)

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["Daily", "Weekly", "Monthly"]
)

def show_distribution(returns, title):
    mean = returns.mean()
    std = returns.std()

    fig = px.histogram(
        returns,
        nbins=50,
        title=title
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(returns.describe())

with tab1:
    show_distribution(
        daily_returns,
        "Daily Returns Distribution"
    )

with tab2:
    show_distribution(
        weekly_returns,
        "Weekly Returns Distribution"
    )

with tab3:
    show_distribution(
        monthly_returns,
        "Monthly Returns Distribution"
    )

# Cumulative Returns
st.subheader("Cumulative Returns")

fig_cum = go.Figure()

fig_cum.add_trace(
    go.Scatter(
        x=cumulative_returns.index,
        y=(cumulative_returns - 1) * 100,
        mode="lines",
        name="Cumulative Return"
    )
)

fig_cum.update_layout(
    height=500,
    yaxis_title="Return (%)"
)

st.plotly_chart(
    fig_cum,
    use_container_width=True
)

# Download CSV
returns_df = pd.DataFrame({
    "Daily Returns": daily_returns
})

csv = returns_df.to_csv().encode("utf-8")

st.download_button(
    label="📥 Download Daily Returns CSV",
    data=csv,
    file_name=f"{ticker}_returns.csv",
    mime="text/csv"
)
