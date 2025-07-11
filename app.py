import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="📊 NSE Returns Analyzer", layout="wide")

st.title("📈 NSE Returns Distribution Analyzer")
st.markdown("Analyze Daily, Weekly, and Monthly Returns for any NSE stock or index.")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Enter NSE Ticker Symbol (e.g., RELIANCE.NS, ^NSEI)", value="RELIANCE.NS")
period = st.sidebar.selectbox("Select Time Period", options=["6mo", "1y", "3y", "5y"], index=2)

# --- Fetch Data ---
@st.cache_data(show_spinner=True)
def get_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

data = get_data(ticker, period)

if data.empty:
    st.error("❌ Failed to fetch data. Please check the ticker symbol or internet connection.")
    st.stop()

st.success(f"✅ Data fetched: {len(data)} records from Yahoo Finance")

# --- Calculate Returns ---
data['Daily_Return'] = data['Close'].pct_change().dropna()
daily_returns = data['Daily_Return'].dropna()

weekly_returns = data['Close'].resample('W').ffill().pct_change().dropna()
monthly_returns = data['Close'].resample('M').ffill().pct_change().dropna()

# --- Bell Curve Plot Function ---
def plot_bell_curve(returns, label, color, kde_color):
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    p = norm.pdf(x, mu, std)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(returns, bins=50, kde=True, stat="density", color=color, edgecolor='black')
    ax.plot(x, p, color=kde_color, linewidth=2, label=f"Normal PDF\nμ={mu:.4f}, σ={std:.4f}")
    ax.set_title(f"{label} Returns Distribution")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# --- Show Summary + Plots ---
st.header(f"📊 Summary Statistics: {ticker} ({period})")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Daily Returns")
    st.write(daily_returns.describe())
    plot_bell_curve(daily_returns, "Daily", "skyblue", "red")

with col2:
    st.subheader("📆 Weekly Returns")
    st.write(weekly_returns.describe())
    plot_bell_curve(weekly_returns.squeeze(), "Weekly", "lightgreen", "darkgreen")

with col3:
    st.subheader("📆 Monthly Returns")
    st.write(monthly_returns.describe())
    plot_bell_curve(monthly_returns.squeeze(), "Monthly", "salmon", "darkred")
