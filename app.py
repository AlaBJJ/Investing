import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Crypto Dashboard", page_icon="📈", layout="wide")

# ---------------------------
# Helpers & Config
# ---------------------------
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

def get_cmc_key() -> str:
    # Prefer Streamlit secrets; fall back to env var for local dev
    key = st.secrets.get("CMC_API_KEY") or os.getenv("CMC_API_KEY")
    return key

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cmc_listings(limit=50, convert="USD") -> pd.DataFrame:
    """Fetch top crypto listings from CMC. Returns a tidy DataFrame."""
    key = get_cmc_key()
    if not key:
        raise RuntimeError(
            "Missing CMC_API_KEY. Add it to .streamlit/secrets.toml "
            "and/or Streamlit Cloud → App → Settings → Secrets."
        )

    headers = {"X-CMC_PRO_API_KEY": key}
    params = {"start": 1, "limit": limit, "convert": convert}
    r = requests.get(CMC_URL, headers=headers, params=params, timeout=20)

    if r.status_code == 401:
        raise PermissionError(
            "CMC returned 401 Unauthorized. Check your API key, header name, and plan."
        )
    r.raise_for_status()

    payload = r.json()
    rows = []
    for item in payload.get("data", []):
        quote = item["quote"][convert]
        rows.append({
            "rank": item.get("cmc_rank"),
            "name": item.get("name"),
            "symbol": item.get("symbol"),
            "price": quote.get("price"),
            "market_cap": quote.get("market_cap"),
            "24h %": quote.get("percent_change_24h"),
            "7d %": quote.get("percent_change_7d"),
            "volume_24h": quote.get("volume_24h"),
            "circulating_supply": item.get("circulating_supply"),
        })
    df = pd.DataFrame(rows).sort_values("rank")
    return df

@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_history_yf(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    """Fetch historical price via yfinance, guessing the -USD ticker."""
    # Common mapping for exceptions if needed
    # You can extend this mapping if any coin uses a different yfinance symbol.
    yfin_symbol = f"{symbol}-USD"
    data = yf.download(yfin_symbol, period=period, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()[["Date", "Close"]]
    data.rename(columns={"Close": "close"}, inplace=True)
    return data

def money(x):
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e9:
        return f"${x/1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    return f"${x:,.2f}"

# ---------------------------
# UI – Sidebar
# ---------------------------
st.sidebar.title("⚙️ Settings")
currency = st.sidebar.selectbox("Convert currency", ["USD"], index=0, help="CMC free tier typically uses USD.")
limit = st.sidebar.slider("Number of coins", min_value=10, max_value=100, value=50, step=10)
history_period = st.sidebar.selectbox("History period (yfinance)", ["3mo", "6mo", "1y", "2y"], index=2)
history_interval = st.sidebar.selectbox("History interval", ["1d", "1wk"], index=0)

st.sidebar.caption("Tip: Keep your API key in `.streamlit/secrets.toml` and also in Streamlit Cloud Secrets.")

# ---------------------------
# UI – Header
# ---------------------------
st.title("📈 Crypto Market Dashboard")
st.caption("Data source: CoinMarketCap (listings), yfinance (price history).")

# ---------------------------
# Main – Data Fetch
# ---------------------------
error_box = st.empty()
try:
    with st.spinner("Fetching market data from CoinMarketCap…"):
        df = fetch_cmc_listings(limit=limit, convert=currency)
except PermissionError as e:
    error_box.error(str(e))
    st.stop()
except Exception as e:
    error_box.error(f"Failed to fetch CMC data: {e}")
    st.stop()

# ---------------------------
# Display – Summary KPIs
# ---------------------------
top5 = df.head(5)
total_mcap = df["market_cap"].sum()
st.subheader("Market Snapshot")
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Coins loaded", len(df))
with k2:
    st.metric("Total Market Cap (loaded)", money(total_mcap))
with k3:
    movers = df["24h %"].dropna()
    avg_24h = movers.mean() if not movers.empty else 0.0
    st.metric("Avg 24h Change", f"{avg_24h:.2f}%")

# ---------------------------
# Display – Table
# ---------------------------
st.subheader("Top Coins")
display_df = df.copy()
display_df["price"] = display_df["price"].apply(money)
display_df["market_cap"] = display_df["market_cap"].apply(money)
display_df["volume_24h"] = display_df["volume_24h"].apply(money)
st.dataframe(
    display_df.set_index("rank"),
    use_container_width=True,
    height=480,
)

# ---------------------------
# Display – Chart (Top by Market Cap)
# ---------------------------
st.subheader("Market Cap – Top 10")
chart_df = df.nlargest(10, "market_cap")[["name", "market_cap"]]
fig = go.Figure(
    data=[
        go.Bar(
            x=chart_df["name"],
            y=chart_df["market_cap"],
            text=[money(x) for x in chart_df["market_cap"]],
            textposition="auto",
        )
    ]
)
fig.update_layout(
    xaxis_title="Coin",
    yaxis_title="Market Cap (USD)",
    margin=dict(l=20, r=20, t=20, b=20),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Optional: Price history per symbol
# ---------------------------
st.subheader("Price History (yfinance)")
coin = st.selectbox(
    "Pick a coin to view its USD price history (via yfinance)",
    df["symbol"].tolist(),
)

with st.spinner(f"Loading {coin}-USD history…"):
    hist = fetch_price_history_yf(coin, period=history_period, interval=history_interval)

if hist.empty:
    st.info(f"No yfinance data found for {coin}-USD. Try another coin or a different period.")
else:
    fig2 = go.Figure(
        data=[go.Scatter(x=hist["Date"], y=hist["close"], mode="lines", name=f"{coin}-USD")]
    )
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Note: Some crypto tickers may not exist on yfinance; if a series is empty, select a different symbol.")
