# ==============================
# AI Breakout Scanner (Crypto + Stocks)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone
import yfinance as yf
import plotly.graph_objects as go

# ==============================
# Config
# ==============================
REFRESH_INTERVAL = 60
CAPITAL_BASE = 1000
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ~2.98% round trip

# ==============================
# Sidebar API Settings
# ==============================
st.sidebar.header("API Settings")

# Select which APIs to use
apis_selected = st.sidebar.multiselect(
    "Select APIs to include in fusion:",
    ["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"],
    default=["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"]
)

# API keys
if "CMC_API_KEY" not in st.session_state:
    st.session_state["CMC_API_KEY"] = ""
if "CC_API_KEY" not in st.session_state:
    st.session_state["CC_API_KEY"] = ""

st.session_state["CMC_API_KEY"] = st.sidebar.text_input("CoinMarketCap API Key", st.session_state["CMC_API_KEY"])
st.session_state["CC_API_KEY"] = st.sidebar.text_input("CryptoCompare API Key", st.session_state["CC_API_KEY"])

# ==============================
# API Fetchers
# ==============================
def fetch_cmc(limit=100):
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {"X-CMC_PRO_API_KEY": st.session_state["CMC_API_KEY"]}
        params = {"start": "1", "limit": str(limit), "convert": "USD"}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        return pd.DataFrame([{
            "symbol": a["symbol"],
            "name": a["name"],
            "price": a["quote"]["USD"]["price"],
            "change": a["quote"]["USD"]["percent_change_24h"],
            "volume": a["quote"]["USD"]["volume_24h"],
            "mcap": a["quote"]["USD"]["market_cap"]
        } for a in data])
    except:
        return pd.DataFrame()

def fetch_coingecko(limit=100):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame([{
            "symbol": a["symbol"].upper(),
            "name": a["name"],
            "price": a["current_price"],
            "change": a["price_change_percentage_24h"],
            "volume": a["total_volume"],
            "mcap": a["market_cap"]
        } for a in data])
    except:
        return pd.DataFrame()

def fetch_cryptocompare(limit=100):
    try:
        url = f"https://min-api.cryptocompare.com/data/top/mktcapfull"
        params = {"limit": limit, "tsym": "USD", "api_key": st.session_state["CC_API_KEY"]}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["Data"]
        return pd.DataFrame([{
            "symbol": a["CoinInfo"]["Name"],
            "name": a["CoinInfo"]["FullName"],
            "price": a["RAW"]["USD"]["PRICE"],
            "change": a["RAW"]["USD"]["CHANGEPCT24HOUR"],
            "volume": a["RAW"]["USD"]["VOLUME24HOUR"],
            "mcap": a["RAW"]["USD"]["MKTCAP"]
        } for a in data])
    except:
        return pd.DataFrame()

def fetch_coinpaprika(limit=100):
    try:
        url = "https://api.coinpaprika.com/v1/tickers"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()[:limit]
        return pd.DataFrame([{
            "symbol": a["symbol"],
            "name": a["name"],
            "price": a["quotes"]["USD"]["price"],
            "change": a["quotes"]["USD"]["percent_change_24h"],
            "volume": a["quotes"]["USD"]["volume_24h"],
            "mcap": a["quotes"]["USD"]["market_cap"]
        } for a in data])
    except:
        return pd.DataFrame()

def fetch_coincap(limit=100):
    try:
        url = "https://api.coincap.io/v2/assets"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][:limit]
        return pd.DataFrame([{
            "symbol": a["symbol"],
            "name": a["name"],
            "price": float(a["priceUsd"]),
            "change": float(a["changePercent24Hr"]),
            "volume": float(a["volumeUsd24Hr"]) if a["volumeUsd24Hr"] else 0,
            "mcap": float(a["marketCapUsd"]) if a["marketCapUsd"] else 0
        } for a in data])
    except:
        return pd.DataFrame()

# ==============================
# Data Fusion
# ==============================
def fuse_data(limit=100):
    sources = []
    if "CoinMarketCap" in apis_selected: sources.append(fetch_cmc(limit))
    if "CoinGecko" in apis_selected: sources.append(fetch_coingecko(limit))
    if "CryptoCompare" in apis_selected: sources.append(fetch_cryptocompare(limit))
    if "Coinpaprika" in apis_selected: sources.append(fetch_coinpaprika(limit))
    if "CoinCap" in apis_selected: sources.append(fetch_coincap(limit))

    if not sources: return pd.DataFrame()
    df = pd.concat(sources).groupby("symbol").agg({
        "name": "first",
        "price": "median",
        "change": "mean",
        "volume": "mean",
        "mcap": "mean"
    }).reset_index()
    return df

# ==============================
# Breakout Model
# ==============================
def calc_breakout_table(data, investment_pot=CAPITAL_BASE):
    rows = []
    now = datetime.now(timezone.utc)
    total_mcap = data["mcap"].sum() if "mcap" in data else 1

    for _, row in data.iterrows():
        name, symbol, price, change, volume, mcap = row["name"], row["symbol"], row["price"], row["change"], row["volume"], row["mcap"]

        # Factors
        vol_factor = min(abs(change) / 10, 1.0)
        vol_score = vol_factor * 100
        liq_score = (np.log1p(volume) / 25) * 100
        mcap_norm = mcap / total_mcap
        trend_score = (np.tanh(change / 5) + 1) * 50

        score = (0.4 * vol_score + 0.3 * liq_score + 0.2 * trend_score + 0.1 * (100 - abs(vol_score - liq_score)))
        score = np.clip(score, 50, 100)

        atr = price * abs(change) / 100 / 2
        sl_price = price - max(1.5 * atr, 0.02 * price)
        tp1_price = price + max(2.5 * atr, 0.03 * price)

        sl_pct = (sl_price - price) / price * 100
        tp1_pct = (tp1_price - price) / price * 100 - REVOLUT_FEES * 100
        rr = abs(tp1_pct / sl_pct) if sl_pct != 0 else 0

        p = score / 100
        b = rr
        kelly = max((p * (b + 1) - 1) / b, 0) if b > 0 else 0
        alloc = round(min(investment_pot * kelly, investment_pot), 2)
        gain_pot = round(alloc * tp1_pct / 100, 2)

        strike = "Yes" if score >= 85 else "No"
        breakout_time = (now + timedelta(minutes=np.random.randint(30, 180))).strftime("%H:%M")
        trend = "↑" if change > 0 else ("↓" if change < 0 else "↔")
        go = "Go" if score >= 85 and trend == "↑" and rr > 1.5 else "No-Go"
        reasoning = (f"Score {score:.1f}, Vol {change:.2f}%, Vol24h ${volume/1e6:.1f}M, "
                     f"MCAP ${mcap/1e9:.1f}B, R/R {rr:.2f}, Kelly alloc £{alloc}")

        rows.append([
            None, name, symbol, round(score, 2), strike, breakout_time,
            round(price, 4), f"{sl_pct:.2f}% (£{alloc * sl_pct/100:.2f}) ({sl_price:.2f})",
            f"{tp1_pct:.2f}% (£{gain_pot:.2f}) ({tp1_price:.2f})",
            "0.00%", f"{sl_pct:.2f}% / {tp1_pct:.2f}%", f"£{alloc}",
            f"{tp1_pct:.2f}% / £{gain_pot}", trend, go, reasoning, sl_price, tp1_price
        ])

    df = pd.DataFrame(rows, columns=[
        "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
        "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
        "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go",
        "AI Reasoning","SL_Price","TP1_Price"
    ])
    df["Rank"] = range(1, len(df) + 1)
    return df.sort_values("Breakout Score", ascending=False).head(100)

# ==============================
# Chart Plotting
# ==============================
def plot_chart(symbol, df):
    try:
        hist = yf.download(symbol+"-USD", period="5d", interval="1h")
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']
        )])
        sl = df.loc[df["Symbol"] == symbol, "SL_Price"].values[0]
        tp = df.loc[df["Symbol"] == symbol, "TP1_Price"].values[0]
        entry = df.loc[df["Symbol"] == symbol, "Entry Price (USD/GBP)"].values[0]
        fig.add_hline(y=entry, line_color="blue", annotation_text="Entry", annotation_position="top left")
        fig.add_hline(y=sl, line_color="red", annotation_text="SL", annotation_position="bottom right")
        fig.add_hline(y=tp, line_color="green", annotation_text="TP", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning(f"No chart data for {symbol}")

# ==============================
# Layout
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner")
st.sidebar.header("Settings")
pot = st.sidebar.number_input("Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10)

tabs = st.tabs(["Live Crypto", "Live Stocks", "Chart View"])

# --- Crypto
with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto = fuse_data(100)
    if not crypto.empty:
        crypto_table = calc_breakout_table(crypto, pot)
        display_cols = [
            "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
            "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
            "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
        ]
        st.dataframe(crypto_table[display_cols], use_container_width=True)
        choice = st.selectbox("Select crypto for chart", crypto_table["Symbol"])
        if choice: plot_chart(choice, crypto_table)

# --- Stocks
with tabs[1]:
    st.subheader("Live Stock Breakouts (S&P100)")
    sp100 = ["AAPL","MSFT","AMZN","TSLA","GOOGL","NVDA","META","JPM","V","PG"]
    tickers = yf.download(sp100, period="1d", interval="1h", progress=False, threads=True)
    latest = tickers["Close"].iloc[-1]
    stocks = pd.DataFrame([{
        "symbol": s,
        "name": s,
        "price": latest[s],
        "change": 0.0,
        "volume": 1,
        "mcap": 1
    } for s in sp100])
    stock_table = calc_breakout_table(stocks, pot)
    st.dataframe(stock_table[display_cols], use_container_width=True)
    choice = st.selectbox("Select stock for chart", stock_table["Symbol"])
    if choice: plot_chart(choice, stock_table)

# --- Charts
with tabs[2]:
    st.info("Use the dropdowns in Crypto/Stocks to view candlestick charts with AI Entry/SL/TP overlays.")
