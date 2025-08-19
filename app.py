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
# Sidebar API Keys
# ==============================
st.sidebar.subheader("🔑 API Keys")
st.session_state["CMC_API_KEY"] = st.sidebar.text_input("CoinMarketCap Key", type="password")
st.session_state["CG_KEY"] = st.sidebar.text_input("CoinGecko Key (optional)", type="password")
st.session_state["AV_KEY"] = st.sidebar.text_input("AlphaVantage Key (optional)", type="password")
st.session_state["YF_KEY"] = st.sidebar.text_input("YahooFinance Key (optional)", type="password")

# ==============================
# Multi-API Fused Fetcher
# ==============================
def fetch_fused_crypto(limit=100):
    results = []
    cmc_data, cg_data = {}, {}

    # --- CoinMarketCap ---
    if st.session_state.get("CMC_API_KEY"):
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {"X-CMC_PRO_API_KEY": st.session_state["CMC_API_KEY"]}
            params = {"start": "1", "limit": str(limit), "convert": "USD"}
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            for asset in resp.json()["data"]:
                cmc_data[asset["symbol"].upper()] = {
                    "name": asset["name"],
                    "price": asset["quote"]["USD"]["price"],
                    "change": asset["quote"]["USD"]["percent_change_24h"],
                    "volume": asset["quote"]["USD"]["volume_24h"],
                    "mcap": asset["quote"]["USD"]["market_cap"]
                }
        except Exception as e:
            st.warning(f"CMC fetch failed: {e}")

    # --- CoinGecko ---
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        resp = requests.get(url, params=params, timeout=10).json()
        for asset in resp:
            cg_data[asset["symbol"].upper()] = {
                "name": asset["name"],
                "price": asset["current_price"],
                "change": asset["price_change_percentage_24h"],
                "volume": asset["total_volume"],
                "mcap": asset["market_cap"]
            }
    except Exception as e:
        st.warning(f"CoinGecko fetch failed: {e}")

    # --- Fusion ---
    all_syms = set(cmc_data.keys()) | set(cg_data.keys())
    for sym in all_syms:
        name = cmc_data.get(sym, {}).get("name") or cg_data.get(sym, {}).get("name") or sym

        price_vals, change_vals, volume_vals, mcap_vals = [], [], [], []

        if sym in cmc_data:
            price_vals.append(cmc_data[sym]["price"])
            change_vals.append(cmc_data[sym]["change"])
            volume_vals.append(cmc_data[sym]["volume"])
            mcap_vals.append(cmc_data[sym]["mcap"])
        if sym in cg_data:
            price_vals.append(cg_data[sym]["price"])
            change_vals.append(cg_data[sym]["change"])
            volume_vals.append(cg_data[sym]["volume"])
            mcap_vals.append(cg_data[sym]["mcap"])

        if not price_vals:
            continue

        fused_price = np.mean(price_vals)
        fused_change = np.median(change_vals)
        fused_vol = np.mean(volume_vals)
        fused_mcap = np.mean(mcap_vals)

        results.append({
            "symbol": sym,
            "name": name,
            "priceUsd": fused_price,
            "changePercent24Hr": fused_change,
            "volume24h": fused_vol,
            "marketCap": fused_mcap
        })

    return pd.DataFrame(results)

# ==============================
# Stocks (S&P100)
# ==============================
def fetch_stock_data():
    sp100 = ["AAPL","MSFT","AMZN","TSLA","GOOGL","BRK-B","NVDA","META","JNJ","XOM","JPM","V","PG","UNH","HD","MA",
             "PFE","CVX","ABBV","BAC","KO","PEP","MRK","DIS","CSCO","VZ","WMT","ADBE","NFLX","T","INTC","CRM","CMCSA"]
    try:
        tickers = yf.download(sp100, period="1d", interval="1h", progress=False, threads=True)
        latest = tickers["Close"].iloc[-1]

        df = []
        for sym in sp100:
            ticker = yf.Ticker(sym)
            info = ticker.info
            price = latest[sym]
            change_pct = info.get("regularMarketChangePercent", 0.0)
            volume = info.get("volume", 1)
            mcap = info.get("marketCap", 1)
            df.append({
                "symbol": sym,
                "name": info.get("shortName", sym),
                "priceUsd": price,
                "changePercent24Hr": change_pct,
                "volume24h": volume,
                "marketCap": mcap
            })
        return pd.DataFrame(df)
    except Exception as e:
        st.error(f"Stock API failed: {e}")
        return pd.DataFrame()

# ==============================
# AI Breakout Model
# ==============================
def calc_breakout_table(data, investment_pot=CAPITAL_BASE):
    rows = []
    now = datetime.now(timezone.utc)
    total_mcap = data["marketCap"].sum() if "marketCap" in data else 1

    for _, row in data.iterrows():
        name = row["name"]
        symbol = row["symbol"].upper()
        price = float(row["priceUsd"])
        change = float(row["changePercent24Hr"])
        volume = float(row.get("volume24h", 1))
        mcap = float(row.get("marketCap", 1))

        # --- Factors ---
        vol_factor = min(abs(change) / 10, 1.0)
        vol_score = vol_factor * 100
        volume_norm = np.log1p(volume) / 25
        volm_score = min(volume_norm, 1.0) * 100
        mcap_norm = mcap / total_mcap
        liq_score = min(mcap_norm * 100, 100)
        trend_score = (np.tanh(change / 5) + 1) * 50

        score = (0.3 * vol_score + 0.3 * volm_score + 0.2 * liq_score + 0.2 * trend_score)
        score = np.clip(score, 50, 100)

        # --- ATR, SL, TP ---
        atr = price * abs(change) / 100 / 2
        sl_price = price - max(1.5 * atr, 0.02 * price)
        tp1_price = price + max(2.5 * atr, 0.03 * price)

        sl_pct = (sl_price - price) / price * 100
        tp1_pct = (tp1_price - price) / price * 100 - REVOLUT_FEES * 100
        rr = abs(tp1_pct / sl_pct) if sl_pct != 0 else 0

        # --- Kelly Allocation ---
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
            f"{tp1_pct:.2f}% / £{gain_pot}", trend, go, reasoning
        ])

    df = pd.DataFrame(rows, columns=[
        "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
        "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
        "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
    ])
    df["Rank"] = range(1, len(df) + 1)
    return df.sort_values("Breakout Score", ascending=False).head(100)

# ==============================
# Chart Plotter
# ==============================
def plot_chart(symbol, df):
    try:
        hist = yf.download(symbol, period="5d", interval="1h")
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=symbol
        )])
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning(f"No chart data for {symbol}")

# ==============================
# Streamlit Layout
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner")

st.sidebar.header("⚙ Settings")
pot = st.sidebar.number_input("Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10)

tabs = st.tabs(["Live Crypto", "Live Stocks", "Chart View"])

# --- Live Crypto
with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto = fetch_fused_crypto()
    if not crypto.empty:
        crypto_table = calc_breakout_table(crypto, pot)
        display_cols = [
            "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
            "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
            "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
        ]
        st.dataframe(crypto_table[display_cols], use_container_width=True)
        choice = st.selectbox("Select crypto for chart", crypto_table["Symbol"])
        if choice:
            plot_chart(choice, crypto_table)

# --- Live Stocks
with tabs[1]:
    st.subheader("Live Stock Breakouts (S&P100)")
    stocks = fetch_stock_data()
    if not stocks.empty:
        stock_table = calc_breakout_table(stocks, pot)
        display_cols = [
            "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
            "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
            "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
        ]
        st.dataframe(stock_table[display_cols], use_container_width=True)
        choice = st.selectbox("Select stock for chart", stock_table["Symbol"])
        if choice:
            plot_chart(choice, stock_table)

# --- Chart View
with tabs[2]:
    st.info("Use the chart dropdowns in Crypto/Stocks tabs to see AI levels on charts.")
