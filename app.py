import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# ==============================
# Config
# ==============================
REFRESH_INTERVAL = 60  # seconds
PORTFOLIO_FILE = "portfolio.json"
CAPITAL_BASE = 1000  # default investment pot
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ~2.98% round trip

# ==============================
# Helpers
# ==============================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return []

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)

def fetch_crypto_data():
    """Get crypto prices (CoinMarketCap free API as placeholder)."""
    url = "https://api.coincap.io/v2/assets"
    try:
        resp = requests.get(url, timeout=10).json()
        data = pd.DataFrame(resp["data"])
        data["priceUsd"] = data["priceUsd"].astype(float)
        data["changePercent24Hr"] = data["changePercent24Hr"].astype(float)
        return data
    except Exception as e:
        st.error(f"Crypto API failed: {e}")
        return pd.DataFrame()

def fetch_stock_data(symbols=["AAPL", "MSFT", "TSLA"]):
    """Yahoo finance fallback via yfinance would be better, here dummy."""
    return pd.DataFrame({
        "symbol": symbols,
        "price": np.random.uniform(100, 300, len(symbols)),
        "change": np.random.uniform(-5, 5, len(symbols))
    })

def calc_breakout_table(data, investment_pot=CAPITAL_BASE, currency="USD"):
    rows = []
    now = datetime.utcnow()

    for _, row in data.iterrows():
        name = row.get("name", row["symbol"])
        symbol = row["symbol"].upper()
        price = float(row.get("priceUsd", row.get("price", 0)))
        change = float(row.get("changePercent24Hr", row.get("change", 0)))
        vol_factor = min(abs(change) / 10, 1.0)
        score = np.clip(50 + vol_factor * 50, 50, 100)
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
        reasoning = f"Score {score:.1f}, R/R {rr:.2f}, ATR {atr:.2f}, Change {change:.2f}%, Alloc £{alloc}"

        rows.append([
            None, name, symbol, round(score, 2), strike, breakout_time,
            round(price, 4), f"{sl_pct:.2f}% (£{alloc * sl_pct/100:.2f}) ({sl_price:.2f})",
            f"{tp1_pct:.2f}% (£{gain_pot:.2f}) ({tp1_price:.2f})",
            "0.00%", f"{sl_pct:.2f}% / {tp1_pct:.2f}%", f"£{alloc}",
            f"{tp1_pct:.2f}% / £{gain_pot}", trend, go, reasoning
        ])

    df = pd.DataFrame(rows, columns=[
        "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
        "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %","Distance to SL / TP (%)",
        "AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
    ])
    df["Rank"] = range(1, len(df) + 1)
    return df.sort_values("Breakout Score", ascending=False).head(5)

# ==============================
# Streamlit Layout
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner")

st.sidebar.header("Settings")
pot = st.sidebar.number_input("Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10)
currency = st.sidebar.selectbox("Currency", ["GBP", "USD"])
refresh = st.sidebar.selectbox("Refresh Interval", [30, 60, 300], index=1)

tabs = st.tabs(["Live Crypto", "Live Stocks", "Simulation Crypto", "Simulation Stocks", "Your Tracker", "AI Simulated Tracker"])

# --- Live Crypto
with tabs[0]:
    st.subheader("Live Crypto Breakouts")
    crypto = fetch_crypto_data()
    if not crypto.empty:
        table = calc_breakout_table(crypto, pot, currency)
        st.dataframe(table, use_container_width=True)
        st.caption(f"Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC")

# --- Live Stocks
with tabs[1]:
    st.subheader("Live Stock Breakouts")
    stocks = fetch_stock_data()
    table = calc_breakout_table(stocks, pot, currency)
    st.dataframe(table, use_container_width=True)

# --- Simulation Tabs
with tabs[2]:
    st.subheader("Simulation Crypto")
    st.info("Historical backtest not yet implemented — placeholder.")

with tabs[3]:
    st.subheader("Simulation Stocks")
    st.info("Historical backtest not yet implemented — placeholder.")

# --- Your Tracker
with tabs[4]:
    st.subheader("Your Tracker (Manual Portfolio)")
    portfolio = load_portfolio()

    with st.form("add_position", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol")
        with col2:
            entry = st.number_input("Entry Price", min_value=0.0, value=0.0)
        with col3:
            size = st.number_input("Size (£)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Add Position")
        if submitted and symbol and entry > 0 and size > 0:
            portfolio.append({"symbol": symbol.upper(), "entry": entry, "size": size})
            save_portfolio(portfolio)
            st.success(f"Added {symbol.upper()}")

    # Display portfolio
    if portfolio:
        rows = []
        crypto = fetch_crypto_data()
        for pos in portfolio:
            sym = pos["symbol"]
            entry = pos["entry"]
            size = pos["size"]
            price_row = crypto[crypto["symbol"].str.upper() == sym]
            if not price_row.empty:
                current = float(price_row.iloc[0]["priceUsd"])
            else:
                current = entry
            value = (current / entry) * size
            pl = value - size
            pl_pct = (pl / size) * 100
            sl = entry * 0.95
            tp1 = entry * 1.3
            rows.append([sym, entry, round(current,4), size, round(value,2), round(pl,2), round(pl_pct,2), sl, tp1, f"{(tp1-current)/current*100:.2f}%", "Open"])
        df = pd.DataFrame(rows, columns=["Symbol","Entry","Current","Size (£)","Value","P/L £","P/L %","SL Price","TP1 Price","% to TP1","Status"])
        st.dataframe(df, use_container_width=True)

        if st.button("Clear Portfolio"):
            portfolio = []
            save_portfolio(portfolio)
            st.warning("Portfolio cleared!")

# --- AI Simulated Tracker
with tabs[5]:
    st.subheader("AI Simulated Tracker")
    st.info("This will display AI’s simulated holdings (not implemented fully yet).")

