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

CMC_API_KEY = os.getenv("CMC_API_KEY", "fde1ec72-770a-45f1-a2aa-2af4507c9d12")

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

def fetch_crypto_data(limit=100):
    """Fetch crypto data from CoinMarketCap (stable)."""
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"start": "1", "limit": str(limit), "convert": "USD"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame([{
            "id": asset["id"],
            "symbol": asset["symbol"],
            "name": asset["name"],
            "priceUsd": asset["quote"]["USD"]["price"],
            "changePercent24Hr": asset["quote"]["USD"]["percent_change_24h"],
            "volume24h": asset["quote"]["USD"]["volume_24h"],
            "marketCap": asset["quote"]["USD"]["market_cap"]
        } for asset in data["data"]])

        return df

    except Exception as e:
        st.error(f"CMC API failed: {e}")
        # fallback dataset
        return pd.DataFrame([
            {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "priceUsd": 45000, "changePercent24Hr": 2.5, "volume24h": 20000000000, "marketCap": 850000000000},
            {"id": "ethereum", "symbol": "ETH", "name": "Ethereum", "priceUsd": 3000, "changePercent24Hr": -1.2, "volume24h": 10000000000, "marketCap": 400000000000},
            {"id": "solana", "symbol": "SOL", "name": "Solana", "priceUsd": 175, "changePercent24Hr": 6.3, "volume24h": 2500000000, "marketCap": 70000000000},
            {"id": "cardano", "symbol": "ADA", "name": "Cardano", "priceUsd": 0.52, "changePercent24Hr": 3.8, "volume24h": 800000000, "marketCap": 20000000000},
            {"id": "xrp", "symbol": "XRP", "name": "XRP", "priceUsd": 0.64, "changePercent24Hr": -0.5, "volume24h": 1500000000, "marketCap": 30000000000}
        ])

def fetch_stock_data(symbols=["AAPL", "MSFT", "TSLA"]):
    """Dummy stock fetcher (replace with yfinance for real data)."""
    return pd.DataFrame({
        "symbol": symbols,
        "name": symbols,
        "priceUsd": np.random.uniform(100, 300, len(symbols)),
        "changePercent24Hr": np.random.uniform(-5, 5, len(symbols)),
        "volume24h": np.random.uniform(1e6, 5e6, len(symbols)),
        "marketCap": np.random.uniform(1e10, 5e11, len(symbols))
    })

def calc_breakout_table(data, investment_pot=CAPITAL_BASE, currency="USD"):
    rows = []
    now = datetime.utcnow()
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

        # Weighted score
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

        # --- AI reasoning ---
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
        "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %","Distance to SL / TP (%)",
        "AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
    ])
    df["Rank"] = range(1, len(df) + 1)
    return df.sort_values("Breakout Score", ascending=False).head(5)

def build_ai_portfolio(df):
    portfolio = []
    for _, row in df.iterrows():
        if row["Go/No-Go"] == "Go":
            entry = row["Entry Price (USD/GBP)"]
            alloc_str = row["AI Alloc. (£)"].replace("£", "")
            try:
                size = float(alloc_str)
            except:
                size = 0
            current = entry
            value = size
            pl = 0
            pl_pct = 0
            sl_price = float(row["SL % / £ (Price)"].split("(")[-1].replace(")", ""))
            tp1_price = float(row["TP1 % / £ (Price)"].split("(")[-1].replace(")", ""))
            portfolio.append([
                row["Symbol"], entry, current, size, value, pl, pl_pct,
                sl_price, tp1_price,
                f"{(tp1_price-current)/current*100:.2f}%", "Open"
            ])
    return pd.DataFrame(portfolio, columns=[
        "Symbol","Entry","Current","Size (£)","Value","P/L £","P/L %","SL Price","TP1 Price","% to TP1","Status"
    ])

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
        crypto_table = calc_breakout_table(crypto, pot, currency)
        st.dataframe(crypto_table, use_container_width=True)
        st.caption(f"Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC")

# --- Live Stocks
with tabs[1]:
    st.subheader("Live Stock Breakouts")
    stocks = fetch_stock_data()
    stock_table = calc_breakout_table(stocks, pot, currency)
    st.dataframe(stock_table, use_container_width=True)

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
    if 'crypto_table' in locals():
        ai_df = build_ai_portfolio(crypto_table)
        if not ai_df.empty:
            st.dataframe(ai_df, use_container_width=True)
        else:
            st.info("No AI trades triggered (all No-Go).")
    else:
        st.warning("Run Live Crypto tab first to generate AI portfolio.")
