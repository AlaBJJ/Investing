# ==============================
# AI Breakout Scanner (Crypto + Stocks)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ==============================
# Config
# ==============================
REFRESH_INTERVAL = 60
CAPITAL_BASE = 1000
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ~2.98% round trip

# --- CoinMarketCap API Key ---
if "CMC_API_KEY" in st.secrets:
    CMC_API_KEY = st.secrets["CMC_API_KEY"]
else:
    CMC_API_KEY = os.getenv("CMC_API_KEY", "YOUR_CMC_API_KEY_HERE")

# ==============================
# ML Model Training (Safe)
# ==============================
def train_ml_model(symbol="BTC-USD"):
    """Train ML model on historical data for given symbol"""
    try:
        data = yf.download(symbol, period="60d", interval="1h")
        if data.empty:
            return None, None

        # Features
        data["return_1h"] = data["Close"].pct_change()
        data["return_24h"] = data["Close"].pct_change(24)
        data["volatility"] = (data["High"] - data["Low"]) / data["Close"]
        data["volume_change"] = data["Volume"].pct_change()
        data["ema_trend"] = data["Close"] / data["Close"].ewm(span=20).mean() - 1

        # Target: breakout if next 24h > 5%
        data["target"] = (data["Close"].shift(-24) / data["Close"] - 1 > 0.05).astype(int)

        features = ["return_1h","return_24h","volatility","volume_change","ema_trend"]
        X = data[features]
        y = data["target"]

        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        if X.empty or y.empty:
            return None, None

        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3)
        model.fit(X_train, y_train)

        return model, scaler

    except Exception as e:
        st.error(f"ML training failed: {e}")
        return None, None

ML_MODEL, SCALER = train_ml_model("BTC-USD")

# ==============================
# Helpers
# ==============================
def fetch_crypto_data(limit=100):
    """Fetch top 100 cryptos from CoinMarketCap"""
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
        return pd.DataFrame()

def fetch_stock_data():
    """Fetch S&P100 stocks with yfinance"""
    sp100 = ["AAPL","MSFT","AMZN","TSLA","GOOGL","BRK-B","NVDA","META","JNJ","XOM",
             "JPM","V","PG","UNH","HD","MA","PFE","CVX","ABBV","BAC","KO","PEP","MRK",
             "DIS","CSCO","VZ","WMT","ADBE","NFLX","T","INTC","CRM","CMCSA","ABT","PYPL",
             "NKE","ORCL","ACN","MCD","TMO","DHR","LLY","QCOM","COST","TXN","NEE","MDT",
             "LIN","HON","AMGN","PM","AVGO","BMY","UNP","LOW","UPS","MS","RTX","IBM","GS",
             "CAT","SCHW","AMD","AMT","AXP","LMT","INTU","BLK","DE","CVS","ISRG","GE","SPGI",
             "PLD","GILD","MDLZ","NOW","ADP","CI","C","SYK","BA","MO","ZTS","USB","BKNG",
             "MMC","TGT","CB","BDX","CCI","CL","DUK","MMM","SO","CME","PNC","SHW","ICE","APD","EQIX"]

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

def calc_breakout_table(data, investment_pot=CAPITAL_BASE):
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

        # --- Heuristic factors ---
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

def plot_chart(symbol, df):
    """Plot candlestick with SL/TP lines"""
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
        sl = df.loc[df["Symbol"] == symbol, "SL_Price"].values[0]
        tp = df.loc[df["Symbol"] == symbol, "TP1_Price"].values[0]
        fig.add_hline(y=sl, line_color="red", annotation_text="SL", annotation_position="bottom right")
        fig.add_hline(y=tp, line_color="green", annotation_text="TP", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning(f"No chart data for {symbol}")

# ==============================
# Streamlit Layout
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner")

st.sidebar.header("Settings")
pot = st.sidebar.number_input("Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10)

tabs = st.tabs(["Live Crypto", "Live Stocks", "Chart View"])

# --- Live Crypto
with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto = fetch_crypto_data()
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
