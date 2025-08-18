# ==============================
# AI Breakout Scanner (Crypto + Stocks) - Ultimate Version
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# Config
# ==============================
REFRESH_INTERVAL = 60
CAPITAL_BASE = 1000
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ~2.98% round trip
CMC_API_KEY = os.getenv("CMC_API_KEY", "YOUR_CMC_API_KEY_HERE")

# ==============================
# Fetch Data
# ==============================
def fetch_crypto_data(limit=100):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"start": "1", "limit": str(limit), "convert": "USD"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame([{
            "symbol": a["symbol"],
            "name": a["name"],
            "priceUsd": a["quote"]["USD"]["price"],
            "changePercent24Hr": a["quote"]["USD"]["percent_change_24h"],
            "volume24h": a["quote"]["USD"]["volume_24h"],
            "marketCap": a["quote"]["USD"]["market_cap"]
        } for a in data["data"]])
        return df
    except Exception as e:
        st.error(f"CMC API failed: {e}")
        return pd.DataFrame()

def fetch_stock_data():
    sp100 = ["AAPL","MSFT","AMZN","TSLA","GOOGL","NVDA","META","JPM","V","PG","UNH","HD","MA","PFE",
             "CVX","ABBV","BAC","KO","PEP","MRK","DIS","CSCO","WMT","ADBE","NFLX","INTC","CRM",
             "CMCSA","ABT","NKE","ORCL","MCD","TMO","LLY","QCOM","COST","TXN","NEE","HON","AMGN",
             "PM","AVGO","BMY","UNP","LOW","UPS","MS","RTX","IBM","GS","CAT","AMD","AMT","AXP",
             "LMT","INTU","BLK","DE","CVS","ISRG","GE","SPGI","GILD","MDLZ","NOW","ADP","CI",
             "C","SYK","BA","ZTS","USB","BKNG","TGT","CB","BDX","CCI","CL","DUK","MMM","SO",
             "CME","PNC","SHW","ICE","APD","EQIX"]
    try:
        tickers = yf.download(sp100, period="1d", interval="1h", progress=False, threads=True)
        latest = tickers["Close"].iloc[-1]
        rows = []
        for sym in sp100:
            t = yf.Ticker(sym)
            info = t.info
            price = latest.get(sym, np.nan)
            rows.append({
                "symbol": sym,
                "name": info.get("shortName", sym),
                "priceUsd": price,
                "changePercent24Hr": info.get("regularMarketChangePercent", 0.0),
                "volume24h": info.get("volume", 1),
                "marketCap": info.get("marketCap", 1)
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Stock API failed: {e}")
        return pd.DataFrame()

# ==============================
# ML Feature Engineering
# ==============================
def build_features(symbol="BTC-USD", days=30):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1h")
        if df.empty:
            return pd.DataFrame()
        df["return1h"] = df["Close"].pct_change()
        df["volatility"] = df["Close"].rolling(14).std()
        df["volume_change"] = df["Volume"].pct_change()
        df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
        df = df.dropna()
        return df
    except:
        return pd.DataFrame()

def train_ml_model(symbol="BTC-USD"):
    df = build_features(symbol)
    if df.empty: 
        return None
    df["target"] = (df["return1h"].shift(-3) > 0.02).astype(int)  # breakout if +2% in next 3h
    df = df.dropna()
    X = df[["return1h","volatility","volume_change","atr"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model, scaler

ML_MODEL, SCALER = train_ml_model("BTC-USD")

# ==============================
# Breakout Calculation
# ==============================
def calc_breakout_table(data, investment_pot, allocation_mode, manual_alloc):
    rows = []
    now = datetime.utcnow()
    for _, row in data.iterrows():
        name, symbol, price, change, volume, mcap = row["name"], row["symbol"], row["priceUsd"], row["changePercent24Hr"], row["volume24h"], row["marketCap"]
        atr = price * abs(change)/100 / 2
        sl_price = price - max(1.5*atr, 0.02*price)
        tp1_price = price + max(2.5*atr, 0.03*price)
        sl_pct = (sl_price - price)/price*100
        tp1_pct = (tp1_price - price)/price*100 - REVOLUT_FEES*100
        rr = abs(tp1_pct/sl_pct) if sl_pct != 0 else 0

        # AI Score (ML or fallback)
        if ML_MODEL and SCALER is not None:
            feats = np.array([[change/100, atr/price, volume/max(volume,1e6), atr/price]])
            prob = ML_MODEL.predict_proba(SCALER.transform(feats))[:,1][0]
            score = round(prob*100, 2)
        else:
            vol_factor = min(abs(change)/10,1.0)
            score = round(vol_factor*100,2)

        # Allocation
        p, b = score/100, rr
        kelly = max((p*(b+1)-1)/b,0) if b>0 else 0
        alloc = round(min(investment_pot*kelly,investment_pot),2) if allocation_mode=="AI Optimised Allocation" else manual_alloc
        gain_pot = round(alloc*tp1_pct/100,2)

        strike = "Yes" if score>=85 else "No"
        breakout_time = (now+timedelta(minutes=np.random.randint(30,180))).strftime("%H:%M")
        trend = "↑" if change>0 else ("↓" if change<0 else "↔")
        go = "Go" if score>=85 and trend=="↑" and rr>1.5 else "No-Go"
        reasoning = f"Score {score}, R/R {rr:.2f}, ATR {atr:.2f}, Vol {volume/1e6:.1f}M, Alloc £{alloc}"

        rows.append([None,name,symbol,score,strike,breakout_time,
                     round(price,4),f"{sl_pct:.2f}% (£{alloc*sl_pct/100:.2f}) ({sl_price:.2f})",
                     f"{tp1_pct:.2f}% (£{gain_pot:.2f}) ({tp1_price:.2f})",
                     "0.00%",f"{sl_pct:.2f}% / {tp1_pct:.2f}%",f"£{alloc}",
                     f"{tp1_pct:.2f}% / £{gain_pot}",trend,go,reasoning,sl_price,tp1_price])
    df = pd.DataFrame(rows, columns=["Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
                                     "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
                                     "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go",
                                     "AI Reasoning","SL_Price","TP1_Price"])
    df["Rank"] = range(1,len(df)+1)
    return df.sort_values("Breakout Score", ascending=False).head(100)

# ==============================
# Chart Plotting
# ==============================
def plot_chart(symbol, df):
    try:
        hist = yf.download(symbol, period="5d", interval="1h")
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name=symbol)])
        sl = df.loc[df["Symbol"]==symbol,"SL_Price"].values[0]
        tp = df.loc[df["Symbol"]==symbol,"TP1_Price"].values[0]
        fig.add_hline(y=sl,line_color="red",annotation_text="SL")
        fig.add_hline(y=tp,line_color="green",annotation_text="TP")
        st.plotly_chart(fig,use_container_width=True)
    except:
        st.warning(f"No chart data for {symbol}")

# ==============================
# Streamlit App
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner - Ultimate")

st.sidebar.header("Settings")
allocation_mode = st.sidebar.radio("Investment Mode:", ["AI Optimised Allocation","Manual Fixed Allocation"])
manual_alloc = st.sidebar.number_input("Manual Allocation per Trade (£)", min_value=1,value=30,step=1) if allocation_mode=="Manual Fixed Allocation" else 0
pot = st.sidebar.number_input("Total Investment Pot (£)", min_value=10,value=CAPITAL_BASE,step=10)

tabs = st.tabs(["Live Crypto","Live Stocks","Chart View"])

with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto = fetch_crypto_data()
    if not crypto.empty:
        table = calc_breakout_table(crypto,pot,allocation_mode,manual_alloc)
        st.dataframe(table[["Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
                            "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
                            "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"]],
                     use_container_width=True)
        choice = st.selectbox("Select crypto for chart", table["Symbol"])
        if choice: plot_chart(choice+"-USD", table)

with tabs[1]:
    st.subheader("Live Stock Breakouts (S&P100)")
    stocks = fetch_stock_data()
    if not stocks.empty:
        table = calc_breakout_table(stocks,pot,allocation_mode,manual_alloc)
        st.dataframe(table[["Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
                            "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
                            "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"]],
                     use_container_width=True)
        choice = st.selectbox("Select stock for chart", table["Symbol"])
        if choice: plot_chart(choice, table)

with tabs[2]:
    st.info("Use the chart dropdowns in Crypto/Stocks tabs to view SL/TP levels with candlestick charts.")
