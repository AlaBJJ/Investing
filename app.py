# ==============================
# AI Breakout Scanner (Crypto + Stocks) — Enhanced & Fixed
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import yfinance as yf
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from pandas.io.formats.style import Styler  # safe type hint

# ==============================
# Config
# ==============================
REFRESH_INTERVAL = 60
CAPITAL_BASE = 1000
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ~2.98% round trip

# Tighter SL/TP parameters (same table outputs)
ATR_SL_MULTIPLIER = 1.0
MIN_SL_RATIO = 0.015
ATR_TP_MULTIPLIER = 2.5
MIN_TP_RATIO = 0.03

# Sentiment wordlists (simple heuristic)
POSITIVE_WORDS = {
    "surge", "rise", "rises", "rising", "soar", "soars", "gain", "gains",
    "gaining", "increase", "increases", "increased", "up", "bull", "bullish",
    "rally", "rallies", "boom", "booms", "record", "records"
}
NEGATIVE_WORDS = {
    "fall", "falls", "falling", "drop", "drops", "decline", "declines",
    "decrease", "decreases", "down", "bear", "bearish", "plunge", "plunges",
    "crash", "crashes", "slump", "slumps"
}

# Columns to display (kept identical across tabs)
display_cols = [
    "Rank","Name","Symbol","Breakout Score","⚡ Strike Window","Pred. Breakout (hh:mm)",
    "Entry Price (USD/GBP)","SL % / £ (Price)","TP1 % / £ (Price)","Trigger %",
    "Distance to SL / TP (%)","AI Alloc. (£)","Gain Pot. % / £","Trend","Go/No-Go","AI Reasoning"
]

# ==============================
# Sidebar API Settings
# ==============================
st.set_page_config(layout="wide", page_title="AI Breakout Scanner (Enhanced)")
st.sidebar.header("API Settings")

apis_selected = st.sidebar.multiselect(
    "Select APIs to include in fusion:",
    ["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"],
    default=["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"]
)

if "CMC_API_KEY" not in st.session_state:
    st.session_state["CMC_API_KEY"] = ""
if "CC_API_KEY" not in st.session_state:
    st.session_state["CC_API_KEY"] = ""

st.session_state["CMC_API_KEY"] = st.sidebar.text_input(
    "CoinMarketCap API Key", st.session_state["CMC_API_KEY"], type="password")
st.session_state["CC_API_KEY"] = st.sidebar.text_input(
    "CryptoCompare API Key", st.session_state["CC_API_KEY"], type="password")

# ==============================
# API Fetchers (Free tiers)
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
        url = "https://api.coingecko.com/api/v3/coins/markets"
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
        url = "https://min-api.cryptocompare.com/data/top/mktcapfull"
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
# Extra Free Signals/Indicators
# ==============================
def fetch_fear_greed() -> Tuple[float, str]:
    """
    Fetch Crypto Fear & Greed Index (Alternative.me free JSON).
    Returns (normalized_score_0_to_1, label). Fails gracefully to (0.5, "Neutral").
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/", timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        data = (payload.get("data") or payload.get("Data") or [])
        if not data:
            return 0.5, "Neutral"
        value = float(data[0].get("value", 50))
        classification = data[0].get("value_classification", "Neutral")
        return value / 100.0, classification
    except:
        return 0.5, "Neutral"

def fetch_trending_coins() -> set:
    """
    CoinGecko trending endpoint (no key). Returns a set of upper-case symbols.
    """
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("coins", [])
        out = set()
        for c in data:
            sym = c.get("item", {}).get("symbol")
            if sym:
                out.add(sym.upper())
        return out
    except:
        return set()

def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if np.isfinite(val) else None

def compute_macd(closes: pd.Series, fast=12, slow=26, signal=9) -> Tuple[Optional[float], Optional[float]]:
    if closes is None or len(closes) < slow + signal + 5:
        return None, None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_val = float(macd.iloc[-1])
    signal_val = float(signal_line.iloc[-1])
    if not np.isfinite(macd_val) or not np.isfinite(signal_val):
        return None, None
    return macd_val, signal_val

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
# Breakout Model (Refined; same table schema)
# ==============================
def calc_breakout_table(data, investment_pot=CAPITAL_BASE):
    rows = []
    now = datetime.now(timezone.utc)
    total_mcap = data["mcap"].sum() if "mcap" in data and data["mcap"].sum() else 1

    # One-time context fetches (free)
    fg_score, fg_label = fetch_fear_greed()     # 0..1
    trending = fetch_trending_coins()           # set of symbols

    for _, row in data.iterrows():
        name, symbol = row.get("name"), row.get("symbol")
        price = row.get("price") or 0.0
        change = row.get("change") or 0.0
        volume = row.get("volume") or 0.0
        mcap = row.get("mcap") or 0.0

        # Base factors
        vol_factor = min(abs(change) / 10, 1.0)
        vol_score = vol_factor * 100
        liq_score = (np.log1p(volume) / 25) * 100
        trend_score = (np.tanh(change / 5) + 1) * 50
        base_score = (0.4 * vol_score + 0.3 * liq_score + 0.2 * trend_score + 0.1 * (100 - abs(vol_score - liq_score)))
        base_score = float(np.clip(base_score, 50, 100))

        # Free news sentiment (CryptoCompare; neutral if no key or failure)
        def fetch_news_sentiment(sym: str, max_articles: int = 20) -> float:
            api_key = st.session_state.get("CC_API_KEY", "")
            if not api_key:
                return 0.0
            try:
                url = "https://min-api.cryptocompare.com/data/v2/news/"
                params = {"lang": "EN", "categories": sym, "api_key": api_key}
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                arts = resp.json().get("Data", [])[:max_articles]
                if not arts:
                    return 0.0
                pos = sum(any(w in (a.get("title","")+a.get("body","")).lower() for w in POSITIVE_WORDS) for a in arts)
                neg = sum(any(w in (a.get("title","")+a.get("body","")).lower() for w in NEGATIVE_WORDS) for a in arts)
                total = pos + neg
                return (pos - neg) / total if total else 0.0
           
