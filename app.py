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
from pandas.io.formats.style import Styler  # safe type hint import

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
    macd_val = float(macd.iloc[-1]); signal_val = float(signal_line.iloc[-1])
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
            except:
                return 0.0

        sentiment_adj = fetch_news_sentiment(symbol) * 5.0  # -5 .. +5

        # Fear & Greed adjustment: center 0.5; scale to ±4 points
        fg_adj = (fg_score - 0.5) * 8.0  # -4 .. +4

        # Trending boost if CoinGecko lists the symbol (small nudge)
        trending_adj = 2.0 if isinstance(symbol, str) and symbol.upper() in trending else 0.0

        # Technical confirmation (daily history)
        rsi_adj = 0.0
        macd_adj = 0.0
        try:
            t1 = f"{symbol}-USD" if isinstance(symbol, str) else symbol
            hist = yf.download(t1, period="3mo", interval="1d", progress=False)
            if hist is None or hist.empty:
                hist = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if hist is not None and not hist.empty:
                closes = hist["Close"].dropna()
                rsi_val = compute_rsi(closes, period=14)
                macd_val, macd_sig = compute_macd(closes)

                if rsi_val is not None:
                    if rsi_val < 30:
                        rsi_adj = 3.0
                    elif rsi_val > 70:
                        rsi_adj = -2.0

                if macd_val is not None and macd_sig is not None:
                    if macd_val > macd_sig:
                        macd_adj = 2.0
                    elif macd_val < macd_sig:
                        macd_adj = -2.0
        except Exception:
            # Fail gracefully if Yahoo Finance or indicator calc breaks
            pass

        score = float(np.clip(base_score + sentiment_adj + fg_adj + trending_adj + rsi_adj + macd_adj, 50, 100))

        # Risk levels (same columns; tightened bounds)
        atr = price * abs(change) / 100 / 2
        sl_price = price - max(ATR_SL_MULTIPLIER * atr, MIN_SL_RATIO * price)
        tp1_price = price + max(ATR_TP_MULTIPLIER * atr, MIN_TP_RATIO * price)
        sl_pct = (sl_price - price) / price * 100 if price else 0.0
        tp1_pct = (tp1_price - price) / price * 100 - REVOLUT_FEES * 100 if price else 0.0
        rr = abs(tp1_pct / sl_pct) if sl_pct != 0 else 0.0

        # Kelly sizing (advisory)
        p = score / 100
        b = rr
        kelly = max((p * (b + 1) - 1) / b, 0) if b > 0 else 0
        alloc = round(min(investment_pot * kelly, investment_pot), 2)
        gain_pot = round(alloc * tp1_pct / 100, 2)

        strike = "Yes" if score >= 85 else "No"
        breakout_time = (now + timedelta(minutes=np.random.randint(30, 180))).strftime("%H:%M")
        trend = "↑" if change > 0 else ("↓" if change < 0 else "↔")
        go = "Go" if score >= 85 and trend == "↑" and rr > 1.5 else "No-Go"

        reasoning = (
            f"Score {score:.1f}, Vol {change:.2f}%, Vol24h ${volume/1e6:.1f}M, "
            f"MCAP ${mcap/1e9:.1f}B, R/R {rr:.2f}, Kelly £{alloc}; "
            f"FG {int(fg_score*100)}({fg_label}) {'+2' if trending_adj else ''} "
            f"{'RSI<30 +3' if rsi_adj==3.0 else ('RSI>70 -2' if rsi_adj==-2.0 else '')} "
            f"{'MACD+2' if macd_adj==2.0 else ('MACD-2' if macd_adj==-2.0 else '')}"
        ).strip()

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
# Chart Plotting (with guards)
# ==============================
def plot_chart(symbol, df):
    try:
        t1 = f"{symbol}-USD"
        hist = yf.download(t1, period="5d", interval="1h", progress=False)
        if hist is None or hist.empty:
            hist = yf.download(symbol, period="5d", interval="1h", progress=False)
        if hist is None or hist.empty:
            st.warning(f"No chart data for {symbol}")
            return

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
# Conditional Formatting
# ==============================
def style_table(df: pd.DataFrame) -> Styler:
    def highlight_go(row: pd.Series) -> List[str]:
        return ["background-color: #e6ffe6" if row.get("Go/No-Go") == "Go" else "" for _ in row]
    return df.style.apply(highlight_go, axis=1)

# ==============================
# Row selection helper (click row -> chart)
# ==============================
def pick_symbol_interactive(df: pd.DataFrame, key: str) -> Optional[str]:
    """Try selection-enabled dataframe; fallback to symbol buttons."""
    selected_symbol = None
    try:
        _ = st.dataframe(
            style_table(df[display_cols]),
            use_container_width=True,
            selection_mode="single",  # newer Streamlit
            on_select="rerun",        # newer Streamlit
            key=f"dfsel_{key}"
        )
        state = st.session_state.get(f"dfsel_{key}-selection")
        if state and "rows" in state and state["rows"]:
            idx = state["rows"][0]
            selected_symbol = df.iloc[idx]["Symbol"]
    except TypeError:
        st.dataframe(style_table(df[display_cols]), use_container_width=True)
        with st.expander("Click a symbol to open its chart"):
            cols = st.columns(5)
            for i, sym in enumerate(df["Symbol"].tolist()):
                if cols[i % 5].button(sym, key=f"symbtn_{key}_{sym}"):
                    selected_symbol = sym
    return selected_symbol

# ==============================
# Portfolio storage
# ==============================
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = pd.DataFrame(
        columns=["Symbol", "Type", "Quantity", "Entry_Price"]
    )

def price_lookup(symbol: str, crypto_prices: dict) -> Optional[float]:
    # Try crypto map first
    if symbol in crypto_prices:
        return float(crypto_prices[symbol])
    # Fallback to Yahoo for stocks (or crypto if not in map)
    try:
        t1 = f"{symbol}-USD"
        hist = yf.download(t1, period="1d", interval="1m", progress=False)
        if hist is None or hist.empty:
            hist = yf.download(symbol, period="1d", interval="1m", progress=False)
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except:
        pass
    return None

# ==============================
# Layout
# ==============================
st.sidebar.header("Settings")
pot = st.sidebar.number_input("Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10)

tabs = st.tabs(["Live Crypto", "Live Stocks", "eToro Crypto & Stocks", "Chart View", "My Portfolio"])

# --- Crypto
with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto_data = fuse_data(100)
    if not crypto_data.empty:
        crypto_table = calc_breakout_table(crypto_data, pot)
        selected_crypto = pick_symbol_interactive(crypto_table, key="crypto")
        if selected_crypto:
            st.markdown(f"**Chart: {selected_crypto}**")
            plot_chart(selected_crypto, crypto_table)
    else:
        st.warning("No cryptocurrency data available. Check your API selections/keys.")

# --- Stocks
with tabs[1]:
    st.subheader("Live Stock Breakouts (S&P100)")
    sp100 = ["AAPL","MSFT","AMZN","TSLA","GOOGL","NVDA","META","JPM","V","PG"]
    tickers = yf.download(sp100, period="1d", interval="1h", progress=False, threads=True)
    if tickers is None or ("Close" not in tickers):
        st.warning("No stock data available from Yahoo Finance.")
        stock_table = pd.DataFrame(columns=[c for c in display_cols] + ["SL_Price","TP1_Price"])
    else:
        latest = tickers["Close"].iloc[-1]
        stocks = pd.DataFrame([{
            "symbol": s,
            "name": s,
            "price": float(latest.get(s, np.nan)),
            "change": 0.0,
            "volume": 1.0,
            "mcap": 1.0
        } for s in sp100])
        stock_table = calc_breakout_table(stocks, pot)

    if not stock_table.empty:
        selected_stock = pick_symbol_interactive(stock_table, key="stocks")
        if selected_stock:
            st.markdown(f"**Chart: {selected_stock}**")
            plot_chart(selected_stock, stock_table)

# --- eToro (same tables, no proprietary data)
with tabs[2]:
    st.subheader("eToro Crypto & Stocks")
    st.markdown("### Crypto (Top 100)")
    if 'crypto_table' in locals() and not crypto_table.empty:
        _ = pick_symbol_interactive(crypto_table, key="etoro_crypto")
    else:
        st.warning("No crypto data available.")
    st.markdown("### Stocks (S&P100)")
    if 'stock_table' in locals() and not stock_table.empty:
        _ = pick_symbol_interactive(stock_table, key="etoro_stocks")
    else:
        st.warning("No stock data available.")

# --- Chart View (info)
with tabs[3]:
    st.info("Click a row in any table (or use the symbol buttons in the expander) to load its candlestick chart with AI Entry/SL/TP overlays.")

# --- My Portfolio (manual entries with live P/L)
with tabs[4]:
    st.subheader("My Portfolio (Manual Entries with Live P/L)")
    st.caption("Add rows for your holdings. Type: 'Crypto' or 'Stock'.")

    port_df = st.data_editor(
        st.session_state["portfolio"],
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor"
    )
    st.session_state["portfolio"] = port_df

    if not port_df.empty:
        crypto_price_map = {}
        if 'crypto_data' in locals() and not crypto_data.empty:
            crypto_price_map = {r["symbol"]: r["price"] for _, r in crypto_data.iterrows() if pd.notnull(r["price"])}

        rows = []
        total_cost = 0.0
        total_value = 0.0

        for _, r in port_df.iterrows():
            sym = str(r.get("Symbol") or "").upper().strip()
            typ = str(r.get("Type") or "")
            qty = float(r.get("Quantity") or 0)
            entry = float(r.get("Entry_Price") or 0)

            if not sym or qty <= 0 or entry <= 0:
                continue

            live = price_lookup(sym, crypto_price_map)
            pl = None; pl_pct = None; val = None; cost = qty * entry
            total_cost += cost

            if live is not None:
                val = qty * live
                total_value += val
                pl = (live - entry) * qty
                if entry > 0:
                    pl_pct = (live - entry) / entry * 100.0

            rows.append({
                "Symbol": sym,
                "Type": typ,
                "Quantity": qty,
                "Entry_Price": entry,
                "Live_Price": round(live, 6) if live is not None else None,
                "Value": round(val, 2) if val is not None else None,
                "Cost": round(cost, 2),
                "P/L (£)": round(pl, 2) if pl is not None else None,
                "P/L (%)": f"{pl_pct:.2f}%" if pl_pct is not None else None
            })

        if rows:
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.markdown("---")
            st.metric("Total Cost (£)", f"{total_cost:,.2f}")
            if total_value:
                st.metric("Total Value (£)", f"{total_value:,.2f}")
                st.metric("Total P/L (£)", f"{(total_value-total_cost):,.2f}")
        else:
            st.info("Add your holdings above to see live P/L.")
    else:
        st.info("Add your holdings above to see live P/L.")
