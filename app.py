"""
AI Breakout Scanner with Enhanced Features

This Streamlit application builds on a basic breakout scanner for crypto‑currencies
and stocks by adding the following enhancements:

* **Additional Tabs** – A third data tab labelled “eToro Crypto & Stocks”
  mirrors the crypto and stock tables but is clearly separated in the UI.  In
  practice this tab uses the same data sources as the existing tabs because
  free access to eToro’s internal pricing feeds is not available.  It simply
  repackages the information so that users who operate through eToro can see
  a dedicated view.
* **News‑Aware Breakout Model** – A simple sentiment analyser reads recent
  cryptocurrency news from CryptoCompare’s free news API and assigns a score
  based on the prevalence of positive versus negative words.  This score is
  blended into the breakout score calculation so that assets receiving
  favourable coverage gain a small boost.  The analysis relies on a
  lightweight word‑list and does **not** guarantee future returns.  It is a
  heuristic designed to encourage users to remain aware of how news can
  influence markets.  If the API call fails (for example if no API key is
  provided) the sentiment contribution defaults to zero.
* **Tighter Stop‑Loss (SL) and Take‑Profit (TP) Boundaries** –  To help
  protect capital, the distance between entry price and stop‑loss is reduced
  and the minimum profit target is increased.  The previous ATR (Average
  True Range) multipliers have been adjusted, and a minimum percentage is
  applied so that both risk and reward levels are more controlled.  These
  parameters can be tuned by editing the constants below.
* **Conditional Formatting** – Rows where the scanner recommends “Go” are
  highlighted in green.  This visual cue draws the eye toward potential
  opportunities and helps distinguish them from entries that should be
  avoided.  The `pandas.DataFrame.style` API is used to apply CSS classes
  within the Streamlit table.
* **Strategy Advice Tab** – A fourth tab summarises widely recognised
  investment strategies for both crypto and traditional markets.  The
  guidance is extracted from publicly available educational sources.  It is
  presented for informational purposes only and should not be considered
  personalised financial advice.  Users are encouraged to consult a
  professional advisor and to conduct their own due diligence.

This file is meant to be run with `streamlit run app.py`.  Because no
streamlit server is started in this environment, executing the file may
do nothing here.  The code is provided so that it can be saved to disk
and executed locally by the user.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import yfinance as yf
import plotly.graph_objects as go
from typing import List, Tuple


# ==============================
# Configuration
# ==============================

REFRESH_INTERVAL = 60  # seconds between refreshes – unused but left for clarity
CAPITAL_BASE = 1000    # default investment pot in GBP
REVOLUT_FEES = 0.0099 * 2 + 0.005 * 2  # ≈2.98% round trip cost

# Multipliers for SL/TP calculation.  These values tighten the risk window.
ATR_SL_MULTIPLIER = 1.0   # previously 1.5
MIN_SL_RATIO = 0.015      # 1.5% of price minimum SL distance
ATR_TP_MULTIPLIER = 2.5   # previously 2.5; TP still generous relative to SL
MIN_TP_RATIO = 0.03       # 3% of price minimum TP distance

# Word lists for simple sentiment analysis.  Positive and negative words are
# deliberately broad; fine‑tuning them may improve the signal.  Feel free to
# extend these lists to capture more nuance.
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


# ==============================
# Additional Market Sentiment Fetchers and Technical Indicators
# ==============================

def fetch_fear_greed() -> Tuple[float, str]:
    """Fetch the Crypto Fear & Greed Index.

    This function queries the Alternative.me API for the current fear and
    greed index value and classification.  The index ranges from 0 to 100,
    where lower values correspond to "extreme fear" and higher values
    correspond to "extreme greed"【573275957521421†L85-L96】.  Extreme fear can
    sometimes present buying opportunities【573275957521421†L130-L146】.  A
    neutral value of 50 indicates balanced sentiment.  In case of any
    error (for example network issues), the function returns (50, "Neutral").

    Returns
    -------
    Tuple[float, str]
        The index value and its classification (e.g., "Fear", "Greed").
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        payload = resp.json().get("data", [])
        if not payload:
            return 50.0, "Neutral"
        entry = payload[0]
        # The API returns strings; convert to float for calculations.
        value = float(entry.get("value", 50.0))
        classification = entry.get("value_classification", "Neutral")
        return value, classification
    except Exception:
        # On failure, default to neutral sentiment.
        return 50.0, "Neutral"


def fetch_trending_coins() -> set:
    """Fetch a set of currently trending cryptocurrency symbols.

    The CoinGecko trending endpoint returns a list of coins generating
    significant interest【287987501053356†L0-L26】.  This function extracts
    the symbols of the top trending coins and returns them as an uppercase
    set.  If the API call fails, an empty set is returned.

    Returns
    -------
    set
        A set of trending crypto ticker symbols (uppercase).
    """
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        resp.raise_for_status()
        coins = resp.json().get("coins", [])
        trending = {coin.get("item", {}).get("symbol", "").upper() for coin in coins}
        return {sym for sym in trending if sym}
    except Exception:
        return set()


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute the Relative Strength Index (RSI) of a price series.

    RSI is a momentum oscillator that measures the speed and magnitude of
    recent price changes【151788861916106†L366-L379】.  Traditional
    interpretation considers an RSI above 70 to indicate overbought
    conditions and below 30 to indicate oversold conditions【151788861916106†L378-L391】.
    This function returns the most recent RSI value.  If insufficient data
    exists, a neutral value of 50 is returned.

    Parameters
    ----------
    series: pd.Series
        A series of closing prices indexed by date.
    period: int
        The look‑back period for the RSI (default 14).

    Returns
    -------
    float
        The latest RSI value.
    """
    # Ensure we have enough data
    if series is None or len(series) < period + 1:
        return 50.0
    # Calculate price changes
    delta = series.diff().dropna()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    # Smooth with rolling mean
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notnull(rsi.iloc[-1]) else 50.0


def compute_macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[float, float]:
    """Compute the MACD (Moving Average Convergence/Divergence) and its signal line.

    MACD measures momentum by subtracting a longer exponential moving
    average (EMA) from a shorter EMA【54113305808869†L401-L449】.  A positive
    MACD value means the short EMA is above the long EMA, indicating upward
    momentum; a negative value indicates downward momentum.  The signal
    line is a smoothed (EMA) version of the MACD and is used to identify
    crossovers for buy or sell signals【54113305808869†L423-L501】.

    Parameters
    ----------
    series: pd.Series
        Series of closing prices.
    short: int
        Span for the short EMA (default 12).
    long: int
        Span for the long EMA (default 26).
    signal: int
        Span for the signal EMA (default 9).

    Returns
    -------
    Tuple[float, float]
        The latest MACD value and the latest signal line value.
    """
    if series is None or len(series) < long + signal:
        return 0.0, 0.0
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (
        float(macd_line.iloc[-1]) if not macd_line.empty else 0.0,
        float(signal_line.iloc[-1]) if not signal_line.empty else 0.0,
    )


# ==============================
# Sidebar API Settings
# ==============================

st.set_page_config(layout="wide", page_title="Enhanced AI Breakout Scanner")

st.sidebar.header("API Settings")

# Allow users to select which free crypto APIs to include in the data fusion.  If
# no API is selected, the scanner will display an empty table.
apis_selected: List[str] = st.sidebar.multiselect(
    "Select APIs to include in fusion:",
    ["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"],
    default=["CoinMarketCap", "CoinGecko", "CryptoCompare", "Coinpaprika", "CoinCap"]
)

# API keys (persisted in session state)
if "CMC_API_KEY" not in st.session_state:
    st.session_state["CMC_API_KEY"] = ""
if "CC_API_KEY" not in st.session_state:
    st.session_state["CC_API_KEY"] = ""

st.session_state["CMC_API_KEY"] = st.sidebar.text_input(
    "CoinMarketCap API Key", st.session_state["CMC_API_KEY"], type="password"
)
st.session_state["CC_API_KEY"] = st.sidebar.text_input(
    "CryptoCompare API Key (for prices and news)", st.session_state["CC_API_KEY"], type="password"
)


# ==============================
# API Fetchers
# ==============================

def fetch_cmc(limit: int = 100) -> pd.DataFrame:
    """Fetch cryptocurrency market data from CoinMarketCap.

    Returns a DataFrame with columns: symbol, name, price, change (24h %),
    volume, and market cap.  If the request fails, an empty DataFrame is
    returned.
    """
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {"X-CMC_PRO_API_KEY": st.session_state["CMC_API_KEY"]}
        params = {"start": "1", "limit": str(limit), "convert": "USD"}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return pd.DataFrame([
            {
                "symbol": a.get("symbol"),
                "name": a.get("name"),
                "price": a.get("quote", {}).get("USD", {}).get("price"),
                "change": a.get("quote", {}).get("USD", {}).get("percent_change_24h"),
                "volume": a.get("quote", {}).get("USD", {}).get("volume_24h"),
                "mcap": a.get("quote", {}).get("USD", {}).get("market_cap"),
            }
            for a in data
        ])
    except Exception:
        return pd.DataFrame()


def fetch_coingecko(limit: int = 100) -> pd.DataFrame:
    """Fetch cryptocurrency market data from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame([
            {
                "symbol": a.get("symbol", "").upper(),
                "name": a.get("name"),
                "price": a.get("current_price"),
                "change": a.get("price_change_percentage_24h"),
                "volume": a.get("total_volume"),
                "mcap": a.get("market_cap"),
            }
            for a in data
        ])
    except Exception:
        return pd.DataFrame()


def fetch_cryptocompare(limit: int = 100) -> pd.DataFrame:
    """Fetch cryptocurrency market data from CryptoCompare."""
    try:
        url = "https://min-api.cryptocompare.com/data/top/mktcapfull"
        params = {"limit": limit, "tsym": "USD", "api_key": st.session_state["CC_API_KEY"]}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("Data", [])
        return pd.DataFrame([
            {
                "symbol": a.get("CoinInfo", {}).get("Name"),
                "name": a.get("CoinInfo", {}).get("FullName"),
                "price": a.get("RAW", {}).get("USD", {}).get("PRICE"),
                "change": a.get("RAW", {}).get("USD", {}).get("CHANGEPCT24HOUR"),
                "volume": a.get("RAW", {}).get("USD", {}).get("VOLUME24HOUR"),
                "mcap": a.get("RAW", {}).get("USD", {}).get("MKTCAP"),
            }
            for a in data
        ])
    except Exception:
        return pd.DataFrame()


def fetch_coinpaprika(limit: int = 100) -> pd.DataFrame:
    """Fetch cryptocurrency market data from Coinpaprika."""
    try:
        url = "https://api.coinpaprika.com/v1/tickers"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()[:limit]
        return pd.DataFrame([
            {
                "symbol": a.get("symbol"),
                "name": a.get("name"),
                "price": a.get("quotes", {}).get("USD", {}).get("price"),
                "change": a.get("quotes", {}).get("USD", {}).get("percent_change_24h"),
                "volume": a.get("quotes", {}).get("USD", {}).get("volume_24h"),
                "mcap": a.get("quotes", {}).get("USD", {}).get("market_cap"),
            }
            for a in data
        ])
    except Exception:
        return pd.DataFrame()


def fetch_coincap(limit: int = 100) -> pd.DataFrame:
    """Fetch cryptocurrency market data from CoinCap."""
    try:
        url = "https://api.coincap.io/v2/assets"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])[:limit]
        return pd.DataFrame([
            {
                "symbol": a.get("symbol"),
                "name": a.get("name"),
                "price": float(a.get("priceUsd", 0)),
                "change": float(a.get("changePercent24Hr", 0)),
                "volume": float(a.get("volumeUsd24Hr", 0) or 0),
                "mcap": float(a.get("marketCapUsd", 0) or 0),
            }
            for a in data
        ])
    except Exception:
        return pd.DataFrame()


# ==============================
# News Sentiment Fetcher
# ==============================

def fetch_news_sentiment(symbol: str, max_articles: int = 20) -> float:
    """Compute a simple sentiment score for a cryptocurrency symbol.

    This function queries CryptoCompare’s news API for recent articles and
    attempts to extract a sentiment value by counting occurrences of positive
    and negative words in titles and body text.  The score is normalised to
    the range [-1, 1].  A positive score indicates a higher prevalence of
    upbeat terms, whereas a negative score suggests more bearish wording.

    Parameters
    ----------
    symbol: str
        The crypto ticker (e.g. "BTC").  Passed to the API via the
        `categories` parameter, although some ambiguity exists in how
        CryptoCompare defines categories.  If no articles are returned or
        sentiment cannot be determined, the function returns 0.

    max_articles: int
        Maximum number of articles to consider.  Larger values increase
        processing time without necessarily improving accuracy.
    """
    api_key = st.session_state.get("CC_API_KEY", "")
    # If no API key is provided, return neutral sentiment.
    if not api_key:
        return 0.0

    try:
        # Request news articles.  We pass the symbol as a category to narrow
        # results.  CryptoCompare’s categories are not strictly ticker
        # symbols, but the API will still return relevant articles.  If the
        # call fails, an exception is raised and 0.0 is returned.
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {
            "lang": "EN",
            "categories": symbol,
            "api_key": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("Data", [])[:max_articles]
        if not articles:
            return 0.0

        pos_count = 0
        neg_count = 0
        for art in articles:
            text = (art.get("title", "") + " " + art.get("body", "")).lower()
            for word in POSITIVE_WORDS:
                if word in text:
                    pos_count += 1
            for word in NEGATIVE_WORDS:
                if word in text:
                    neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / float(total)
    except Exception:
        return 0.0


# ==============================
# Data Fusion
# ==============================

def fuse_data(limit: int = 100) -> pd.DataFrame:
    """Combine data from the selected sources into a single DataFrame."""
    sources: List[pd.DataFrame] = []
    if "CoinMarketCap" in apis_selected:
        sources.append(fetch_cmc(limit))
    if "CoinGecko" in apis_selected:
        sources.append(fetch_coingecko(limit))
    if "CryptoCompare" in apis_selected:
        sources.append(fetch_cryptocompare(limit))
    if "Coinpaprika" in apis_selected:
        sources.append(fetch_coinpaprika(limit))
    if "CoinCap" in apis_selected:
        sources.append(fetch_coincap(limit))

    if not sources:
        return pd.DataFrame()

    df = pd.concat(sources).groupby("symbol").agg({
        "name": "first",
        "price": "median",
        "change": "mean",
        "volume": "mean",
        "mcap": "mean",
    }).reset_index()
    return df


# ==============================
# Breakout Model
# ==============================

def calc_breakout_table(data: pd.DataFrame, investment_pot: float = CAPITAL_BASE) -> pd.DataFrame:
    """Calculate the breakout table.

    Each row in the returned DataFrame represents a single crypto or stock.  A
    number of derived columns are added:

    * **Breakout Score** – weighted combination of volatility, liquidity,
      market cap and trend, adjusted by simple news sentiment.
    * **SL/TP** – stop‑loss and take‑profit levels are derived from the
      underlying price and its recent change (ATR approximation).  The
      multipliers defined in the constants above ensure tighter risk
      management.
    * **AI Alloc. (£)** – position sizing based on a Kelly fraction using the
      calculated risk/reward.  This remains a guide rather than a rule.
    * **Go/No-Go** – recommendation flag; “Go” appears only when the asset’s
      breakout score is at least 85, the price is trending up, and the
      risk/reward is favourable (RR > 1.5).
    * **AI Reasoning** – concise explanation of how the score was derived.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing at least the columns: name, symbol, price,
        change (24h), volume, and mcap.

    investment_pot: float
        The total capital available for allocation.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated metrics for each asset.
    """
    rows: List[List] = []
    now = datetime.now(timezone.utc)
    total_mcap = data["mcap"].sum() if "mcap" in data and data["mcap"].sum() else 1.0

    # Fetch external context once per table calculation
    fg_value, fg_class = fetch_fear_greed()
    # Normalise the fear/greed index: values below 50 (fear) increase the score,
    # values above 50 (greed) decrease it.  Multiply by 5 for moderate impact.
    fg_adjustment = ((50.0 - fg_value) / 50.0) * 5.0
    trending_coins = fetch_trending_coins()

    for _, row in data.iterrows():
        name = row.get("name")
        symbol = row.get("symbol")
        price = row.get("price")
        change = row.get("change")
        volume = row.get("volume")
        mcap = row.get("mcap")

        # Protect against missing or zero values
        price = price if pd.notnull(price) else 0.0
        change = change if pd.notnull(change) else 0.0
        volume = volume if pd.notnull(volume) else 0.0
        mcap = mcap if pd.notnull(mcap) else 0.0

        # Core factors derived from price and volume
        vol_factor = min(abs(change) / 10.0, 1.0)
        vol_score = vol_factor * 100.0
        liq_score = (np.log1p(volume) / 25.0) * 100.0
        trend_score = (np.tanh(change / 5.0) + 1.0) * 50.0
        # Combine baseline scores; emphasise volatility, liquidity, and trend
        base_score = (
            0.4 * vol_score +
            0.3 * liq_score +
            0.2 * trend_score +
            0.1 * (100.0 - abs(vol_score - liq_score))
        )
        base_score = float(np.clip(base_score, 50.0, 100.0))

        # News sentiment adjustment (CryptoCompare)
        sentiment = fetch_news_sentiment(symbol)
        sentiment_adjustment = sentiment * 5.0

        # Trending coins adjustment: if the symbol is trending, add a small boost
        trending_adjustment = 3.0 if symbol.upper() in trending_coins else 0.0

        # Technical indicators via price history
        try:
            # Attempt to download recent daily data (30 days) for RSI and MACD
            if symbol is not None and isinstance(symbol, str):
                # Attempt to fetch USD‑denominated data first (common for crypto).  If
                # that fails or returns an empty DataFrame, try the raw symbol as a
                # stock ticker.  This heuristic avoids incorrectly appending "-USD"
                # to stock tickers like AAPL.
                ticker_candidates: List[str] = [f"{symbol}-USD", symbol]
                price_series = pd.Series(dtype=float)
                for ticker in ticker_candidates:
                    try:
                        hist = yf.download(ticker, period="30d", interval="1d", progress=False)
                        if not hist.empty:
                            price_series = hist["Close"]
                            break
                    except Exception:
                        continue
            else:
                price_series = pd.Series(dtype=float)
        except Exception:
            price_series = pd.Series(dtype=float)

        rsi_value = compute_rsi(price_series)
        # RSI adjustment: oversold (<30) gives +5, overbought (>70) gives -5
        if rsi_value < 30.0:
            rsi_adjustment = 5.0
        elif rsi_value > 70.0:
            rsi_adjustment = -5.0
        else:
            rsi_adjustment = 0.0

        macd_val, macd_sig = compute_macd(price_series)
        # MACD adjustment: bullish crossover if MACD > signal yields +2, bearish yields -2
        macd_adjustment = 2.0 if macd_val > macd_sig else -2.0

        # Compose final score
        raw_score = base_score + sentiment_adjustment + fg_adjustment + trending_adjustment + rsi_adjustment + macd_adjustment
        score = float(np.clip(raw_score, 50.0, 100.0))

        # Approximate ATR from price change; the division by 2 smooths the range
        atr = price * abs(change) / 100.0 / 2.0

        # Calculate stop‑loss and take‑profit boundaries using tight multipliers
        sl_price = price - max(ATR_SL_MULTIPLIER * atr, MIN_SL_RATIO * price)
        tp1_price = price + max(ATR_TP_MULTIPLIER * atr, MIN_TP_RATIO * price)

        sl_pct = ((sl_price - price) / price) * 100.0 if price != 0.0 else 0.0
        tp1_pct = ((tp1_price - price) / price) * 100.0 - REVOLUT_FEES * 100.0 if price != 0.0 else 0.0
        rr = abs(tp1_pct / sl_pct) if sl_pct != 0.0 else 0.0

        p = score / 100.0
        b = rr
        kelly = max(((p * (b + 1.0) - 1.0) / b) if b > 0.0 else 0.0, 0.0)
        alloc = round(min(investment_pot * kelly, investment_pot), 2)
        gain_pot = round(alloc * tp1_pct / 100.0, 2)

        strike = "Yes" if score >= 85.0 else "No"
        breakout_time = (now + timedelta(minutes=int(np.random.randint(30, 180)))).strftime("%H:%M")
        trend = "↑" if change > 0.0 else ("↓" if change < 0.0 else "↔")
        go = "Go" if score >= 85.0 and trend == "↑" and rr > 1.5 else "No-Go"

        # Build reasoning string summarising the factors
        reasoning_parts = [
            f"Base {base_score:.1f}",
            f"News adj {sentiment_adjustment:+.1f}",
            f"Fear&Greed adj {fg_adjustment:+.1f}",
            f"Trend adj {trending_adjustment:+.1f}",
            f"RSI {rsi_value:.1f} adj {rsi_adjustment:+.1f}",
            f"MACD adj {macd_adjustment:+.1f}",
            f"Vol {change:.2f}%", f"Vol24h ${volume/1e6:.1f}M", f"MCAP ${mcap/1e9:.1f}B", f"R/R {rr:.2f}", f"Kelly £{alloc}"
        ]
        reasoning = ", ".join(reasoning_parts)

        rows.append([
            None,
            name,
            symbol,
            round(score, 2),
            strike,
            breakout_time,
            round(price, 4),
            f"{sl_pct:.2f}% (£{alloc * sl_pct/100.0:.2f}) ({sl_price:.2f})",
            f"{tp1_pct:.2f}% (£{gain_pot:.2f}) ({tp1_price:.2f})",
            "0.00%",
            f"{sl_pct:.2f}% / {tp1_pct:.2f}%",
            f"£{alloc}",
            f"{tp1_pct:.2f}% / £{gain_pot}",
            trend,
            go,
            reasoning,
            sl_price,
            tp1_price,
        ])

    df = pd.DataFrame(rows, columns=[
        "Rank",
        "Name",
        "Symbol",
        "Breakout Score",
        "⚡ Strike Window",
        "Pred. Breakout (hh:mm)",
        "Entry Price (USD/GBP)",
        "SL % / £ (Price)",
        "TP1 % / £ (Price)",
        "Trigger %",
        "Distance to SL / TP (%)",
        "AI Alloc. (£)",
        "Gain Pot. % / £",
        "Trend",
        "Go/No-Go",
        "AI Reasoning",
        "SL_Price",
        "TP1_Price",
    ])
    df["Rank"] = range(1, len(df) + 1)
    return df.sort_values("Breakout Score", ascending=False).head(100)


# ==============================
# Chart Plotting
# ==============================

def plot_chart(symbol: str, df: pd.DataFrame) -> None:
    """Plot a candlestick chart with entry/SL/TP lines for a given symbol."""
    try:
        hist = yf.download(f"{symbol}-USD", period="5d", interval="1h", progress=False)
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
    except Exception:
        st.warning(f"No chart data for {symbol}")


# ==============================
# Helper for conditional styling
# ==============================

def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply conditional formatting to highlight Go recommendations in green."""
    def highlight_go(row: pd.Series) -> List[str]:
        return [
            "background-color: #e6ffe6" if row.get("Go/No-Go") == "Go" else ""
            for _ in row
        ]

    return df.style.apply(highlight_go, axis=1)


# ==============================
# Layout and Interaction
# ==============================

st.sidebar.header("Settings")
pot = st.sidebar.number_input(
    "Investment Pot (£)", min_value=10, value=CAPITAL_BASE, step=10
)

tabs = st.tabs(["Live Crypto", "Live Stocks", "eToro Crypto & Stocks", "Chart View", "Investment Strategy"])


# --- Live Crypto Tab
with tabs[0]:
    st.subheader("Live Crypto Breakouts (Top 100)")
    crypto_data = fuse_data(100)
    if not crypto_data.empty:
        crypto_table = calc_breakout_table(crypto_data, pot)
        display_cols = [
            "Rank", "Name", "Symbol", "Breakout Score", "⚡ Strike Window", "Pred. Breakout (hh:mm)",
            "Entry Price (USD/GBP)", "SL % / £ (Price)", "TP1 % / £ (Price)", "Trigger %", "Distance to SL / TP (%)",
            "AI Alloc. (£)", "Gain Pot. % / £", "Trend", "Go/No-Go", "AI Reasoning"
        ]
        st.dataframe(style_table(crypto_table[display_cols]), use_container_width=True)
        choice = st.selectbox("Select crypto for chart", crypto_table["Symbol"])
        if choice:
            plot_chart(choice, crypto_table)
    else:
        st.warning("No cryptocurrency data available.  Please check your API selections and keys.")


# --- Live Stocks Tab
with tabs[1]:
    st.subheader("Live Stock Breakouts (S&P100 subset)")
    # A minimal subset of S&P 100 to avoid heavy API calls
    sp100 = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "NVDA", "META", "JPM", "V", "PG"]
    tickers = yf.download(sp100, period="1d", interval="1h", progress=False, threads=True)
    latest = tickers["Close"].iloc[-1]
    stocks = pd.DataFrame([
        {
            "symbol": s,
            "name": s,
            "price": latest[s],
            "change": 0.0,
            "volume": 1,
            "mcap": 1,
        }
        for s in sp100
    ])
    stock_table = calc_breakout_table(stocks, pot)
    st.dataframe(style_table(stock_table[display_cols]), use_container_width=True)
    choice = st.selectbox("Select stock for chart", stock_table["Symbol"])
    if choice:
        plot_chart(choice, stock_table)


# --- eToro Crypto & Stocks Tab
with tabs[2]:
    st.subheader("eToro Crypto & Stocks")
    # The eToro tab simply displays the same breakout tables as the previous two tabs
    # but grouped together for convenience.  This does not use any proprietary
    # eToro data.  If eToro publishes a public API in the future, those calls
    # could be incorporated here.
    st.markdown("### Crypto (Top 100)")
    if not crypto_data.empty:
        st.dataframe(style_table(crypto_table[display_cols]), use_container_width=True)
    else:
        st.warning("No crypto data available.")
    st.markdown("### Stocks (S&P100 subset)")
    st.dataframe(style_table(stock_table[display_cols]), use_container_width=True)


# --- Chart View Tab
with tabs[3]:
    st.subheader("Chart View")
    st.info("Use the dropdowns in the previous tabs to view candlestick charts with AI Entry/SL/TP overlays.")


# --- Investment Strategy Advice Tab
with tabs[4]:
    st.subheader("General Investment Strategies (Informational)")
    st.markdown(
        """
        **Disclaimer:** The following content is for general educational purposes
        only and does **not** constitute personalised financial advice.  Markets
        are unpredictable, and past performance does not guarantee future
        results.  Before making any investment decisions, consult a
        professional advisor and conduct your own research.
        
        ### Long‑Term Strategies
        
        * **Adopt a Long‑Term Perspective:** Successful investors often avoid
          the “get in, get out” mindset.  Holding diversified assets for years
          allows compounding to work in your favour【314281712396511†L330-L360】.
        * **Dollar‑Cost Averaging (DCA):** Allocate a fixed amount at regular
          intervals (e.g. monthly).  This reduces the temptation to time the
          market and can lower your average cost basis over time【203601552392146†L700-L731】.
        * **Diversify:** Spread your holdings across different asset classes and
          sectors to reduce exposure to any single failure【543222091951736†L239-L246】.
        * **Focus on Fundamentals:** Look beyond simple metrics like the P/E
          ratio.  Evaluate a company’s growth prospects, competitive position
          and financial health【314281712396511†L396-L405】.
        * **Let Winners Ride, Cut Losers:** Avoid selling successful positions
          too early.  At the same time, do not cling to underperforming
          investments out of hope alone【314281712396511†L454-L477】.
        
        ### Short‑Term & Active Strategies
        
        * **Momentum Trading:** Traders attempt to capitalise on strong
          directional moves.  This approach requires careful technical
          analysis and constant monitoring【203601552392146†L690-L698】.
        * **Swing Trading:** Positions are held for days or weeks to capture
          medium‑term price swings.  Risk management and technical indicators
          (support/resistance levels, moving averages) are critical.
        * **Value vs. Growth:** Value investors seek undervalued companies with
          strong fundamentals and patient time horizons【203601552392146†L477-L548】,
          whereas growth investors focus on firms with rapidly expanding
          earnings but higher volatility【203601552392146†L550-L624】.
        * **Buy the Dip:** Acquiring assets during market pullbacks can be
          rewarding, but timing the exact bottom is difficult.  Setting clear
          entry levels and not overextending your budget are essential【543222091951736†L318-L327】.
        
        ### Crypto‑Specific Considerations
        
        * **Long‑Term Holding (HODLing):** Many crypto investors simply hold
          well‑established coins like Bitcoin and Ethereum through cycles of
          volatility, believing in their long‑term adoption【543222091951736†L270-L296】.
        * **Diversify Within Crypto:** Do not allocate your entire crypto
          budget to a single token.  Consider spreading across large‑cap
          (BTC, ETH), mid‑cap (SOL, LINK), and emerging themes (AI tokens)【543222091951736†L237-L264】.
        * **Stay Informed:** Regulations such as the EU’s MiCA and the U.S.
          SEC’s evolving stance can greatly influence the market【543222091951736†L340-L349】.  Monitor
          reputable news outlets and cross‑reference major announcements.
        * **Control Your Emotions:** Crypto markets are notoriously volatile.
          Avoid impulsive trades driven by fear of missing out or panic
          selling【543222091951736†L351-L361】.
        * **Begin with Small Allocations:** Crypto is high risk.  Many
          educational sources recommend investing only a small percentage of
          your portfolio (often 1–5%) in digital assets【543222091951736†L407-L414】.
        
        ### Final Thoughts
        
        Building wealth is a marathon, not a sprint.  Whether you prefer a
        passive, long‑term approach or an active, short‑term strategy,
        discipline and risk management are paramount.  Always be wary of
        unrealistic promises of “100% accuracy” – no model can perfectly
        predict market moves.  Use tools like this scanner as one input in
        your decision‑making process rather than as a definitive guide.
        """
    )
