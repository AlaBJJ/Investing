import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import random

# ================== SETTINGS ==================
API_KEY = "fde1ec72-770a-45f1-a2aa-2af4507c9d12"  # Replace with your CoinMarketCap API key
API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
VOLATILITY_THRESHOLD = 5  # Minimum % change 24h to be considered high volatility
CAPITAL_BASE = 50  # Max base allocation per trade (£)
REVOLUT_PREMIUM_FEE = 0.0099
REVOLUT_SPREAD = 0.005
ROUND_TRIP_COST = (REVOLUT_PREMIUM_FEE + REVOLUT_SPREAD) * 2
# ===============================================

st.set_page_config(page_title="AI Predictive Crypto Breakouts", layout="wide")
st.title("📊 Top 5 High Volatility Breakout Picks – AI Predictive Model (Revolut Premium Fees Applied)")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ===== Fetch live crypto data =====
headers = {"X-CMC_PRO_API_KEY": API_KEY}
params = {"start": 1, "limit": 200, "convert": "USD"}
response = requests.get(API_URL, headers=headers, params=params)

if response.status_code != 200:
    st.error("Error fetching data. Check your API key.")
else:
    data = response.json()["data"]

    # ===== Placeholder Predictive Model =====
    def ai_predict_breakout_prob(volatility, liquidity, trend_strength):
        """
        Simulated AI breakout probability based on volatility, liquidity, and trend.
        Later replaced with trained ML model.
        """
        base = 50 + (volatility * 2) + (liquidity * 10) + (trend_strength * 15)
        return min(100, max(50, base + random.uniform(-5, 5)))

    def ai_dynamic_allocation(score, liquidity_weight, rr_ratio):
        """
        Adjusts allocation based on conviction, liquidity, and R/R.
        """
        return round(CAPITAL_BASE * (score / 100) * liquidity_weight * min(1, rr_ratio / 2), 2)

    rows = []
    for asset in data:
        symbol = asset["symbol"]
        name = asset["name"]
        price = asset["quote"]["USD"]["price"]
        change_24h = asset["quote"]["USD"]["percent_change_24h"]
        volume_24h = asset["quote"]["USD"]["volume_24h"]

        if abs(change_24h) >= VOLATILITY_THRESHOLD:
            # ===== Feature Engineering =====
            liquidity_weight = min(1, volume_24h / 1e9)
            trend_strength = np.tanh(change_24h / 10)
            volatility_factor = abs(change_24h)

            # ===== Predictions =====
            breakout_score = ai_predict_breakout_prob(volatility_factor, liquidity_weight, trend_strength)

            # ===== ATR (approximation) =====
            atr = price * (volatility_factor / 100) / 2

            # ===== SL & TP =====
            sl_price = price - max(1.5 * atr, price * 0.02)
            tp1_price = price + max(2.5 * atr, price * 0.03)
            sl_percent = (sl_price - price) / price * 100
            tp1_percent = (tp1_price - price) / price * 100
            rr_ratio = abs(tp1_percent / sl_percent)

            # ===== Allocation =====
            ai_alloc = ai_dynamic_allocation(breakout_score, liquidity_weight, rr_ratio)

            # ===== Net After Fees =====
            net_tp1_percent = tp1_percent - (ROUND_TRIP_COST * 100)
            net_tp1_value = ai_alloc * (net_tp1_percent / 100)

            # ===== Trigger & Distance =====
            trigger_price = price
            trigger_percent = ((price - trigger_price) / trigger_price) * 100
            distance_to_sl_percent = ((sl_price - price) / price) * 100
            distance_to_tp_percent = ((tp1_price - price) / price) * 100

            # ===== Predicted Breakout Time =====
            predicted_minutes = random.randint(30, 180)
            pred_breakout = (datetime.utcnow() + timedelta(minutes=predicted_minutes)).strftime("%H:%M")

            # ===== Trend =====
            trend_symbol = "↑" if change_24h > 0 else "↓" if change_24h < 0 else "↔"

            # ===== Reasoning =====
            reasoning = (
                f"Prob {breakout_score:.1f}%, Vol {abs(change_24h):.1f}%, "
                f"Liquidity ${volume_24h/1e6:.1f}M, ATR {atr:.2f}, R/R {rr_ratio:.2f}."
            )

            rows.append({
                "Rank": None,
                "Name": name,
                "Symbol": symbol,
                "Breakout Score": round(breakout_score, 2),
                "⚡ Strike Window": "Yes" if breakout_score >= 85 else "No",
                "Pred. Breakout (hh:mm)": pred_breakout,
                "Entry Price (USD)": round(price, 4),
                "SL % / £ (Price)": f"{sl_percent:.2f}% / £{ai_alloc * (sl_percent/100):.2f} ({sl_price:.4f})",
                "TP1 % / £ (Price)": f"{net_tp1_percent:.2f}% / £{net_tp1_value:.2f} ({tp1_price:.4f})",
                "Trigger %": f"{trigger_percent:.2f}%",
                "Distance to SL / TP (%)": f"{distance_to_sl_percent:.2f}% / {distance_to_tp_percent:.2f}%",
                "AI Alloc. (£)": ai_alloc,
                "Gain Pot. % / £": f"{net_tp1_percent:.2f}% / £{net_tp1_value:.2f}",
                "Trend": trend_symbol,
                "Go/No-Go": "Go" if breakout_score >= 85 else "No-Go",
                "AI Reasoning": reasoning
            })

    # ===== Sort & Display =====
    df = pd.DataFrame(rows).sort_values(by="Breakout Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df.head(5)

    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("**This is not financial advice. Always do your own research before investing.**")
