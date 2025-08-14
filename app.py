import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import random

# ================== SETTINGS ==================
API_KEY = "YOUR_CMC_API_KEY"  # Replace with your CoinMarketCap API key
API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
VOLATILITY_THRESHOLD = 5  # % change in 24h for high volatility
CAPITAL_BASE = 50  # Max possible AI allocation per trade (£)
REVOLUT_PREMIUM_FEE = 0.0099  # 0.99% per trade
REVOLUT_SPREAD = 0.005  # 0.5% per trade
ROUND_TRIP_COST = (REVOLUT_PREMIUM_FEE + REVOLUT_SPREAD) * 2  # ~2.98% total
# ===============================================

st.set_page_config(page_title="Top 5 High Volatility Breakout Picks", layout="wide")
st.title("📊 Top 5 High Volatility Breakout Picks – AI Enhanced (Revolut Premium Fees Applied)")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Fetch live crypto prices
headers = {"X-CMC_PRO_API_KEY": API_KEY}
params = {"start": 1, "limit": 200, "convert": "USD"}
response = requests.get(API_URL, headers=headers, params=params)

if response.status_code != 200:
    st.error("Error fetching data. Check your API key.")
else:
    data = response.json()["data"]

    rows = []
    for asset in data:
        symbol = asset["symbol"]
        name = asset["name"]
        price = asset["quote"]["USD"]["price"]
        change_24h = asset["quote"]["USD"]["percent_change_24h"]
        volume_24h = asset["quote"]["USD"]["volume_24h"]

        # Only include highly volatile assets
        if abs(change_24h) >= VOLATILITY_THRESHOLD:
            # ===== AI Breakout Score =====
            liquidity_weight = min(1, volume_24h / 1e9)
            volatility_factor = min(2, abs(change_24h) / 10)
            score = round(min(100, 50 + (volatility_factor * 20) + (liquidity_weight * 20)), 2)

            # ===== AI Allocation =====
            risk_adjustor = 1.0 if score >= 90 else 0.7 if score >= 85 else 0.5
            ai_alloc = round(CAPITAL_BASE * (score / 100) * liquidity_weight * risk_adjustor, 2)

            # ===== ATR (approximation for demo) =====
            atr = price * (abs(change_24h) / 100) / 2

            # ===== AI Stop Loss =====
            sl_price = price - max(1.5 * atr, price * 0.02)
            sl_percent = (sl_price - price) / price * 100
            sl_value = ai_alloc * (sl_percent / 100)

            # ===== AI Take Profit =====
            tp1_price = price + max(2.5 * atr, price * 0.03)
            tp1_percent = (tp1_price - price) / price * 100
            tp1_value = ai_alloc * (tp1_percent / 100)

            # ===== Predicted Breakout Time =====
            predicted_minutes = random.randint(30, 180)
            pred_breakout = (datetime.utcnow() + timedelta(minutes=predicted_minutes)).strftime("%H:%M")

            # ===== Adjust for Revolut Premium Fees =====
            net_tp1_percent = tp1_percent - (ROUND_TRIP_COST * 100)
            net_tp1_value = tp1_value - (ai_alloc * ROUND_TRIP_COST)

            # ===== Trend =====
            trend_symbol = "↑" if change_24h > 0 else "↓" if change_24h < 0 else "↔"

            # ===== Trigger & Distance =====
            trigger_price = price  # Placeholder – replace with AI breakout trigger
            trigger_percent = ((price - trigger_price) / trigger_price) * 100
            distance_to_sl_percent = ((sl_price - price) / price) * 100
            distance_to_tp_percent = ((tp1_price - price) / price) * 100

            # ===== AI Reasoning =====
            reasoning = (
                f"Volatility {abs(change_24h):.1f}%, vol ${volume_24h/1e6:.1f}M, "
                f"ATR {atr:.2f}, R/R {(net_tp1_percent / abs(sl_percent)):.2f}."
            )

            rows.append({
                "Rank": None,
                "Name": name,
                "Symbol": symbol,
                "Breakout Score": score,
                "⚡ Strike Window": "Yes" if score >= 85 else "No",
                "Pred. Breakout (hh:mm)": pred_breakout,
                "Entry Price (USD)": round(price, 4),
                "SL % / £ (Price)": f"{sl_percent:.2f}% / £{sl_value:.2f} ({sl_price:.4f})",
                "TP1 % / £ (Price)": f"{net_tp1_percent:.2f}% / £{net_tp1_value:.2f} ({tp1_price:.4f})",
                "Trigger %": f"{trigger_percent:.2f}%",
                "Distance to SL / TP (%)": f"{distance_to_sl_percent:.2f}% / {distance_to_tp_percent:.2f}%",
                "AI Alloc. (£)": ai_alloc,
                "Gain Pot. % / £": f"{net_tp1_percent:.2f}% / £{net_tp1_value:.2f}",
                "Trend": trend_symbol,
                "Go/No-Go": "Go" if score >= 85 else "No-Go",
                "AI Reasoning": reasoning
            })

    # ===== Sort & Rank =====
    df = pd.DataFrame(rows).sort_values(by="Breakout Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df.head(5)

    # ===== Display Table =====
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("**This is not financial advice. Always do your own research before investing.**")
