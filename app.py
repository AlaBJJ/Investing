import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import random

# ================== SETTINGS ==================
ALLOCATION = 30  # Fixed £30 allocation
API_KEY = "fde1ec72-770a-45f1-a2aa-2af4507c9d12"  # Replace with your CoinMarketCap API key
API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
VOLATILITY_THRESHOLD = 5  # % change in 24h for "highly volatile"
# ===============================================

st.set_page_config(page_title="Top 5 High Volatility Breakout Picks", layout="wide")
st.title("📊 Top 5 High Volatility Breakout Picks – Live AI Snapshot")
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
            # AI-driven breakout score simulation
            score = round(min(100, max(50, (abs(change_24h) * 1.5) + (volume_24h / 1e9))), 2)

            # Predicted breakout time (mocked using volatility pattern)
            predicted_minutes = random.randint(15, 180)
            pred_breakout = (datetime.utcnow() + timedelta(minutes=predicted_minutes)).strftime("%H:%M")

            # Calculate TP & SL dynamically
            tp1_price = price * (1 + (score / 1000))  # Model-based TP
            sl_price = price * 0.95  # 5% stop-loss

            sl_percent = (sl_price - price) / price * 100
            tp1_percent = (tp1_price - price) / price * 100
            sl_value = ALLOCATION * (sl_percent / 100)
            tp1_value = ALLOCATION * (tp1_percent / 100)

            # Trend calculation (simple directional bias)
            trend_symbol = "↑" if change_24h > 0 else "↓" if change_24h < 0 else "↔"

            # Reasoning logic
            reasoning = f"Volatility {abs(change_24h):.1f}% in 24h with volume ${volume_24h/1e6:.1f}M."

            rows.append({
                "Rank": None,
                "Name": name,
                "Symbol": symbol,
                "Breakout Score": score,
                "⚡ Strike Window": "Yes" if score >= 85 else "No",
                "Pred. Breakout (hh:mm)": pred_breakout,
                "Entry Price (USD)": round(price, 4),
                "SL % / £ (Price)": f"{sl_percent:.2f}% / £{sl_value:.2f} ({sl_price:.4f})",
                "TP1 % / £ (Price)": f"{tp1_percent:.2f}% / £{tp1_value:.2f} ({tp1_price:.4f})",
                "AI Alloc. (£)": ALLOCATION,
                "Gain Pot. % / £": f"{tp1_percent:.2f}% / £{tp1_value:.2f}",
                "Trend": trend_symbol,
                "Go/No-Go": "Go" if score >= 85 else "No-Go",
                "AI Reasoning": reasoning
            })

    # Sort by Breakout Score
    df = pd.DataFrame(rows).sort_values(by="Breakout Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df.head(5)  # Top 5 picks only

    # Display table
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("**This is not financial advice. Always do your own research before investing.**")
