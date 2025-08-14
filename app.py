import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Page setup
st.set_page_config(page_title="Top 5 Breakout Picks", layout="wide")
st.title("📈 Top 5 Breakout Picks – Live AI Snapshot")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Settings
ALLOCATION = 30  # £30 fixed allocation
API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
API_KEY = "YOUR_CMC_API_KEY"  # <-- Replace with your CoinMarketCap API key
LIMIT = 50  # number of assets to pull

# Fetch live data
headers = {"X-CMC_PRO_API_KEY": API_KEY}
params = {"start": 1, "limit": LIMIT, "convert": "USD"}
response = requests.get(API_URL, headers=headers, params=params)
data = response.json()["data"]

# Simulated Breakout Scores (replace with your AI model output)
breakout_scores = {
    "ETH": 95, "SOL": 93, "ADA": 90, "XRP": 88, "BNB": 85
}

# Build table
rows = []
for asset in data:
    sym = asset["symbol"]
    if sym in breakout_scores:
        price = asset["quote"]["USD"]["price"]
        score = breakout_scores[sym]
        tp1_price = price * (1 + score / 1000)  # model-based TP1
        sl_price = price * 0.94  # 6% stop-loss

        sl_percent = (sl_price - price) / price * 100
        tp1_percent = (tp1_price - price) / price * 100
        sl_value = ALLOCATION * (sl_percent / 100)
        tp1_value = ALLOCATION * (tp1_percent / 100)

        rows.append({
            "Rank": len(rows) + 1,
            "Name": asset["name"],
            "Symbol": sym,
            "Breakout Score": score,
            "Strike Window": "Yes" if score >= 85 else "No",
            "Pred. Breakout": "—",  # placeholder
            "Entry Price": round(price, 4),
            "SL % / £ (Price)": f"{sl_percent:.2f}% / £{sl_value:.2f} ({sl_price:.4f})",
            "TP1 % / £ (Price)": f"{tp1_percent:.2f}% / £{tp1_value:.2f} ({tp1_price:.4f})",
            "AI Alloc. (£)": ALLOCATION,
            "Gain Pot. % / £": f"{tp1_percent:.2f}% / £{tp1_value:.2f}",
            "Trend": "↑",
            "Go/No-Go": "Go" if score >= 85 else "No-Go",
            "AI Reasoning": "Strong momentum with volume confirmation"
        })

# Sort by Breakout Score
df = pd.DataFrame(rows).sort_values(by="Breakout Score", ascending=False).head(5)

# Display table
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("**This is not financial advice. Always do your own research before investing.**")
