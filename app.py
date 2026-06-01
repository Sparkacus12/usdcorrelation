import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY vs S&P 500 Correlation", layout="wide")

st.title("DXY vs S&P 500 Rolling Correlation")

@st.cache_data
def load_data():
    dxy = yf.download("DX-Y.NYB", auto_adjust=True, progress=False)[["Close"]]
    spx = yf.download("^GSPC", auto_adjust=True, progress=False)[["Close"]]

    dxy = dxy.rename(columns={"Close": "DXY"})
    spx = spx.rename(columns={"Close": "S&P 500"})

    dxy_w = dxy.resample("W-FRI").last()
    spx_w = spx.resample("W-FRI").last()

    data = dxy_w.join(spx_w, how="inner")
    returns = data.pct_change().dropna()

    corr_52 = returns["DXY"].rolling(52).corr(returns["S&P 500"])
    corr_260 = returns["DXY"].rolling(260).corr(returns["S&P 500"])

    return data, corr_52, corr_260

data, corr_52, corr_260 = load_data()

st.write(
    f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**."
)

# Chart 1: 1-year rolling correlation
st.subheader("1-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_52.index, corr_52.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: DXY vs S&P 500")
st.pyplot(fig)

# Chart 2: 5-year rolling correlation
st.subheader("5-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_260.index, corr_260.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: DXY vs S&P 500")
st.pyplot(fig)

# Chart 3: regime chart
st.subheader("Correlation regimes")

regime = corr_52.dropna()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(regime.index, regime.values)
ax.axhline(0, linestyle="--")

for i in range(len(regime) - 1):
    start = regime.index[i]
    end = regime.index[i + 1]
    value = regime.iloc[i]

    if value >= 0:
        ax.axvspan(start, end, alpha=0.12)
    else:
        ax.axvspan(start, end, alpha=0.04)

ax.set_ylabel("Correlation")
ax.set_title("Positive vs negative correlation regimes")
st.pyplot(fig)

# Chart 4: levels chart
st.subheader("DXY and S&P 500 levels")

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(data.index, data["S&P 500"])
ax1.set_ylabel("S&P 500")

ax2 = ax1.twinx()
ax2.plot(data.index, data["DXY"])
ax2.set_ylabel("DXY")

ax1.set_title("DXY and S&P 500 levels")
st.pyplot(fig)

# Data table
st.subheader("Latest values")

latest = pd.DataFrame({
    "DXY": data["DXY"],
    "S&P 500": data["S&P 500"],
    "1-year correlation": corr_52,
    "5-year correlation": corr_260
}).dropna()

st.dataframe(latest.tail(20))