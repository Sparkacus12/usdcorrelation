import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY vs S&P 500 Correlation", layout="wide")

st.title("DXY vs S&P 500 Rolling Correlation")

@st.cache_data
def get_close(ticker, name):
    df = yf.download(
        ticker,
        start="1971-01-01",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.rename(name)
    return close


@st.cache_data
def load_data():
    dxy = get_close("DX-Y.NYB", "DXY")
    spx = get_close("^GSPC", "S&P 500")

    data = pd.concat([dxy, spx], axis=1).dropna()
    weekly = data.resample("W-FRI").last().dropna()

    returns = weekly.pct_change().dropna()

    corr_52 = returns["DXY"].rolling(52).corr(returns["S&P 500"])
    corr_260 = returns["DXY"].rolling(260).corr(returns["S&P 500"])

    latest = weekly.copy()
    latest["1-year correlation"] = corr_52
    latest["5-year correlation"] = corr_260

    return weekly, corr_52, corr_260, latest


data, corr_52, corr_260, latest = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write(f"Number of weekly observations: **{len(data)}**")

# 1-year rolling correlation
st.subheader("1-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_52.index, corr_52.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: DXY vs S&P 500")
st.pyplot(fig)

# 5-year rolling correlation
st.subheader("5-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_260.index, corr_260.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: DXY vs S&P 500")
st.pyplot(fig)

# Regime chart
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

# Levels chart
st.subheader("DXY and S&P 500 levels")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["S&P 500"])
ax1.set_ylabel("S&P 500")

ax2 = ax1.twinx()
ax2.plot(data.index, data["DXY"])
ax2.set_ylabel("DXY")

ax1.set_title("DXY and S&P 500 levels")
st.pyplot(fig)

# Latest data
st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))