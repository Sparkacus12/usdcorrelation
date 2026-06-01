import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="US Relative Performance vs DXY", layout="wide")

st.title("US Relative Equity Performance vs DXY")

START_DATE = "1971-01-01"

@st.cache_data
def get_close(ticker, name):
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.rename(name)

@st.cache_data
def load_data():
    dxy = get_close("DX-Y.NYB", "DXY")
    spx = get_close("^GSPC", "S&P 500")
    efa = get_close("EFA", "MSCI EAFE ETF")

    data = pd.concat([dxy, spx, efa], axis=1).dropna()

    data["US relative performance"] = data["S&P 500"] / data["MSCI EAFE ETF"]

    weekly = data.resample("W-FRI").last().dropna()

    rel_returns = weekly["US relative performance"].pct_change()
    dxy_returns = weekly["DXY"].pct_change()

    corr_52 = rel_returns.rolling(52).corr(dxy_returns)
    corr_260 = rel_returns.rolling(260).corr(dxy_returns)
    corr_520 = rel_returns.rolling(520).corr(dxy_returns)

    latest = weekly.copy()
    latest["1-year correlation"] = corr_52
    latest["5-year correlation"] = corr_260
    latest["10-year correlation"] = corr_520

    return weekly, corr_52, corr_260, corr_520, latest

data, corr_52, corr_260, corr_520, latest = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write("Foreign equity proxy: **EFA — iShares MSCI EAFE ETF**")

st.subheader("1-year rolling correlation")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_52.index, corr_52.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

st.subheader("5-year rolling correlation")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_260.index, corr_260.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

st.subheader("10-year rolling correlation")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_520.index, corr_520.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("520-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

st.subheader("US relative performance vs DXY")
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["US relative performance"])
ax1.set_ylabel("S&P 500 / EFA")

ax2 = ax1.twinx()
ax2.plot(data.index, data["DXY"])
ax2.set_ylabel("DXY")

ax1.set_title("US equity outperformance vs DXY")
st.pyplot(fig)

st.subheader("Underlying series")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data["S&P 500"], label="S&P 500")
ax.plot(data.index, data["MSCI EAFE ETF"], label="EFA")
ax.legend()
st.pyplot(fig)

st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))