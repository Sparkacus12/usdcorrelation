import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="US Outperformance vs DXY", layout="wide")

st.title("US Equity Outperformance vs the Dollar")

st.write(
    "This compares S&P 500 relative performance versus MSCI ACWI ex-US, "
    "and then compares that relative performance with DXY."
)

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

    return close.rename(name)


@st.cache_data
def load_data():
    dxy = get_close("DX-Y.NYB", "DXY")
    spx = get_close("SPY", "S&P 500 ETF")
    exus = get_close("ACWX", "MSCI ACWI ex-US ETF")

    data = pd.concat([dxy, spx, exus], axis=1).dropna()

    data["US relative performance"] = data["S&P 500 ETF"] / data["MSCI ACWI ex-US ETF"]

    weekly = data.resample("W-FRI").last().dropna()

    rel_returns = weekly["US relative performance"].pct_change()
    dxy_returns = weekly["DXY"].pct_change()

    corr_52 = rel_returns.rolling(52).corr(dxy_returns)
    corr_260 = rel_returns.rolling(260).corr(dxy_returns)

    latest = weekly.copy()
    latest["1-year correlation"] = corr_52
    latest["5-year correlation"] = corr_260

    return weekly, corr_52, corr_260, latest


data, corr_52, corr_260, latest = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write(f"Number of weekly observations: **{len(data)}**")

# Chart 1
st.subheader("1-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_52.index, corr_52.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

# Chart 2
st.subheader("5-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_260.index, corr_260.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

# Chart 3
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

# Chart 4
st.subheader("US relative performance vs DXY")

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(data.index, data["US relative performance"])
ax1.set_ylabel("S&P 500 ETF / MSCI ACWI ex-US ETF")

ax2 = ax1.twinx()
ax2.plot(data.index, data["DXY"])
ax2.set_ylabel("DXY")

ax1.set_title("US equity outperformance vs DXY")
st.pyplot(fig)

# Chart 5
st.subheader("Underlying ETFs")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data["S&P 500 ETF"], label="S&P 500 ETF")
ax.plot(data.index, data["MSCI ACWI ex-US ETF"], label="MSCI ACWI ex-US ETF")
ax.set_title("S&P 500 ETF and MSCI ACWI ex-US ETF")
ax.legend()
st.pyplot(fig)

# Latest values
st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))