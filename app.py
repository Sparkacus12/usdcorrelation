import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY vs Yield Differentials", layout="wide")

st.title("DXY vs Bond Yields and Rate Differentials")

START_DATE = "2000-01-01"

@st.cache_data
def get_close(ticker, name):
    df = yf.download(ticker, start=START_DATE, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.rename(name)

@st.cache_data
def load_data():
    dxy = get_close("DX-Y.NYB", "DXY")
    us10 = get_close("^TNX", "US 10Y")
    de10 = get_close("^DE10Y", "Germany 10Y")
    jp10 = get_close("^JP10Y", "Japan 10Y")

    data = pd.concat([dxy, us10, de10, jp10], axis=1).dropna()

    data["US-DE 10Y spread"] = data["US 10Y"] - data["Germany 10Y"]
    data["US-JP 10Y spread"] = data["US 10Y"] - data["Japan 10Y"]

    data["Weighted spread"] = (
        0.57 * data["US-DE 10Y spread"]
        + 0.14 * data["US-JP 10Y spread"]
    )

    weekly = data.resample("W-FRI").last().dropna()

    dxy_returns = weekly["DXY"].pct_change()

    us10_changes = weekly["US 10Y"].diff()
    usde_changes = weekly["US-DE 10Y spread"].diff()
    usjp_changes = weekly["US-JP 10Y spread"].diff()
    weighted_changes = weekly["Weighted spread"].diff()

    corr_52 = pd.DataFrame({
        "US 10Y": dxy_returns.rolling(52).corr(us10_changes),
        "US-DE spread": dxy_returns.rolling(52).corr(usde_changes),
        "US-JP spread": dxy_returns.rolling(52).corr(usjp_changes),
        "Weighted spread": dxy_returns.rolling(52).corr(weighted_changes),
    })

    corr_260 = pd.DataFrame({
        "US 10Y": dxy_returns.rolling(260).corr(us10_changes),
        "US-DE spread": dxy_returns.rolling(260).corr(usde_changes),
        "US-JP spread": dxy_returns.rolling(260).corr(usjp_changes),
        "Weighted spread": dxy_returns.rolling(260).corr(weighted_changes),
    })

    corr_520 = pd.DataFrame({
        "US 10Y": dxy_returns.rolling(520).corr(us10_changes),
        "US-DE spread": dxy_returns.rolling(520).corr(usde_changes),
        "US-JP spread": dxy_returns.rolling(520).corr(usjp_changes),
        "Weighted spread": dxy_returns.rolling(520).corr(weighted_changes),
    })

    latest = weekly.copy()
    for col in corr_52.columns:
        latest[f"1Y corr: {col}"] = corr_52[col]
        latest[f"5Y corr: {col}"] = corr_260[col]
        latest[f"10Y corr: {col}"] = corr_520[col]

    return weekly, corr_52, corr_260, corr_520, latest

data, corr_52, corr_260, corr_520, latest = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write("DXY is compared with weekly changes in yields/spreads.")

# 1-year rolling correlations
st.subheader("1-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_52.columns:
    ax.plot(corr_52.index, corr_52[col], label=col)

ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: DXY vs yield/spread changes")
ax.legend()
st.pyplot(fig)

# 5-year rolling correlations
st.subheader("5-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_260.columns:
    ax.plot(corr_260.index, corr_260[col], label=col)

ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: DXY vs yield/spread changes")
ax.legend()
st.pyplot(fig)

# 10-year rolling correlations
st.subheader("10-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_520.columns:
    ax.plot(corr_520.index, corr_520[col], label=col)

ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("520-week rolling correlation: DXY vs yield/spread changes")
ax.legend()
st.pyplot(fig)

# DXY vs US 10Y
st.subheader("DXY vs US 10-year yield")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["DXY"])
ax1.set_ylabel("DXY")

ax2 = ax1.twinx()
ax2.plot(data.index, data["US 10Y"])
ax2.set_ylabel("US 10Y yield")

ax1.set_title("DXY vs US 10-year yield")
st.pyplot(fig)

# DXY vs US-DE spread
st.subheader("DXY vs US-Germany 10-year spread")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["DXY"])
ax1.set_ylabel("DXY")

ax2 = ax1.twinx()
ax2.plot(data.index, data["US-DE 10Y spread"])
ax2.set_ylabel("US-Germany 10Y spread")

ax1.set_title("DXY vs US-Germany 10-year spread")
st.pyplot(fig)

# DXY vs US-JP spread
st.subheader("DXY vs US-Japan 10-year spread")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["DXY"])
ax1.set_ylabel("DXY")

ax2 = ax1.twinx()
ax2.plot(data.index, data["US-JP 10Y spread"])
ax2.set_ylabel("US-Japan 10Y spread")

ax1.set_title("DXY vs US-Japan 10-year spread")
st.pyplot(fig)

# DXY vs weighted spread
st.subheader("DXY vs weighted rate differential")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data["DXY"])
ax1.set_ylabel("DXY")

ax2 = ax1.twinx()
ax2.plot(data.index, data["Weighted spread"])
ax2.set_ylabel("Weighted 10Y spread")

ax1.set_title("DXY vs weighted 10-year rate differential")
st.pyplot(fig)

# Data table
st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))