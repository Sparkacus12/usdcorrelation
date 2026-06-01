import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY vs Yield Differentials", layout="wide")

st.title("DXY vs US Yields and Rate Differentials")

START_DATE = "1973-01-01"

@st.cache_data
def get_yahoo_close(ticker, name):
    df = yf.download(ticker, start=START_DATE, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.rename(name)

@st.cache_data
def get_fred_series(series_id, name):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.set_index("observation_date")

    s = pd.to_numeric(df[series_id], errors="coerce")
    s = s.loc[START_DATE:]
    s = s.dropna()
    return s.rename(name)

@st.cache_data
def load_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat([dxy, us10, de10, jp10, uk10], axis=1).sort_index()

    # Monthly foreign yields are forward-filled to daily/weekly frequency
    data[["Germany 10Y", "Japan 10Y", "UK 10Y"]] = data[["Germany 10Y", "Japan 10Y", "UK 10Y"]].ffill()

    data = data.dropna()

    data["US-DE 10Y spread"] = data["US 10Y"] - data["Germany 10Y"]
    data["US-JP 10Y spread"] = data["US 10Y"] - data["Japan 10Y"]
    data["US-UK 10Y spread"] = data["US 10Y"] - data["UK 10Y"]

    data["Weighted spread"] = (
        0.576 * data["US-DE 10Y spread"]
        + 0.136 * data["US-JP 10Y spread"]
        + 0.119 * data["US-UK 10Y spread"]
    )

    weekly = data.resample("W-FRI").last().dropna()

    dxy_returns = weekly["DXY"].pct_change()

    changes = pd.DataFrame({
        "US 10Y": weekly["US 10Y"].diff(),
        "US-DE spread": weekly["US-DE 10Y spread"].diff(),
        "US-JP spread": weekly["US-JP 10Y spread"].diff(),
        "US-UK spread": weekly["US-UK 10Y spread"].diff(),
        "Weighted spread": weekly["Weighted spread"].diff(),
    })

    corr_52 = changes.apply(lambda x: dxy_returns.rolling(52).corr(x))
    corr_260 = changes.apply(lambda x: dxy_returns.rolling(260).corr(x))
    corr_520 = changes.apply(lambda x: dxy_returns.rolling(520).corr(x))

    latest = weekly.copy()
    for col in changes.columns:
        latest[f"1Y corr: {col}"] = corr_52[col]
        latest[f"5Y corr: {col}"] = corr_260[col]
        latest[f"10Y corr: {col}"] = corr_520[col]

    return weekly, corr_52, corr_260, corr_520, latest

data, corr_52, corr_260, corr_520, latest = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write("Yields are from FRED. German, Japanese and UK long yields are monthly series, forward-filled to weekly frequency.")

st.subheader("1-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_52.columns:
    ax.plot(corr_52.index, corr_52[col], label=col)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("52-week rolling correlation: DXY returns vs yield/spread changes")
ax.legend()
st.pyplot(fig)

st.subheader("5-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_260.columns:
    ax.plot(corr_260.index, corr_260[col], label=col)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("260-week rolling correlation: DXY returns vs yield/spread changes")
ax.legend()
st.pyplot(fig)

st.subheader("10-year rolling correlation with DXY")

fig, ax = plt.subplots(figsize=(14, 6))
for col in corr_520.columns:
    ax.plot(corr_520.index, corr_520[col], label=col)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("520-week rolling correlation: DXY returns vs yield/spread changes")
ax.legend()
st.pyplot(fig)

def dual_chart(title, right_col, right_label):
    st.subheader(title)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(data.index, data["DXY"])
    ax1.set_ylabel("DXY")

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[right_col])
    ax2.set_ylabel(right_label)

    ax1.set_title(title)
    st.pyplot(fig)

dual_chart("DXY vs US 10-year yield", "US 10Y", "US 10Y")
dual_chart("DXY vs US-Germany 10-year spread", "US-DE 10Y spread", "US-DE spread")
dual_chart("DXY vs US-Japan 10-year spread", "US-JP 10Y spread", "US-JP spread")
dual_chart("DXY vs US-UK 10-year spread", "US-UK 10Y spread", "US-UK spread")
dual_chart("DXY vs weighted 10-year rate differential", "Weighted spread", "Weighted spread")

st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))