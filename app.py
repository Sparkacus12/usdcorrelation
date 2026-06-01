import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import date

st.set_page_config(page_title="US Relative Performance vs DXY", layout="wide")

st.title("US Equity Relative Performance vs DXY")

st.write(
    "This compares S&P 500 relative performance against MSCI EAFE, "
    "and then compares that relative performance with DXY."
)

START_DATE = "1971-01-01"


@st.cache_data
def get_yahoo_close(ticker, name):
    df = yf.download(
        ticker,
        start=START_DATE,
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
def get_msci_eafe():
    """
    MSCI EAFE index.
    MSCI code 990300 is commonly used for MSCI EAFE.
    This tries MSCI's public chart endpoint.
    """

    end_date = date.today().strftime("%Y-%m-%d")

    variants = ["NETR", "STRD", "GRTR"]

    for variant in variants:
        url = (
            "https://app2.msci.com/products/service/index/indexmaster/"
            "getLevelDataForGraph"
            f"?currency_symbol=USD"
            f"&index_variant={variant}"
            f"&start_date={START_DATE}"
            f"&end_date={end_date}"
            f"&data_frequency=DAILY"
            f"&index_codes=990300"
        )

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            js = r.json()

            data = js.get("data", [])

            rows = []

            for item in data:
                if isinstance(item, dict):
                    d = item.get("date") or item.get("calc_date")
                    v = (
                        item.get("level")
                        or item.get("value")
                        or item.get("index_level")
                    )

                    if d is not None and v is not None:
                        rows.append((d, v))

            if rows:
                s = pd.Series(
                    data=[float(v) for d, v in rows],
                    index=pd.to_datetime([d for d, v in rows]),
                    name=f"MSCI EAFE {variant}"
                )

                s = s.sort_index()
                return s

        except Exception:
            continue

    raise ValueError(
        "Could not download MSCI EAFE from MSCI. "
        "If this fails, MSCI may have changed its public data endpoint."
    )


@st.cache_data
def load_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")
    spx = get_yahoo_close("^GSPC", "S&P 500")
    eafe = get_msci_eafe()

    data = pd.concat([dxy, spx, eafe], axis=1).dropna()

    eafe_col = eafe.name

    data["US relative performance"] = data["S&P 500"] / data[eafe_col]

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

    return weekly, corr_52, corr_260, corr_520, latest, eafe_col


data, corr_52, corr_260, corr_520, latest, eafe_col = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write(f"Number of weekly observations: **{len(data)}**")
st.write(f"Foreign equity series used: **{eafe_col}**")

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
st.subheader("10-year rolling correlation")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(corr_520.index, corr_520.values)
ax.axhline(0, linestyle="--")
ax.set_ylabel("Correlation")
ax.set_title("520-week rolling correlation: US relative performance vs DXY")
st.pyplot(fig)

# Chart 4
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

# Chart 5
st.subheader("US relative performance vs DXY")

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(data.index, data["US relative performance"])
ax1.set_ylabel("S&P 500 / MSCI EAFE")

ax2 = ax1.twinx()
ax2.plot(data.index, data["DXY"])
ax2.set_ylabel("DXY")

ax1.set_title("US equity outperformance vs DXY")
st.pyplot(fig)

# Chart 6
st.subheader("Underlying equity indices")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data["S&P 500"], label="S&P 500")
ax.plot(data.index, data[eafe_col], label=eafe_col)
ax.set_title("S&P 500 and MSCI EAFE")
ax.legend()
st.pyplot(fig)

# Latest data
st.subheader("Latest values")
st.dataframe(latest.dropna().tail(20))