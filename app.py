import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY, Equities and Yield Differentials", layout="wide")

st.title("DXY, Equities and Yield Differentials")

START_DATE = "1973-01-01"


@st.cache_data
def get_yahoo_close(ticker, name):
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)

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
    s = s.loc[START_DATE:].dropna()

    return s.rename(name)


def rolling_corr(x, y, weeks):
    return x.rolling(weeks).corr(y)


def plot_corr(title, corr_df):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(14, 6))

    for col in corr_df.columns:
        ax.plot(corr_df.index, corr_df[col], label=col)

    ax.axhline(0, linestyle="--")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def dual_chart(title, left_series, left_label, right_series, right_label):
    st.subheader(title)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(left_series.index, left_series.values)
    ax1.set_ylabel(left_label)

    ax2 = ax1.twinx()
    ax2.plot(right_series.index, right_series.values)
    ax2.set_ylabel(right_label)

    ax1.set_title(title)
    st.pyplot(fig)


@st.cache_data
def load_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")
    spx = get_yahoo_close("^GSPC", "S&P 500")
    efa = get_yahoo_close("EFA", "MSCI EAFE ETF")

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat([dxy, spx, efa, us10, de10, jp10, uk10], axis=1).sort_index()

    data[["Germany 10Y", "Japan 10Y", "UK 10Y"]] = data[
        ["Germany 10Y", "Japan 10Y", "UK 10Y"]
    ].ffill()

    data = data.dropna()

    data["US relative performance"] = data["S&P 500"] / data["MSCI EAFE ETF"]

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
    spx_returns = weekly["S&P 500"].pct_change()
    relative_returns = weekly["US relative performance"].pct_change()

    yield_changes = pd.DataFrame({
        "US 10Y": weekly["US 10Y"].diff(),
        "US-DE spread": weekly["US-DE 10Y spread"].diff(),
        "US-JP spread": weekly["US-JP 10Y spread"].diff(),
        "US-UK spread": weekly["US-UK 10Y spread"].diff(),
        "Weighted spread": weekly["Weighted spread"].diff(),
    })

    equity_corr_52 = pd.DataFrame({
        "S&P 500 vs DXY": rolling_corr(spx_returns, dxy_returns, 52),
        "US relative vs DXY": rolling_corr(relative_returns, dxy_returns, 52),
    })

    equity_corr_260 = pd.DataFrame({
        "S&P 500 vs DXY": rolling_corr(spx_returns, dxy_returns, 260),
        "US relative vs DXY": rolling_corr(relative_returns, dxy_returns, 260),
    })

    equity_corr_520 = pd.DataFrame({
        "S&P 500 vs DXY": rolling_corr(spx_returns, dxy_returns, 520),
        "US relative vs DXY": rolling_corr(relative_returns, dxy_returns, 520),
    })

    yield_corr_52 = yield_changes.apply(lambda x: rolling_corr(dxy_returns, x, 52))
    yield_corr_260 = yield_changes.apply(lambda x: rolling_corr(dxy_returns, x, 260))
    yield_corr_520 = yield_changes.apply(lambda x: rolling_corr(dxy_returns, x, 520))

    latest = weekly.copy()

    for col in equity_corr_52.columns:
        latest[f"1Y corr: {col}"] = equity_corr_52[col]
        latest[f"5Y corr: {col}"] = equity_corr_260[col]
        latest[f"10Y corr: {col}"] = equity_corr_520[col]

    for col in yield_corr_52.columns:
        latest[f"1Y corr: DXY vs {col}"] = yield_corr_52[col]
        latest[f"5Y corr: DXY vs {col}"] = yield_corr_260[col]
        latest[f"10Y corr: DXY vs {col}"] = yield_corr_520[col]

    return (
        weekly,
        equity_corr_52,
        equity_corr_260,
        equity_corr_520,
        yield_corr_52,
        yield_corr_260,
        yield_corr_520,
        latest,
    )


(
    data,
    equity_corr_52,
    equity_corr_260,
    equity_corr_520,
    yield_corr_52,
    yield_corr_260,
    yield_corr_520,
    latest,
) = load_data()


st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write("Equity foreign proxy: **EFA — iShares MSCI EAFE ETF**.")
st.write("Yield data: **FRED**. German, Japanese and UK yields are monthly and forward-filled.")

st.header("1. DXY correlation with equities")

plot_corr("1-year rolling correlation: equities vs DXY", equity_corr_52)
plot_corr("5-year rolling correlation: equities vs DXY", equity_corr_260)
plot_corr("10-year rolling correlation: equities vs DXY", equity_corr_520)

st.header("2. DXY correlation with yields and rate differentials")

plot_corr("1-year rolling correlation: DXY vs yield/spread changes", yield_corr_52)
plot_corr("5-year rolling correlation: DXY vs yield/spread changes", yield_corr_260)
plot_corr("10-year rolling correlation: DXY vs yield/spread changes", yield_corr_520)

st.header("3. Level charts")

dual_chart(
    "DXY vs S&P 500",
    data["DXY"],
    "DXY",
    data["S&P 500"],
    "S&P 500",
)

dual_chart(
    "DXY vs US relative equity performance",
    data["DXY"],
    "DXY",
    data["US relative performance"],
    "S&P 500 / EFA",
)

dual_chart(
    "DXY vs US 10-year yield",
    data["DXY"],
    "DXY",
    data["US 10Y"],
    "US 10Y",
)

dual_chart(
    "DXY vs US-Germany 10-year spread",
    data["DXY"],
    "DXY",
    data["US-DE 10Y spread"],
    "US-DE spread",
)

dual_chart(
    "DXY vs US-Japan 10-year spread",
    data["DXY"],
    "DXY",
    data["US-JP 10Y spread"],
    "US-JP spread",
)

dual_chart(
    "DXY vs US-UK 10-year spread",
    data["DXY"],
    "DXY",
    data["US-UK 10Y spread"],
    "US-UK spread",
)

dual_chart(
    "DXY vs weighted 10-year rate differential",
    data["DXY"],
    "DXY",
    data["Weighted spread"],
    "Weighted spread",
)

st.header("4. Latest values")

st.dataframe(latest.dropna().tail(20))