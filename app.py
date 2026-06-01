import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY, Equities, Yields, VIXEQ and SKEW", layout="wide")

st.title("DXY, Equities, Yield Differentials, VIXEQ and SKEW")

START_DATE = "1973-01-01"
VIX_FILE = "vix data.xlsx"


@st.cache_data
def get_yahoo_close(ticker, name):
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)

    if df.empty:
        return pd.Series(dtype=float, name=name)

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
    return s.loc[START_DATE:].dropna().rename(name)


@st.cache_data
def load_vixeq_file():
    df = pd.read_excel(VIX_FILE)

    df = df.rename(columns={
        "Date": "Date",
        "VIXEQ Index  (L1)": "VIXEQ",
        "VIX Index  (R1)": "VIX",
        "SPX Index  (R2)": "S&P 500"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    df = df[["VIXEQ", "VIX", "S&P 500"]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    weekly = df.resample("W-FRI").last().dropna()

    weekly["VIXEQ minus VIX"] = weekly["VIXEQ"] - weekly["VIX"]
    weekly["VIXEQ / VIX"] = weekly["VIXEQ"] / weekly["VIX"]

    weekly["VIXEQ/VIX z-score"] = (
        weekly["VIXEQ / VIX"] - weekly["VIXEQ / VIX"].rolling(52).mean()
    ) / weekly["VIXEQ / VIX"].rolling(52).std()

    weekly["Forward 3m S&P return"] = weekly["S&P 500"].shift(-13) / weekly["S&P 500"] - 1
    weekly["Forward 6m S&P return"] = weekly["S&P 500"].shift(-26) / weekly["S&P 500"] - 1
    weekly["Forward 12m S&P return"] = weekly["S&P 500"].shift(-52) / weekly["S&P 500"] - 1

    return weekly


def plot_lines(title, df):
    st.subheader(title)
    df = df.dropna(how="all")

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    ax.axhline(0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Correlation")
    ax.legend()
    st.pyplot(fig)


def dual_chart(title, left, left_label, right, right_label):
    st.subheader(title)
    df = pd.concat([left, right], axis=1).dropna()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df.iloc[:, 0])
    ax1.set_ylabel(left_label)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df.iloc[:, 1])
    ax2.set_ylabel(right_label)

    ax1.set_title(title)
    st.pyplot(fig)


def scatter_chart(title, x, y, x_label, y_label):
    st.subheader(title)
    df = pd.concat([x, y], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    st.pyplot(fig)


@st.cache_data
def load_core_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")
    spx = get_yahoo_close("^GSPC", "S&P 500")
    efa = get_yahoo_close("EFA", "MSCI EAFE ETF")
    skew = get_yahoo_close("^SKEW", "SKEW")

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat([dxy, spx, efa, skew, us10, de10, jp10, uk10], axis=1).sort_index()

    data[["Germany 10Y", "Japan 10Y", "UK 10Y"]] = data[
        ["Germany 10Y", "Japan 10Y", "UK 10Y"]
    ].ffill()

    data["US relative performance"] = data["S&P 500"] / data["MSCI EAFE ETF"]

    data["US-DE 10Y spread"] = data["US 10Y"] - data["Germany 10Y"]
    data["US-JP 10Y spread"] = data["US 10Y"] - data["Japan 10Y"]
    data["US-UK 10Y spread"] = data["US 10Y"] - data["UK 10Y"]

    data["Weighted spread"] = (
        0.576 * data["US-DE 10Y spread"]
        + 0.136 * data["US-JP 10Y spread"]
        + 0.119 * data["US-UK 10Y spread"]
    )

    weekly = data.resample("W-FRI").last()

    weekly["SKEW z-score"] = (
        weekly["SKEW"] - weekly["SKEW"].rolling(52).mean()
    ) / weekly["SKEW"].rolling(52).std()

    weekly["SKEW 13w change"] = weekly["SKEW"] - weekly["SKEW"].shift(13)
    weekly["SKEW trending lower"] = weekly["SKEW 13w change"] < 0

    weekly["SPX 26w return"] = weekly["S&P 500"] / weekly["S&P 500"].shift(26) - 1
    weekly["SPX above 40w MA"] = weekly["S&P 500"] > weekly["S&P 500"].rolling(40).mean()
    weekly["SPX firmly higher"] = (
        (weekly["SPX 26w return"] > 0.10)
        & weekly["SPX above 40w MA"]
    )

    weekly["SKEW low + falling + SPX uptrend"] = (
        (weekly["SKEW z-score"] < -0.5)
        & weekly["SKEW trending lower"]
        & weekly["SPX firmly higher"]
    )

    weekly["Forward 3m S&P return"] = weekly["S&P 500"].shift(-13) / weekly["S&P 500"] - 1
    weekly["Forward 6m S&P return"] = weekly["S&P 500"].shift(-26) / weekly["S&P 500"] - 1
    weekly["Forward 12m S&P return"] = weekly["S&P 500"].shift(-52) / weekly["S&P 500"] - 1

    return weekly


core = load_core_data()
vixdata = load_vixeq_file()

st.write(f"Core data runs from **{core.dropna(how='all').index.min().date()}** to **{core.dropna(how='all').index.max().date()}**.")
st.write(f"VIXEQ data runs from **{vixdata.index.min().date()}** to **{vixdata.index.max().date()}**.")

dxy_returns = core["DXY"].pct_change()
spx_returns = core["S&P 500"].pct_change()
relative_returns = core["US relative performance"].pct_change()
weighted_spread_change = core["Weighted spread"].diff()

corr_52 = pd.DataFrame({
    "S&P 500 vs DXY": spx_returns.rolling(52).corr(dxy_returns),
    "US relative vs DXY": relative_returns.rolling(52).corr(dxy_returns),
    "DXY vs weighted yield differential": dxy_returns.rolling(52).corr(weighted_spread_change),
})

corr_260 = pd.DataFrame({
    "S&P 500 vs DXY": spx_returns.rolling(260).corr(dxy_returns),
    "US relative vs DXY": relative_returns.rolling(260).corr(dxy_returns),
    "DXY vs weighted yield differential": dxy_returns.rolling(260).corr(weighted_spread_change),
})

corr_520 = pd.DataFrame({
    "S&P 500 vs DXY": spx_returns.rolling(520).corr(dxy_returns),
    "US relative vs DXY": relative_returns.rolling(520).corr(dxy_returns),
    "DXY vs weighted yield differential": dxy_returns.rolling(520).corr(weighted_spread_change),
})


st.header("1. Rolling correlations")

plot_lines("1-year rolling correlations", corr_52)
plot_lines("5-year rolling correlations", corr_260)
plot_lines("10-year rolling correlations", corr_520)


st.header("2. Level charts")

dual_chart("DXY vs S&P 500", core["DXY"], "DXY", core["S&P 500"], "S&P 500")

dual_chart(
    "DXY vs US relative equity performance",
    core["DXY"],
    "DXY",
    core["US relative performance"],
    "S&P 500 / EFA",
)

dual_chart(
    "DXY vs weighted 10-year rate differential",
    core["DXY"],
    "DXY",
    core["Weighted spread"],
    "Weighted spread",
)


st.header("3. VIXEQ vs VIX dispersion signal")

dual_chart("VIXEQ vs VIX", vixdata["VIXEQ"], "VIXEQ", vixdata["VIX"], "VIX")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(vixdata.index, vixdata["VIXEQ minus VIX"])
ax.axhline(0, linestyle="--")
ax.set_title("VIXEQ minus VIX")
ax.set_ylabel("Vol points")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(vixdata.index, vixdata["VIXEQ / VIX"])
ax.axhline(1, linestyle="--")
ax.set_title("VIXEQ / VIX")
ax.set_ylabel("Ratio")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(vixdata.index, vixdata["VIXEQ/VIX z-score"])
ax.axhline(0, linestyle="--")
ax.axhline(1, linestyle="--")
ax.axhline(-1, linestyle="--")
ax.set_title("VIXEQ/VIX 1-year rolling z-score")
ax.set_ylabel("Z-score")
st.pyplot(fig)


st.header("4. VIXEQ signal vs forward S&P 500 returns")

scatter_chart(
    "VIXEQ / VIX vs forward 12m return",
    vixdata["VIXEQ / VIX"],
    vixdata["Forward 12m S&P return"],
    "VIXEQ / VIX",
    "Forward 12m return",
)

scatter_chart(
    "VIXEQ minus VIX vs forward 12m return",
    vixdata["VIXEQ minus VIX"],
    vixdata["Forward 12m S&P return"],
    "VIXEQ minus VIX",
    "Forward 12m return",
)


st.header("5. VIXEQ quintile analysis")

test = vixdata[
    [
        "VIXEQ minus VIX",
        "VIXEQ / VIX",
        "VIXEQ/VIX z-score",
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].dropna()

test["Ratio quintile"] = pd.qcut(
    test["VIXEQ / VIX"],
    5,
    labels=["Lowest", "Low", "Middle", "High", "Highest"],
    duplicates="drop",
)

ratio_table = test.groupby("Ratio quintile")[
    [
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].mean()

st.subheader("Average forward returns by VIXEQ/VIX ratio quintile")
st.dataframe(ratio_table.style.format("{:.2%}"))


st.header("6. SKEW case study")

st.write(
    "Signal: SKEW z-score below -0.5, SKEW falling over 13 weeks, "
    "and S&P 500 firmly higher: 26-week return above 10% and above its 40-week moving average."
)

skew_case = core[
    [
        "SKEW",
        "SKEW z-score",
        "SKEW 13w change",
        "SPX 26w return",
        "SPX above 40w MA",
        "SPX firmly higher",
        "SKEW low + falling + SPX uptrend",
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].dropna()

signal = skew_case[skew_case["SKEW low + falling + SPX uptrend"]]

st.subheader("SKEW z-score vs forward 12m return")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    skew_case["SKEW z-score"],
    skew_case["Forward 12m S&P return"],
    alpha=0.35,
    label="All observations",
)

ax.scatter(
    signal["SKEW z-score"],
    signal["Forward 12m S&P return"],
    alpha=0.9,
    label="Signal observations",
)

ax.axhline(0, linestyle="--")
ax.axvline(-0.5, linestyle="--")
ax.set_xlabel("SKEW z-score")
ax.set_ylabel("Forward 12m S&P 500 return")
ax.set_title("SKEW z-score vs forward 12m return")
ax.legend()
st.pyplot(fig)

st.subheader("Signal-only scatter: SKEW z-score vs forward returns")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(signal["SKEW z-score"], signal["Forward 3m S&P return"], alpha=0.7, label="3m")
ax.scatter(signal["SKEW z-score"], signal["Forward 6m S&P return"], alpha=0.7, label="6m")
ax.scatter(signal["SKEW z-score"], signal["Forward 12m S&P return"], alpha=0.7, label="12m")
ax.axhline(0, linestyle="--")
ax.axvline(-0.5, linestyle="--")
ax.set_xlabel("SKEW z-score")
ax.set_ylabel("Forward S&P 500 return")
ax.set_title("Forward returns after low/falling SKEW + strong SPX trend")
ax.legend()
st.pyplot(fig)

event_table = pd.DataFrame({
    "All observations": skew_case[
        ["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]
    ].mean(),
    "SKEW z < -0.5": skew_case[skew_case["SKEW z-score"] < -0.5][
        ["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]
    ].mean(),
    "SKEW z < -0.5 + falling": skew_case[
        (skew_case["SKEW z-score"] < -0.5)
        & (skew_case["SKEW 13w change"] < 0)
    ][
        ["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]
    ].mean(),
    "SKEW low/falling + SPX uptrend": signal[
        ["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]
    ].mean(),
})

st.subheader("Forward return comparison")
st.dataframe(event_table.style.format("{:.2%}"))

st.subheader("Number of observations")
st.write({
    "All observations": len(skew_case),
    "SKEW low/falling + SPX uptrend": len(signal),
})


st.header("7. Latest values")

st.dataframe(core.tail(20))