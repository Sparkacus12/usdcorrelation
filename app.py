import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY, Equities, Yields and VIXEQ", layout="wide")

st.title("DXY, Equities, Yield Differentials and VIXEQ")

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
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

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

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat([dxy, spx, efa, us10, de10, jp10, uk10], axis=1).sort_index()

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

    return data.resample("W-FRI").last()


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
    "VIXEQ minus VIX vs forward 3m return",
    vixdata["VIXEQ minus VIX"],
    vixdata["Forward 3m S&P return"],
    "VIXEQ minus VIX",
    "Forward 3m return",
)

scatter_chart(
    "VIXEQ minus VIX vs forward 6m return",
    vixdata["VIXEQ minus VIX"],
    vixdata["Forward 6m S&P return"],
    "VIXEQ minus VIX",
    "Forward 6m return",
)

scatter_chart(
    "VIXEQ minus VIX vs forward 12m return",
    vixdata["VIXEQ minus VIX"],
    vixdata["Forward 12m S&P return"],
    "VIXEQ minus VIX",
    "Forward 12m return",
)

scatter_chart(
    "VIXEQ / VIX vs forward 12m return",
    vixdata["VIXEQ / VIX"],
    vixdata["Forward 12m S&P return"],
    "VIXEQ / VIX",
    "Forward 12m return",
)

st.header("5. Quintile analysis")

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

test["Gap quintile"] = pd.qcut(
    test["VIXEQ minus VIX"],
    5,
    labels=["Lowest", "Low", "Middle", "High", "Highest"],
    duplicates="drop",
)

test["Ratio quintile"] = pd.qcut(
    test["VIXEQ / VIX"],
    5,
    labels=["Lowest", "Low", "Middle", "High", "Highest"],
    duplicates="drop",
)

gap_table = test.groupby("Gap quintile")[
    [
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].mean()

ratio_table = test.groupby("Ratio quintile")[
    [
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].mean()

st.subheader("Average forward returns by VIXEQ-VIX gap quintile")
st.dataframe(gap_table.style.format("{:.2%}"))

st.subheader("Average forward returns by VIXEQ/VIX ratio quintile")
st.dataframe(ratio_table.style.format("{:.2%}"))

st.header("6. Extreme-event study")

low_gap = test[test["VIXEQ minus VIX"] <= test["VIXEQ minus VIX"].quantile(0.10)]
high_gap = test[test["VIXEQ minus VIX"] >= test["VIXEQ minus VIX"].quantile(0.90)]

low_ratio = test[test["VIXEQ / VIX"] <= test["VIXEQ / VIX"].quantile(0.10)]
high_ratio = test[test["VIXEQ / VIX"] >= test["VIXEQ / VIX"].quantile(0.90)]

extreme_table = pd.DataFrame({
    "Low gap": low_gap[["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]].mean(),
    "High gap": high_gap[["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]].mean(),
    "Low ratio": low_ratio[["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]].mean(),
    "High ratio": high_ratio[["Forward 3m S&P return", "Forward 6m S&P return", "Forward 12m S&P return"]].mean(),
})

st.dataframe(extreme_table.style.format("{:.2%}"))

st.header("7. Latest values")

st.dataframe(vixdata.tail(20))