import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="VIXEQ, SKEW and Breadth Signal", layout="wide")

st.title("VIXEQ/VIX, SKEW, Breadth and Forward Equity Returns")

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

    if df.empty:
        st.warning(f"No data available for {title}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    ax.axhline(0, linestyle="--")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def dual_chart(title, left, left_label, right, right_label):
    st.subheader(title)

    df = pd.concat([left, right], axis=1).dropna()

    if df.empty:
        st.warning(f"No data available for {title}")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df.index, df.iloc[:, 0])
    ax1.set_ylabel(left_label)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df.iloc[:, 1])
    ax2.set_ylabel(right_label)

    ax1.set_title(title)
    st.pyplot(fig)


@st.cache_data
def load_core_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")
    spx = get_yahoo_close("^GSPC", "S&P 500")
    spy = get_yahoo_close("SPY", "SPY")
    rsp = get_yahoo_close("RSP", "RSP")
    efa = get_yahoo_close("EFA", "MSCI EAFE ETF")
    skew = get_yahoo_close("^SKEW", "SKEW")

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat(
        [dxy, spx, spy, rsp, efa, skew, us10, de10, jp10, uk10],
        axis=1
    ).sort_index()

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

    # SKEW signal
    weekly["SKEW z-score"] = (
        weekly["SKEW"] - weekly["SKEW"].rolling(52).mean()
    ) / weekly["SKEW"].rolling(52).std()

    weekly["SKEW 13w change"] = weekly["SKEW"] - weekly["SKEW"].shift(13)
    weekly["SKEW falling"] = weekly["SKEW 13w change"] < 0

    # Equity trend
    weekly["SPX 26w return"] = weekly["S&P 500"] / weekly["S&P 500"].shift(26) - 1
    weekly["SPX 40w MA"] = weekly["S&P 500"].rolling(40).mean()
    weekly["SPX above 40w MA"] = weekly["S&P 500"] > weekly["SPX 40w MA"]
    weekly["SPX rallying"] = (
        (weekly["SPX 26w return"] > 0.10)
        & weekly["SPX above 40w MA"]
    )

    # Breadth proxy: equal-weight S&P vs cap-weight S&P
    weekly["Breadth ratio RSP/SPY"] = weekly["RSP"] / weekly["SPY"]
    weekly["Breadth 13w change"] = (
        weekly["Breadth ratio RSP/SPY"] - weekly["Breadth ratio RSP/SPY"].shift(13)
    )
    weekly["Breadth 26w change"] = (
        weekly["Breadth ratio RSP/SPY"] - weekly["Breadth ratio RSP/SPY"].shift(26)
    )

    weekly["Breadth deteriorating"] = (
        (weekly["Breadth 13w change"] < 0)
        & (weekly["Breadth 26w change"] < 0)
    )

    return weekly


core = load_core_data()
vixdata = load_vixeq_file()

case = pd.concat(
    [
        vixdata[
            [
                "VIXEQ",
                "VIX",
                "VIXEQ / VIX",
                "VIXEQ minus VIX",
                "VIXEQ/VIX z-score",
                "Forward 3m S&P return",
                "Forward 6m S&P return",
                "Forward 12m S&P return",
            ]
        ],
        core[
            [
                "DXY",
                "S&P 500",
                "SKEW",
                "SKEW z-score",
                "SKEW 13w change",
                "SKEW falling",
                "SPX 26w return",
                "SPX above 40w MA",
                "SPX rallying",
                "Breadth ratio RSP/SPY",
                "Breadth 13w change",
                "Breadth 26w change",
                "Breadth deteriorating",
                "Weighted spread",
                "US relative performance",
            ]
        ],
    ],
    axis=1
).dropna()


st.sidebar.header("Signal settings")

vixeq_ratio_threshold = st.sidebar.slider(
    "VIXEQ/VIX threshold",
    min_value=1.5,
    max_value=3.0,
    value=2.4,
    step=0.05,
)

skew_z_threshold = st.sidebar.slider(
    "SKEW z-score threshold",
    min_value=-2.5,
    max_value=0.0,
    value=-1.0,
    step=0.1,
)

spx_26w_threshold = st.sidebar.slider(
    "SPX 26-week return threshold",
    min_value=0.00,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

case["VIXEQ/VIX high"] = case["VIXEQ / VIX"] > vixeq_ratio_threshold
case["SKEW low"] = case["SKEW z-score"] < skew_z_threshold
case["SPX rallying custom"] = (
    (case["SPX 26w return"] > spx_26w_threshold)
    & case["SPX above 40w MA"]
)

case["Combined hedge signal"] = (
    case["VIXEQ/VIX high"]
    & case["SKEW low"]
    & case["SKEW falling"]
    & case["SPX rallying custom"]
    & case["Breadth deteriorating"]
)

signal = case[case["Combined hedge signal"]]


st.write(f"Core data runs from **{core.dropna(how='all').index.min().date()}** to **{core.dropna(how='all').index.max().date()}**.")
st.write(f"VIXEQ data runs from **{vixdata.index.min().date()}** to **{vixdata.index.max().date()}**.")
st.write(f"Combined hedge-signal observations: **{len(signal)}** out of **{len(case)}** weekly observations.")


st.header("1. Macro rolling correlations")

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

plot_lines("1-year rolling correlations", corr_52)
plot_lines("5-year rolling correlations", corr_260)


st.header("2. Breadth and dispersion charts")

dual_chart(
    "VIXEQ/VIX vs S&P 500",
    case["VIXEQ / VIX"],
    "VIXEQ / VIX",
    case["S&P 500"],
    "S&P 500",
)

dual_chart(
    "Breadth proxy: RSP/SPY vs S&P 500",
    case["Breadth ratio RSP/SPY"],
    "RSP / SPY",
    case["S&P 500"],
    "S&P 500",
)

dual_chart(
    "VIXEQ/VIX vs Breadth proxy",
    case["VIXEQ / VIX"],
    "VIXEQ / VIX",
    case["Breadth ratio RSP/SPY"],
    "RSP / SPY",
)


st.header("3. Combined hedge-signal scatter plots")

st.subheader("VIXEQ/VIX vs forward 12m S&P 500 return")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    case["VIXEQ / VIX"],
    case["Forward 12m S&P return"],
    alpha=0.35,
    label="All observations",
)
ax.scatter(
    signal["VIXEQ / VIX"],
    signal["Forward 12m S&P return"],
    alpha=0.95,
    label="Combined hedge signal",
)
ax.axhline(0, linestyle="--")
ax.axvline(vixeq_ratio_threshold, linestyle="--")
ax.set_xlabel("VIXEQ / VIX")
ax.set_ylabel("Forward 12m S&P 500 return")
ax.set_title("VIXEQ/VIX vs forward 12m return")
ax.legend()
st.pyplot(fig)


st.subheader("SKEW z-score vs forward 12m S&P 500 return")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    case["SKEW z-score"],
    case["Forward 12m S&P return"],
    alpha=0.35,
    label="All observations",
)
ax.scatter(
    signal["SKEW z-score"],
    signal["Forward 12m S&P return"],
    alpha=0.95,
    label="Combined hedge signal",
)
ax.axhline(0, linestyle="--")
ax.axvline(skew_z_threshold, linestyle="--")
ax.set_xlabel("SKEW z-score")
ax.set_ylabel("Forward 12m S&P 500 return")
ax.set_title("SKEW z-score vs forward 12m return")
ax.legend()
st.pyplot(fig)


st.subheader("Breadth deterioration vs forward 12m return")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    case["Breadth 26w change"],
    case["Forward 12m S&P return"],
    alpha=0.35,
    label="All observations",
)
ax.scatter(
    signal["Breadth 26w change"],
    signal["Forward 12m S&P return"],
    alpha=0.95,
    label="Combined hedge signal",
)
ax.axhline(0, linestyle="--")
ax.axvline(0, linestyle="--")
ax.set_xlabel("26-week change in RSP/SPY")
ax.set_ylabel("Forward 12m S&P 500 return")
ax.set_title("Breadth deterioration vs forward 12m return")
ax.legend()
st.pyplot(fig)


st.subheader("Signal-only scatter: VIXEQ/VIX vs forward returns")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(signal["VIXEQ / VIX"], signal["Forward 3m S&P return"], alpha=0.75, label="3m")
ax.scatter(signal["VIXEQ / VIX"], signal["Forward 6m S&P return"], alpha=0.75, label="6m")
ax.scatter(signal["VIXEQ / VIX"], signal["Forward 12m S&P return"], alpha=0.75, label="12m")
ax.axhline(0, linestyle="--")
ax.axvline(vixeq_ratio_threshold, linestyle="--")
ax.set_xlabel("VIXEQ / VIX")
ax.set_ylabel("Forward S&P 500 return")
ax.set_title("Forward returns when hedge signal is active")
ax.legend()
st.pyplot(fig)


st.header("4. Forward return comparison")

return_cols = [
    "Forward 3m S&P return",
    "Forward 6m S&P return",
    "Forward 12m S&P return",
]

summary_table = pd.DataFrame({
    "All observations": case[return_cols].mean(),
    "VIXEQ/VIX high": case[case["VIXEQ/VIX high"]][return_cols].mean(),
    "SKEW low + falling": case[case["SKEW low"] & case["SKEW falling"]][return_cols].mean(),
    "SPX rallying": case[case["SPX rallying custom"]][return_cols].mean(),
    "Breadth deteriorating": case[case["Breadth deteriorating"]][return_cols].mean(),
    "Combined hedge signal": signal[return_cols].mean(),
})

median_table = pd.DataFrame({
    "All observations": case[return_cols].median(),
    "Combined hedge signal": signal[return_cols].median(),
})

hit_rate_table = pd.DataFrame({
    "All observations": (case[return_cols] > 0).mean(),
    "Combined hedge signal": (signal[return_cols] > 0).mean(),
})

count_table = pd.DataFrame({
    "Observation count": {
        "All observations": len(case),
        "VIXEQ/VIX high": len(case[case["VIXEQ/VIX high"]]),
        "SKEW low + falling": len(case[case["SKEW low"] & case["SKEW falling"]]),
        "SPX rallying": len(case[case["SPX rallying custom"]]),
        "Breadth deteriorating": len(case[case["Breadth deteriorating"]]),
        "Combined hedge signal": len(signal),
    }
})

st.subheader("Average forward returns")
st.dataframe(summary_table.style.format("{:.2%}"))

st.subheader("Median forward returns")
st.dataframe(median_table.style.format("{:.2%}"))

st.subheader("Positive-return hit rate")
st.dataframe(hit_rate_table.style.format("{:.2%}"))

st.subheader("Observation counts")
st.dataframe(count_table)


st.header("5. Signal dates")

signal_dates = signal[
    [
        "VIXEQ / VIX",
        "VIXEQ/VIX z-score",
        "SKEW",
        "SKEW z-score",
        "SKEW 13w change",
        "SPX 26w return",
        "Breadth ratio RSP/SPY",
        "Breadth 13w change",
        "Breadth 26w change",
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].copy()

st.dataframe(signal_dates.style.format({
    "VIXEQ / VIX": "{:.2f}",
    "VIXEQ/VIX z-score": "{:.2f}",
    "SKEW": "{:.2f}",
    "SKEW z-score": "{:.2f}",
    "SKEW 13w change": "{:.2f}",
    "SPX 26w return": "{:.2%}",
    "Breadth ratio RSP/SPY": "{:.2f}",
    "Breadth 13w change": "{:.2%}",
    "Breadth 26w change": "{:.2%}",
    "Forward 3m S&P return": "{:.2%}",
    "Forward 6m S&P return": "{:.2%}",
    "Forward 12m S&P return": "{:.2%}",
}))


st.header("6. Latest values")

latest_cols = [
    "VIXEQ",
    "VIX",
    "VIXEQ / VIX",
    "VIXEQ/VIX z-score",
    "SKEW",
    "SKEW z-score",
    "SKEW 13w change",
    "SPX 26w return",
    "SPX above 40w MA",
    "Breadth ratio RSP/SPY",
    "Breadth 13w change",
    "Breadth 26w change",
    "Combined hedge signal",
]

st.dataframe(case[latest_cols].tail(20))