import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DXY, Equities, Yields and VIXEQ", layout="wide")

st.title("DXY, Equities, Yield Differentials and VIXEQ")

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


def dual_chart(title, left, left_label, right, right_label):
    st.subheader(title)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(left.index, left.values)
    ax1.set_ylabel(left_label)

    ax2 = ax1.twinx()
    ax2.plot(right.index, right.values)
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
def load_data():
    dxy = get_yahoo_close("DX-Y.NYB", "DXY")
    spx = get_yahoo_close("^GSPC", "S&P 500")
    efa = get_yahoo_close("EFA", "MSCI EAFE ETF")

    vix = get_yahoo_close("^VIX", "VIX")
    vixeq = get_yahoo_close("^VIXEQ", "VIXEQ")

    us10 = get_fred_series("DGS10", "US 10Y")
    de10 = get_fred_series("IRLTLT01DEM156N", "Germany 10Y")
    jp10 = get_fred_series("IRLTLT01JPM156N", "Japan 10Y")
    uk10 = get_fred_series("IRLTLT01GBM156N", "UK 10Y")

    data = pd.concat(
        [dxy, spx, efa, vix, vixeq, us10, de10, jp10, uk10],
        axis=1
    ).sort_index()

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

    data["VIXEQ minus VIX"] = data["VIXEQ"] - data["VIX"]
    data["VIXEQ / VIX"] = data["VIXEQ"] / data["VIX"]

    weekly = data.resample("W-FRI").last().dropna()

    dxy_returns = weekly["DXY"].pct_change()
    spx_returns = weekly["S&P 500"].pct_change()
    relative_returns = weekly["US relative performance"].pct_change()

    weighted_spread_change = weekly["Weighted spread"].diff()

    equity_corr_52 = pd.DataFrame({
        "S&P 500 vs DXY": spx_returns.rolling(52).corr(dxy_returns),
        "US relative vs DXY": relative_returns.rolling(52).corr(dxy_returns),
        "DXY vs weighted yield differential": dxy_returns.rolling(52).corr(weighted_spread_change),
    })

    equity_corr_260 = pd.DataFrame({
        "S&P 500 vs DXY": spx_returns.rolling(260).corr(dxy_returns),
        "US relative vs DXY": relative_returns.rolling(260).corr(dxy_returns),
        "DXY vs weighted yield differential": dxy_returns.rolling(260).corr(weighted_spread_change),
    })

    equity_corr_520 = pd.DataFrame({
        "S&P 500 vs DXY": spx_returns.rolling(520).corr(dxy_returns),
        "US relative vs DXY": relative_returns.rolling(520).corr(dxy_returns),
        "DXY vs weighted yield differential": dxy_returns.rolling(520).corr(weighted_spread_change),
    })

    # Forward equity returns
    weekly["Forward 3m S&P return"] = weekly["S&P 500"].shift(-13) / weekly["S&P 500"] - 1
    weekly["Forward 6m S&P return"] = weekly["S&P 500"].shift(-26) / weekly["S&P 500"] - 1
    weekly["Forward 12m S&P return"] = weekly["S&P 500"].shift(-52) / weekly["S&P 500"] - 1

    return weekly, equity_corr_52, equity_corr_260, equity_corr_520


data, corr_52, corr_260, corr_520 = load_data()

st.write(f"Data runs from **{data.index.min().date()}** to **{data.index.max().date()}**.")
st.write(f"VIXEQ data starts on **{data['VIXEQ'].dropna().index.min().date()}**.")
st.write("Equity foreign proxy: **EFA — iShares MSCI EAFE ETF**.")
st.write("Yield data: **FRED**. German, Japanese and UK yields are monthly and forward-filled.")

st.header("1. Rolling correlations")

plot_corr("1-year rolling correlations", corr_52)
plot_corr("5-year rolling correlations", corr_260)
plot_corr("10-year rolling correlations", corr_520)

st.header("2. DXY and equity level charts")

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
    "DXY vs weighted 10-year rate differential",
    data["DXY"],
    "DXY",
    data["Weighted spread"],
    "Weighted yield spread",
)

st.header("3. VIXEQ vs VIX dispersion signal")

dual_chart(
    "VIXEQ vs VIX",
    data["VIXEQ"],
    "VIXEQ",
    data["VIX"],
    "VIX",
)

st.subheader("VIXEQ minus VIX")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data["VIXEQ minus VIX"])
ax.axhline(0, linestyle="--")
ax.set_title("VIXEQ minus VIX")
ax.set_ylabel("Vol points")
st.pyplot(fig)

st.subheader("VIXEQ / VIX")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data["VIXEQ / VIX"])
ax.axhline(1, linestyle="--")
ax.set_title("VIXEQ / VIX")
ax.set_ylabel("Ratio")
st.pyplot(fig)

st.header("4. Does VIXEQ-VIX predict forward equity returns?")

scatter_chart(
    "VIXEQ minus VIX vs forward 3m S&P 500 return",
    data["VIXEQ minus VIX"],
    data["Forward 3m S&P return"],
    "VIXEQ minus VIX",
    "Forward 3m return",
)

scatter_chart(
    "VIXEQ minus VIX vs forward 6m S&P 500 return",
    data["VIXEQ minus VIX"],
    data["Forward 6m S&P return"],
    "VIXEQ minus VIX",
    "Forward 6m return",
)

scatter_chart(
    "VIXEQ minus VIX vs forward 12m S&P 500 return",
    data["VIXEQ minus VIX"],
    data["Forward 12m S&P return"],
    "VIXEQ minus VIX",
    "Forward 12m return",
)

scatter_chart(
    "VIXEQ / VIX vs forward 12m S&P 500 return",
    data["VIXEQ / VIX"],
    data["Forward 12m S&P return"],
    "VIXEQ / VIX",
    "Forward 12m return",
)

st.header("5. Quintile analysis")

test = data[
    [
        "VIXEQ minus VIX",
        "VIXEQ / VIX",
        "Forward 3m S&P return",
        "Forward 6m S&P return",
        "Forward 12m S&P return",
    ]
].dropna()

test["Gap quintile"] = pd.qcut(test["VIXEQ minus VIX"], 5, labels=[
    "Lowest gap",
    "Low gap",
    "Middle",
    "High gap",
    "Highest gap",
])

test["Ratio quintile"] = pd.qcut(test["VIXEQ / VIX"], 5, labels=[
    "Lowest ratio",
    "Low ratio",
    "Middle",
    "High ratio",
    "Highest ratio",
])

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

st.header("6. Latest values")

st.dataframe(data.tail(20))