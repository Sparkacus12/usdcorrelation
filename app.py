import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data
dxy = yf.download("DX-Y.NYB", auto_adjust=True, progress=False)[["Close"]]
spx = yf.download("^GSPC", auto_adjust=True, progress=False)[["Close"]]

# Weekly closes
dxy_w = dxy.resample("W-FRI").last().rename(columns={"Close":"DXY"})
spx_w = spx.resample("W-FRI").last().rename(columns={"Close":"SPX"})
df = dxy_w.join(spx_w, how="inner")

# Weekly returns
rets = df.pct_change().dropna()

# Rolling correlations
corr_52 = rets["DXY"].rolling(52).corr(rets["SPX"])
corr_260 = rets["DXY"].rolling(260).corr(rets["SPX"])  # ~5 years

# Chart 1: 1-year rolling correlation
plt.figure(figsize=(14,6))
plt.plot(corr_52.index, corr_52)
plt.axhline(0, linestyle="--")
plt.title("1-Year Rolling Correlation: DXY vs S&P 500 (Weekly Returns)")
plt.ylabel("Correlation")
plt.tight_layout()
plt.savefig("dxy_spx_corr_1yr.png")
plt.close()

# Chart 2: 5-year rolling correlation
plt.figure(figsize=(14,6))
plt.plot(corr_260.index, corr_260)
plt.axhline(0, linestyle="--")
plt.title("5-Year Rolling Correlation: DXY vs S&P 500 (Weekly Returns)")
plt.ylabel("Correlation")
plt.tight_layout()
plt.savefig("dxy_spx_corr_5yr.png")
plt.close()

# Chart 3: Regime chart
regime = corr_52.dropna()

x = regime.index.to_pydatetime()
y = regime.to_numpy()

plt.figure(figsize=(14,6))
plt.plot(x, y)

plt.fill_between(x, y, 0, where=(y >= 0), alpha=0.3)
plt.fill_between(x, y, 0, where=(y < 0), alpha=0.3)

plt.axhline(0, linestyle="--")
plt.title("Correlation Regimes: DXY vs S&P 500")
plt.ylabel("52-week Correlation")
plt.tight_layout()
plt.savefig("dxy_spx_regimes.png")
plt.close()

# Chart 4: Levels + correlation
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(df.index, df["SPX"], label="S&P 500")
ax1.set_ylabel("S&P 500")

ax2 = ax1.twinx()
ax2.plot(df.index, df["DXY"], label="DXY")
ax2.set_ylabel("DXY")

plt.title("DXY and S&P 500 Levels")
plt.tight_layout()
plt.savefig("dxy_spx_levels.png")
plt.close()

print("Files created.")
