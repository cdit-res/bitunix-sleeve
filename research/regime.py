"""Does a measured regime separate expectancy? Regime = BTC daily trend x asset volatility percentile, known before entry."""
import numpy as np, pandas as pd
from bt import load
from edge import summ

data = load()

def daily_regime(d4: pd.DataFrame) -> pd.DataFrame:
    dd = d4.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    e = dd.close.ewm(span=50, adjust=False).mean()
    trend = np.where((dd.close > e) & (e > e.shift(5)), "up", np.where((dd.close < e) & (e < e.shift(5)), "down", "flat"))
    tr = pd.concat([dd.high - dd.low, (dd.high - dd.close.shift()).abs(), (dd.low - dd.close.shift()).abs()], axis=1).max(axis=1)
    atrp = tr.ewm(alpha=1/14, adjust=False).mean() / dd.close
    vol = np.where(atrp > atrp.rolling(180, min_periods=60).median(), "hivol", "lovol")
    out = pd.DataFrame({"trend": trend, "vol": vol}, index=dd.index + pd.Timedelta(days=1))  # known next day
    return out

reg = {k: daily_regime(v) for k, v in data.items()}
t = pd.read_csv("edge_trades.csv", parse_dates=["t"])
t = t[(t["mode"] == "fixed3") & (t.stop_pct >= 0.5)]
def lookup(row):
    btc = reg["BTC_17"] if row.pair == "BTC_17" else reg["BTC_22"]
    own = reg[row.pair]
    b = btc.trend.asof(row.t) if row.t >= btc.index[0] else None
    v = own.vol.asof(row.t) if row.t >= own.index[0] else None
    return pd.Series({"btc": b, "vol": v})
t[["btc", "vol"]] = t.apply(lookup, axis=1)
t = t.dropna(subset=["btc", "vol"])
t["align"] = np.where(t.btc == "flat", "btc flat", np.where((t.btc == "up") == (t.side == 1), "with btc", "against btc"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
def tbl(keys):
    g = t.groupby(keys + ["era"]).apply(summ, include_groups=False)[["n", "expR"]].unstack("era")
    g.columns = [f"{a}_{b[:4]}" for a, b in g.columns]; return g.round(2).to_string()
print(tbl(["align", "vol"]))
print(tbl(["setup", "align"]))
print(tbl(["setup", "vol"]))
t.to_csv("regime_trades.csv", index=False)
