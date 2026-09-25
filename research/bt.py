"""Backtest of the sleeve's three structures on hourly data resampled to 4h."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEE_MAKER, FEE_TAKER = 0.0002, 0.0006
MAX_HOLD = 30  # 4h bars, five days
TARGET_R = 3.0

def load() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted((ROOT / "arkanoeth_Binance_Future_Prices/Data").glob("*_20220101_*.csv")):
        d = pd.read_csv(f, usecols=["open_timestamp", "open", "high", "low", "close"])
        d.index = pd.to_datetime(d.open_timestamp, unit="ms"); out[f.name.split("_")[0] + "_22"] = d
    d = pd.read_csv(ROOT / "cryptobigbro_binance-BTCUSDT/binance-BTCUSDT-1h.csv")
    d.index = pd.to_datetime(d.open_timestamp_utc, unit="s"); out["BTC_17"] = d
    res = {}
    for k, d in out.items():
        d = d.sort_index()[["open", "high", "low", "close"]].astype(float)
        res[k] = d.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return res

def ind(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    delta = d.close.diff(); up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)
    d["hh20"] = d.high.rolling(20).max().shift(); d["ll20"] = d.low.rolling(20).min().shift()
    d["lo10"] = d.low.rolling(10).min().shift(); d["hi10"] = d.high.rolling(10).max().shift()
    w = (d.hh20 - d.ll20) / d.atr
    d["coil"] = w < w.rolling(200).quantile(0.2)
    return d

def resolve(d, i, side, entry, stop, fee_rt) -> tuple[float, int] | None:
    """Walk forward from bar i (entry bar). Stop first if both hit in one bar."""
    risk = abs(entry - stop)
    if risk <= 0: return None
    tgt = entry + side * TARGET_R * risk
    fee_r = fee_rt * entry / risk
    H, L, C = d.high.values, d.low.values, d.close.values
    for j in range(i, min(i + MAX_HOLD, len(d))):
        hit_s = L[j] <= stop if side == 1 else H[j] >= stop
        hit_t = H[j] >= tgt if side == 1 else L[j] <= tgt
        if hit_s: return -1 - fee_r, j
        if hit_t and j > i: return TARGET_R - fee_r, j
    j = min(i + MAX_HOLD, len(d)) - 1
    return side * (C[j] - entry) / risk - fee_r, j

def run(d: pd.DataFrame, name: str) -> list[dict]:
    d = ind(d); trades = []; busy = {}
    O, H, L, C = d.open.values, d.high.values, d.low.values, d.close.values
    A, E20, E50, R = d.atr.values, d.ema20.values, d.ema50.values, d.rsi.values
    for i in range(210, len(d) - 1):
        # 1 pullback in trend: maker limit at prior bar's EMA20, trend on prior bar
        for side in (1, -1):
            trend = (E20[i-1] > E50[i-1] and C[i-1] > E50[i-1]) if side == 1 else (E20[i-1] < E50[i-1] and C[i-1] < E50[i-1])
            lvl = E20[i-1]
            if trend and busy.get(("pb", side), -1) < i and ((side == 1 and L[i] <= lvl < O[i]) or (side == -1 and H[i] >= lvl > O[i])):
                stop = (d.lo10.values[i] - 0.25*A[i-1]) if side == 1 else (d.hi10.values[i] + 0.25*A[i-1])
                if (side == 1 and stop < lvl) or (side == -1 and stop > lvl):
                    r = resolve(d, i, side, lvl, stop, 2*FEE_MAKER)
                    if r: trades.append(dict(pair=name, t=d.index[i], setup="pullback", side=side, R=r[0], rsi=R[i-1])); busy[("pb", side)] = r[1]
        # 2 failed move: bar sweeps 20-bar extreme and closes back inside; maker entry next bar at close
        for side in (1, -1):
            swept = (L[i] < d.ll20.values[i] and C[i] > d.ll20.values[i]) if side == 1 else (H[i] > d.hh20.values[i] and C[i] < d.hh20.values[i])
            if swept and busy.get(("fm", side), -1) < i + 1:
                lvl = C[i]
                filled = (L[i+1] <= lvl) if side == 1 else (H[i+1] >= lvl)
                if not filled: continue
                stop = (L[i] - 0.25*A[i]) if side == 1 else (H[i] + 0.25*A[i])
                # regular RSI divergence at the sweep: price beyond 20-bar extreme, RSI not
                k = i - 20 + int(np.argmin(L[i-20:i])) if side == 1 else i - 20 + int(np.argmax(H[i-20:i]))
                div = (R[i] > R[k]) if side == 1 else (R[i] < R[k])
                r = resolve(d, i+1, side, lvl, stop, 2*FEE_MAKER)
                if r: trades.append(dict(pair=name, t=d.index[i+1], setup="failed_move", side=side, R=r[0], div=div, rsi=R[i])); busy[("fm", side)] = r[1]
        # 3 bracket of a coiled 20-bar range: stop entries (taker) both sides, stop at the opposite side
        if d.coil.values[i-1] and busy.get("br", -1) < i:
            hi, lo = d.hh20.values[i], d.ll20.values[i]
            for side, lvl, stop in ((1, hi, lo - 0.25*A[i-1]), (-1, lo, hi + 0.25*A[i-1])):
                if (side == 1 and H[i] >= hi) or (side == -1 and L[i] <= lo):
                    entry = max(lvl, O[i]) if side == 1 else min(lvl, O[i])
                    r = resolve(d, i, side, entry, stop, 2*FEE_TAKER)
                    if r: trades.append(dict(pair=name, t=d.index[i], setup="bracket", side=side, R=r[0])); busy["br"] = r[1]
                    break
    return trades

def summ(t: pd.DataFrame) -> pd.Series:
    w = t.R > 0
    return pd.Series(dict(n=len(t), win=w.mean(), expR=t.R.mean(), pf=t.R[w].sum() / -t.R[~w].sum() if (~w).any() else np.nan))

if __name__ == "__main__":
    data = load(); allt = []
    for k, d in data.items(): allt += run(d, k)
    t = pd.DataFrame(allt); t.to_csv(ROOT / "bt/trades.csv", index=False)
    t["era"] = np.where(t.pair == "BTC_17", "BTC 2017-19", "20 pairs 2022-23")
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
    print(t.groupby(["setup"]).apply(summ, include_groups=False))
    print(t.groupby(["setup", "era"]).apply(summ, include_groups=False))
    print(t.groupby(["setup", "side"]).apply(summ, include_groups=False))
    fm = t[t.setup == "failed_move"]; print(fm.groupby("div").apply(summ, include_groups=False))
    print(t[t.pair.str.startswith(("BTC", "ETH", "XRP", "LINK", "LTC"))].groupby(["setup"]).apply(summ, include_groups=False))
