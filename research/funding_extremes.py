"""Hypothesis F: fade funding extremes, pre-registered 25 Sep 2026 before running.

Mechanism claimed: when perp funding is extreme, one side is crowded and paying to hold; the crowded side unwinds
over the following days (high carry precedes crashes: Schmeling, Schrimpf and Todorov, 2023).
Signal at each Binance funding settlement, on the 8h-equivalent rate (rate x 8 / interval hours):
  extreme high = top 5% of the trailing 90 days and at least 0.02% per 8h (twice the 0.01% base rate) -> short;
  extreme low = bottom 5% of the trailing 90 days and below zero -> long.
Trade: taker entry at the open of the first 4h bar starting at or after the settlement, plus 0.02% slippage.
  Stop 1.0 x daily ATR(14) from entry; target 2R (maker); exit at market after 5 days. One position per symbol.
Execution on 4h bars, stop first when a bar spans stop and target.
Data: funding for BTC and ETH, 2020-2023 (github.com/supervik/historical-funding-rates-fetcher) is the design set;
  funding for BTC, ETH, SOL, XRP, BNB, AVAX, DOGE and LINK, Aug 2025 to Sep 2026
  (github.com/ZuShen168/funding_rate_data) is the held-out set. Prices: Bitstamp BTC; Binance spot 4h for the rest.
Gate: design t >= 3, the same sign in 2020-21 and 2022-23, held-out mean above zero; held-out run once.
Diagnostics, not traded: sign-adjusted forward returns 1, 3 and 5 days after each signal.

Usage: SLEEVE_DATA=/path/to/data python funding_extremes.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("SLEEVE_DATA", Path(__file__).resolve().parent.parent / "data"))
MAKER, TAKER, SLIP = 0.0002, 0.0006, 0.0002
PCT, FLOOR, WINDOW_D, STOP_ATR, TARGET_R, HOLD_BARS = 0.05, 0.0002, 90, 1.0, 2.0, 30
HELD = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT")


def funding() -> pd.DataFrame:
    rows = []
    for sym in ("BTC", "ETH"):
        d = pd.read_csv(DATA / f"supervik_{sym}_binance.csv")
        # timestamps in this file are UTC+3 (00:00 UTC settlements appear as 03:00)
        rows.append(pd.DataFrame({"sym": f"{sym}USDT", "t": pd.to_datetime(d.Date) - pd.Timedelta(hours=3), "r": d["Funding Rate"].astype(float), "set": "design"}))
    z = pd.read_parquet(DATA / "funding_binance.parquet")
    z = z[z.venue_symbol.isin(HELD) & ~z.is_predicted]
    rows.append(pd.DataFrame({"sym": z.venue_symbol, "t": z.settlement_ts.dt.tz_localize(None).dt.floor("min"),
                              "r": z.rate_raw * 8 / z.interval_hours, "set": "held-out"}))
    return pd.concat(rows).sort_values(["sym", "t"]).reset_index(drop=True)


def prices(sym: str) -> pd.DataFrame:
    if sym == "BTCUSDT":
        b = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
        return b.resample("4h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return pd.read_csv(DATA / f"{sym}_4h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)


def signals(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sym, st), g in f.groupby(["sym", "set"]):
        g = g.set_index("t").sort_index()
        hi = g.r.rolling(f"{WINDOW_D}D").quantile(1 - PCT); lo = g.r.rolling(f"{WINDOW_D}D").quantile(PCT)
        full = g.index >= g.index[0] + pd.Timedelta(days=WINDOW_D)
        side = np.where((g.r >= hi) & (g.r >= FLOOR) & full, -1, np.where((g.r <= lo) & (g.r < 0) & full, 1, 0))
        s = g[side != 0].assign(side=side[side != 0], sym=sym, set=st)
        out.append(s.reset_index())
    return pd.concat(out)


def trade(sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades, fwd = [], []
    for sym, g in sig.groupby("sym"):
        p = prices(sym); O, H, L, C = (p[k].values for k in ("o", "h", "l", "c"))
        d = p.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
        tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)  # known before the day starts
        busy_until = pd.Timestamp.min
        for row in g.itertuples():
            k = int(p.index.searchsorted(row.t))
            if k >= len(p) - 2:
                continue
            side = int(row.side); e0 = O[k]
            for days in (1, 3, 5):
                j = min(k + 6 * days, len(C) - 1)
                fwd.append(dict(sym=sym, set=row.set, days=days, ret=side * (C[j] / e0 - 1) * 100))
            if p.index[k] < busy_until:
                continue
            a = atr.asof(p.index[k])
            if not np.isfinite(a):
                continue
            entry = e0 * (1 + side * SLIP); stop = entry - side * STOP_ATR * a; risk = STOP_ATR * a
            tgt = entry + side * TARGET_R * risk; R = None; end = min(k + HOLD_BARS, len(C) - 1); q = k
            for q in range(k, end + 1):
                if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                    R, how = -1 - (TAKER + TAKER + SLIP) * entry / risk, "stop"; break
                if (side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt):
                    R, how = TARGET_R - (TAKER + MAKER) * entry / risk, "target"; break
            if R is None:
                R, how = side * (C[q] - entry) / risk - (TAKER + TAKER + SLIP) * entry / risk, "time"
            trades.append(dict(sym=sym, set=row.set, t=p.index[k], side=side, R=R, how=how, stop_pct=risk / entry * 100, rate=row.r))
            busy_until = p.index[q]
    return pd.DataFrame(trades), pd.DataFrame(fwd)


def summ(x: pd.Series) -> str:
    x = x.dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


def main() -> None:
    f = funding(); sig = signals(f); T, F = trade(sig)
    T.to_csv(Path(__file__).with_name("funding_trades.csv"), index=False)
    d = T[T.set == "design"]; ho = T[T.set == "held-out"]
    print(f"signals: {len(sig)} ({(sig.side == -1).sum()} short, {(sig.side == 1).sum()} long)")
    print(f"design {summ(d.R)} | 2020-21 {summ(d[d.t < '2022-01-01'].R)} / 2022-23 {summ(d[d.t >= '2022-01-01'].R)} | held-out {summ(ho.R)}")
    print(f"   shorts {summ(d[d.side == -1].R)} / {summ(ho[ho.side == -1].R)} | longs {summ(d[d.side == 1].R)} / {summ(ho[ho.side == 1].R)}")
    print(f"   win rate {(T.how == 'target').mean() * 100:.1f}% | median stop {T.stop_pct.median():.2f}%")
    print("   held-out by symbol: " + " | ".join(f"{s[:-4]} {summ(ho[ho.sym == s].R)}" for s in HELD))
    for st in ("design", "held-out"):
        print(f"forward return, sign-adjusted, {st}: " + " | ".join(
            f"{k}d {summ(F[(F.set == st) & (F.days == k)].ret)}%" for k in (1, 3, 5)))


if __name__ == "__main__":
    main()
