"""US stock day-trading tests on 5-minute bars, pre-registered 25 Sep 2026 before running (owner request: what about stocks).

Data: github.com/piekstra/market-data, 5-minute candles for leveraged ETFs, used as intraday proxies for the Bitunix
stock perps: TQQQ (3 x QQQ), SPXL (3 x SPY), SOXL (3 x semiconductors), NVDX (2 x NVDA), TSLT (2 x TSLA),
METU (2 x META), AMZU (2 x AMZN), GGLL (2 x GOOGL). Regular session only (09:30 to 16:00 New York).
R is scale-free, so the leverage factor matters only for costs: each cost is charged on the underlying's move
(ETF move / factor), at Bitunix stock-perp rates: maker 0.02%, taker 0.06%, slippage 0.03% (thinner books).
Sets: core TQQQ, SPXL, SOXL, design 2020-07 to 2023-12, held out 2024-01 to 2026-02; fresh = the five single-stock
2x ETFs, all dates (they start 2022 to 2024).
S1 ORB-5 (Zarattini and Aziz, 2023): first 5m candle's direction; enter at 09:35 (taker), stop at its opposite
   extreme, target 10R, else exit at 16:00. Skip stops under 0.05% of the underlying.
S2 ORB-15: the same on the first 15 minutes, entry at 09:45.
S3 Intraday momentum (Gao, Han, Li and Zhou, 2018): the sign of the return from the prior 16:00 close to 10:00 sets
   the side of a 15:30 to 16:00 trade, taker both ways. Net % of the underlying per trade.
S4 Overnight drift (Lou, Polk and Skouras, 2019): long 16:00 close to 09:30 open, maker both ways, funding 0.02% a
   night charged. Net % per night; the regular session (09:30 to 16:00) printed alongside for reference.
S5 MAX-10 on regular-session daily bars: long at the next 09:30 open (maker) after a close at or above the prior 10
   closes, stop 1 daily ATR(14), exit at the following 09:30 open (taker), rolling while the signal repeats.
S6 Trend ensemble long (trend_ensemble.py T1 rules, lookbacks 5 to 250 given the history), and S7 its short mirror.
Gate as elsewhere: design t >= 3, the same sign in both halves, held-out and fresh above zero.

Usage: STOCKS=/path/to/piekstra/data python stocks_intraday.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.environ.get("STOCKS", "/tmp/pmd/data"))
FACTOR = {"TQQQ": 3, "SPXL": 3, "SOXL": 3, "NVDX": 2, "TSLT": 2, "METU": 2, "AMZU": 2, "GGLL": 2}
CORE, FRESH = ("TQQQ", "SPXL", "SOXL"), ("NVDX", "TSLT", "METU", "AMZU", "GGLL")
MAKER, TAKER, SLIP, FUND_NIGHT = 0.0002, 0.0006, 0.0003, 0.0002
HELD = pd.Timestamp("2024-01-01")


def load(sym: str) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted((SRC / sym).rglob("*.parquet"))]
    d = pd.concat(parts)
    d["t"] = d.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    d = d.set_index("t")[["open", "high", "low", "close"]].astype(float)
    d.columns = ["o", "h", "l", "c"]
    d = d[~d.index.duplicated()].sort_index()
    tod = d.index.hour * 60 + d.index.minute
    return d[(tod >= 570) & (tod < 960)]  # 09:30 to 15:55 bar starts


def summ(x) -> str:
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


def halves(d: pd.DataFrame, col: str) -> str:
    mid = d.t.min() + (d.t.max() - d.t.min()) / 2
    return f"{summ(d[d.t < mid][col])} / {summ(d[d.t >= mid][col])}"


def orb(sym: str, b: pd.DataFrame, minutes: int) -> list[dict]:
    L = FACTOR[sym]; n = minutes // 5; out = []
    for day, g in b.groupby(b.index.normalize()):
        if len(g) < 70 or g.index[0].hour * 60 + g.index[0].minute != 570:
            continue
        o, c = g.o.iat[0], g.c.iat[n - 1]; hi, lo = g.h.iloc[:n].max(), g.l.iloc[:n].min()
        if c == o:
            continue
        side = 1 if c > o else -1; entry = g.o.iat[n] * (1 + side * SLIP * L); stop = lo if side == 1 else hi
        risk = side * (entry - stop); stop_u = risk / entry / L
        if risk <= 0 or stop_u < 0.0005:
            continue
        tgt = entry + side * 10 * risk; R = None
        cost = lambda f: f / stop_u  # fee fraction of the underlying, in R
        for q in range(n, len(g)):
            if (side == 1 and g.l.iat[q] <= stop) or (side == -1 and g.h.iat[q] >= stop):
                R = -1 - cost(TAKER + TAKER + SLIP); break
            if q > n and ((side == 1 and g.h.iat[q] >= tgt) or (side == -1 and g.l.iat[q] <= tgt)):
                R = 10 - cost(TAKER + MAKER); break
        if R is None:
            R = side * (g.c.iat[-1] - entry) / risk - cost(TAKER + TAKER + SLIP)
        out.append(dict(sym=sym, t=g.index[n], side=side, R=R, stop_u=stop_u * 100))
    return out


def daily(b: pd.DataFrame) -> pd.DataFrame:
    return b.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def intraday_momentum_and_overnight(sym: str, b: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    L = FACTOR[sym]; mom, night = [], []; prev_close = None; prev_day = None
    for day, g in b.groupby(b.index.normalize()):
        tod = g.index.hour * 60 + g.index.minute
        if len(g) < 70 or tod[0] != 570 or tod[-1] != 955:
            prev_close = None; continue
        if prev_close is not None:
            r_on = (g.o.iat[0] / prev_close - 1) / L
            night.append(dict(sym=sym, t=g.index[0], net=(r_on - 2 * MAKER - FUND_NIGHT) * 100, session=(g.c.iat[-1] / g.o.iat[0] - 1) / L * 100))
            c10 = g.c[tod == 595]
            k1530 = np.nonzero(tod == 930)[0]
            if len(c10) and len(k1530):
                side = 1 if c10.iat[0] > prev_close else -1
                r = side * (g.c.iat[-1] / g.o.iat[k1530[0]] - 1) / L
                mom.append(dict(sym=sym, t=g.index[0], net=(r - 2 * (TAKER + SLIP)) * 100))
        prev_close = g.c.iat[-1]
    return mom, night


def max10(sym: str, d: pd.DataFrame) -> list[dict]:
    L = FACTOR[sym]; dc = d.c.values; O = d.o.values; lo = d.l.values
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values
    sig = dc >= pd.Series(dc).rolling(10).max().shift(1).values; out = []; i = 15
    while i < len(d) - 2:
        if not sig[i]:
            i += 1; continue
        entry = O[i + 1]; first = True
        while True:
            stop = entry - atr[i]; stop_u = atr[i] / entry / L; j = i + 1
            hit = lo[j] <= stop
            ex = min(stop, O[j]) * (1 - SLIP * L) if hit and O[j] > stop else (O[j] * (1 - SLIP * L) if hit else None)
            # stop checked within day j (the entry day); exit otherwise at the next open
            roll = (not hit) and i + 1 < len(d) - 2 and sig[i + 1]
            if ex is None:
                ex = O[j + 1] if roll else O[j + 1] * (1 - SLIP * L)
            cost = ((MAKER if first else 0.0) + (0.0 if roll else TAKER)) / stop_u
            out.append(dict(sym=sym, t=d.index[j], R=(ex - entry) / atr[i] - cost - FUND_NIGHT / stop_u))
            i += 1
            if not roll:
                break
            entry, first = ex, False
        i += 1
    return out


def trend(sym: str, d: pd.DataFrame, side: int) -> list[dict]:
    L = FACTOR[sym]; dc = d.c.values; O, H, Lw = d.o.values, d.h.values, d.l.values; out = []
    for N in (5, 10, 20, 30, 60, 90, 150, 250):
        ext = (pd.Series(dc).rolling(N).max() if side == 1 else pd.Series(dc).rolling(N).min()).shift(1).values
        mid = ((pd.Series(dc).rolling(N).max() + pd.Series(dc).rolling(N).min()) / 2).values; i = N + 1
        while i < len(d) - 2:
            if not ((side == 1 and dc[i] >= ext[i]) or (side == -1 and dc[i] <= ext[i])):
                i += 1; continue
            entry = O[i + 1] * (1 + side * SLIP * L); stop = mid[i]; risk = side * (entry - stop)
            if risk <= 0 or risk / entry / L < 0.005:
                i += 1; continue
            stop_u = risk / entry / L; j = i + 1; ex = None
            while j < len(d) - 1:
                if (side == 1 and Lw[j] <= stop) or (side == -1 and H[j] >= stop):
                    ex = (min(stop, O[j]) if side == 1 else max(stop, O[j])) * (1 - side * SLIP * L); break
                stop = max(stop, mid[j]) if side == 1 else min(stop, mid[j]); j += 1
            if ex is None:
                ex = dc[j]
            nights = j - (i + 1)
            R = side * (ex - entry) / risk - 2 * TAKER / stop_u - (FUND_NIGHT * nights / stop_u if side == 1 else 0.0)
            out.append(dict(sym=sym, N=N, t=d.index[i + 1], R=R)); i = j + 1
    return out


def report(name: str, T: pd.DataFrame, col: str = "R") -> None:
    core = T[T.sym.isin(CORE)]; des = core[core.t < HELD]
    print(f"{name}: design {summ(des[col])} | halves {halves(des, col)} | held-out {summ(core[core.t >= HELD][col])} | fresh {summ(T[T.sym.isin(FRESH)][col])}")


def main() -> None:
    bars = {s: load(s) for s in FACTOR}
    for s, b in bars.items():
        print(f"{s}: {b.index[0].date()} to {b.index[-1].date()}, {b.index.normalize().nunique()} days", flush=True)
    for m in (5, 15):
        T = pd.DataFrame([r for s, b in bars.items() for r in orb(s, b, m)]); report(f"S{1 if m == 5 else 2} ORB-{m}", T)
        print(f"   longs {summ(T[T.side == 1].R)} | shorts {summ(T[T.side == -1].R)} | median stop {T.stop_u.median():.2f}% of the underlying")
    mom, night = [], []
    for s, b in bars.items():
        a, n = intraday_momentum_and_overnight(s, b); mom += a; night += n
    report("S3 intraday momentum, net % per trade", pd.DataFrame(mom), "net")
    N = pd.DataFrame(night); report("S4 overnight long, net % per night", N, "net")
    print(f"   regular session for reference (no costs): {summ(N.session)}")
    D = {s: daily(b) for s, b in bars.items()}
    report("S5 MAX-10", pd.DataFrame([r for s, d in D.items() for r in max10(s, d)]))
    report("S6 trend long", pd.DataFrame([r for s, d in D.items() for r in trend(s, d, 1)]))
    report("S7 trend short", pd.DataFrame([r for s, d in D.items() for r in trend(s, d, -1)]))


if __name__ == "__main__":
    main()
