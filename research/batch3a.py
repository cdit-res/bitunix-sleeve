"""Batch 3a, pre-registered 25 Sep 2026 before running.

W1: the daily trend effect expressed inside the two ticket windows, at leverage the constraints allow.
    Windows (UTC, 1h bars): morning 08:00 to 19:00 (the 08:30 run to the 20:00 run) and evening 19:00 to 08:00.
    Entry at the window's first open (maker, the run's limit), stop 0.25 x daily ATR(14) below entry (about 0.75%
    on BTC, so about 70x under the 55% rule), target 2R (maker), else exit at the window's end (taker).
    Funding 0.015% a window on longs. Long only.
    W1a: condition = MAX-10 state (last daily close at or above the prior 10 closes).
    W1b: condition = at least 5 of the 9 trend components long (trend_ensemble.py rules, 0.5% rail).
    Four variants counted (W1a/W1b x morning/evening). Unconditional longs printed as a control, not a variant.
    Sets: BTC (Bitstamp) design to 2023, ETH, SOL, XRP design 2022-23, all held out from 2024; six fresh coins 2022-26.
B1: attempts to disprove T03 (MAX-10) and T01 (trend ensemble):
    walk-forward choice of MAX-N lookback (best trailing 3-year mean R among 5, 8, 10, 12, 15, 20, 30, applied the
    next year); regime splits (BTC market trend at signal: up, flat, down; own volatility state; calendar year);
    double costs.

Usage: SLEEVE_DATA=/path/to/data python batch3a.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import max10_checks as MC  # noqa: E402  (runs its own report on import; output is labelled)
import remaining_lines as RL  # noqa: E402
import trend_ensemble as TE  # noqa: E402

DATA = TE.DATA; MAKER, TAKER, SLIP = TE.MAKER, TE.TAKER, TE.SLIP


def hourly(sym: str) -> pd.DataFrame:
    if sym == "BTC":
        return pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet").resample("1h").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return pd.read_csv(DATA / f"{sym}_1h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)


def states(sym: str) -> pd.DataFrame:
    """Daily MAX-10 state and count of long trend components, known after each 00:00 UTC close."""
    _, d = TE.bars(sym); c, lo = d.c.values, d.l.values
    mx10 = c >= pd.Series(c).rolling(10).max().shift(1).values
    count = np.zeros(len(c))
    for n in TE.LOOKBACKS:
        mx = pd.Series(c).rolling(n).max(); mn = pd.Series(c).rolling(n).min()
        prev = mx.shift(1).values; mid = ((mx + mn) / 2).values; long, stop = False, np.nan
        for t in range(n, len(c)):
            if long:
                if lo[t] <= stop:
                    long = False
                else:
                    stop = max(stop, mid[t])
            if not long and c[t] >= prev[t] and (c[t] - mid[t]) / c[t] >= 0.005:
                long, stop = True, mid[t]
            count[t] += long
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    s = pd.DataFrame({"max10": mx10, "count": count, "atr": tr.ewm(alpha=1 / 14, adjust=False).mean().values}, index=d.index)
    s.index = s.index + pd.Timedelta(days=1)  # known from the next day's 00:00 UTC
    return s


def windows(sym: str) -> list[dict]:
    h = hourly(sym); st = states(sym); O, H, L, C = (h[k].values for k in ("o", "h", "l", "c")); idx = h.index; out = []
    for day in st.index:
        s = st.loc[day]
        for lab, a, b in (("morning", 8, 19), ("evening", 19, 32)):
            t0, t1 = day + pd.Timedelta(hours=a), day + pd.Timedelta(hours=b)
            k0, k1 = idx.searchsorted(t0), idx.searchsorted(t1)
            if k1 >= len(idx) or k0 >= k1 or idx[k0] != t0:
                continue
            entry = O[k0]; risk = 0.25 * s.atr
            if not np.isfinite(risk) or risk <= 0:
                continue
            stop, tgt = entry - risk, entry + 2 * risk; R = None
            for q in range(k0, k1):
                if L[q] <= stop:
                    R = -1 - (MAKER + TAKER + SLIP) * entry / risk; break
                if H[q] >= tgt:
                    R = 2 - 2 * MAKER * entry / risk; break
            if R is None:
                R = (O[k1] * (1 - SLIP) - entry) / risk - (MAKER + TAKER) * entry / risk
            R -= 0.00015 * entry / risk
            out.append(dict(sym=sym, t=t0, window=lab, R=R, max10=bool(s.max10), trend=s["count"] >= 5,
                            stop_pct=risk / entry * 100))
    return out


def sets(T: pd.DataFrame):
    core = T[T.sym.isin(TE.CORE)]
    return core[core.t < TE.HELD_FROM], core[core.t >= TE.HELD_FROM], T[T.sym.isin(TE.FRESH)]


def run_w1() -> None:
    T = pd.DataFrame([r for s in TE.CORE + TE.FRESH for r in windows(s)])
    print(f"\nW1 windows: median stop {T.stop_pct.median():.2f}% (implied leverage at the 55% rule about {55 / T.stop_pct.median():.0f}x)")
    for cond in ("max10", "trend"):
        for w in ("morning", "evening"):
            g = T[(T.window == w) & T[cond]]; d, ho, fr = sets(g)
            print(f"W1{'a' if cond == 'max10' else 'b'} {w}: design {RL.summ(d.R)} | halves {RL.halves(d)} | held-out {RL.summ(ho.R)} | fresh {RL.summ(fr.R)}")
    for w in ("morning", "evening"):
        g = T[T.window == w]; d, ho, fr = sets(g)
        print(f"   control, all {w} longs: design {RL.summ(d.R)} | held-out {RL.summ(ho.R)} | fresh {RL.summ(fr.R)}")


def run_b1() -> None:
    print("\nB1: attempts to disprove MAX-10 (T03)")
    lbs = (5, 8, 10, 12, 15, 20, 30)
    allr = {n: MC.allrun(n=n) for n in lbs}
    for n in lbs:
        allr[n]["yr"] = allr[n].t.dt.year
    years = sorted(allr[10].yr.unique()); oos = []
    for y in years:
        past = [y - 3, y - 2, y - 1]
        if y - 3 < years[0]:
            continue
        best = max(lbs, key=lambda n: allr[n][allr[n].yr.isin(past)].R.mean())
        oos.append(allr[best][allr[best].yr == y].assign(pick=best))
    W = pd.concat(oos)
    print(f"   walk-forward lookback choice, next-year results: {RL.summ(W.R)} | picks by year: " +
          ", ".join(f"{y}:{g.pick.iat[0]}" for y, g in W.groupby('yr')))
    base = allr[10].copy()
    bh4, bd = TE.bars("BTC"); e = bd.c.ewm(span=50, adjust=False).mean()
    trend = pd.Series(np.where((bd.c > e) & (e > e.shift(5)), "up", np.where((bd.c < e) & (e < e.shift(5)), "down", "flat")), index=bd.index + pd.Timedelta(days=1))
    base["mkt"] = [trend.asof(t) for t in base.t]
    print("   by BTC market trend: " + " | ".join(f"{k}: {RL.summ(g.R)}" for k, g in base.groupby("mkt")))
    stress = MC.allrun(entry_fee=2 * TE.MAKER)  # entry fee doubled
    TE_TAKER, TE_SLIP = TE.TAKER, TE.SLIP
    MC.TE.TAKER, MC.TE.SLIP = 2 * TE_TAKER, 2 * TE_SLIP
    stress2 = MC.allrun(entry_fee=2 * TE.MAKER)
    MC.TE.TAKER, MC.TE.SLIP = TE_TAKER, TE_SLIP
    d, ho, fr = MC.sets(stress2)
    print(f"   double costs: design {RL.summ(d.R)} | held-out {RL.summ(ho.R)} | fresh {RL.summ(fr.R)}")
    print("\nB1: T01 by year (daily portfolio Sharpe, core coins, 0.5% rail)")
    pn = {}
    for s in TE.CORE:
        h4, dd = TE.bars(s); _, p = TE.t1(s, h4, dd, "INTRADAY"); pn[s] = p
    P = pd.DataFrame(pn).fillna(0.0).sum(axis=1); P = P[P.index >= P.ne(0).idxmax()]
    print("   " + " | ".join(f"{y}: {g.mean() / g.std() * np.sqrt(365):.2f}" for y, g in P.groupby(P.index.year) if g.std() > 0))
    P2 = P.copy(); P2.index = P2.index + pd.Timedelta(days=0)
    reg = pd.Series([trend.asof(t) for t in P.index], index=P.index)
    print("   by BTC market trend (daily Sharpe): " + " | ".join(f"{k}: {g.mean() / g.std() * np.sqrt(365):.2f} ({len(g)} days)" for k, g in P.groupby(reg) if g.std() > 0))


if __name__ == "__main__":
    run_w1()
    run_b1()
