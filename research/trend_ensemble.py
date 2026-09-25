"""Hypotheses T1 and T3, pre-registered 25 Sep 2026 before running, from the literature review (lit_review.md).

T1, multi-horizon daily trend, long only (Zarattini, Pagani and Barbon, 2025; Han, Kang and Ryu, 2023):
  components N in {5, 10, 20, 30, 60, 90, 150, 250, 360} days per coin. A flat component goes long when the daily
  close (00:00 UTC) is at or above the highest of the prior N closes. Entry: taker at the open of the 08:00 UTC 4h
  bar next day, plus 0.02% slippage. Initial stop: midpoint of the highest and lowest of the last N closes; after each
  daily close it trails up to max(previous stop, new midpoint), active from 08:00 UTC next day, never lowered.
  Exit variants, both counted: INTRADAY = a 4h low at or through the stop exits at the stop (or the open if it gapped
  below), taker plus slippage; CLOSE = a daily close below the stop exits at the next 08:00 UTC open, taker plus slippage.
  Each component risks 1R: R = P&L / (entry - initial stop), net of fees and of funding at 0.03% of notional a day.
T3, MAX-10 next-day continuation, long only (Padysak and Vojtko, 2022): when the daily close is at or above the highest
  of the prior 10 closes, maker entry at the next 08:00 UTC open, stop 1.0 x daily ATR(14), exit at the following
  08:00 UTC open unless the signal repeats (then hold and reset the stop). MIN-10 (buy a 10-day closing low) counted.
Data: BTC from Bitstamp (daily from 2013, 4h execution); ETH, SOL, XRP from Binance 4h (2017 onward).
Sets: design = BTC, ETH, SOL, XRP to 2023-12-31, checked in halves; time held-out = the same four from 2024-01-01,
  run once; fresh held-out = LINK, BNB, ADA, DOGE, AVAX, DOT, all dates, never used in any trend test.
Gate: design t >= 3 on the daily portfolio P&L in R and mean R per trade above zero, the same sign in both halves,
  time held-out and fresh held-out above zero.

Usage: SLEEVE_DATA=/path/to/data python trend_ensemble.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("SLEEVE_DATA", Path(__file__).resolve().parent.parent / "data"))
MAKER, TAKER, SLIP, FUND = 0.0002, 0.0006, 0.0002, 0.0003
MIN_STOP = float(os.environ.get("MIN_STOP", "0"))  # sleeve rail: stops under 0.5% do not print (set 0.005)
LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
CORE = ("BTC", "ETHUSDT", "SOLUSDT", "XRPUSDT")
FRESH = ("LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")
HELD_FROM = pd.Timestamp("2024-01-01")


def bars(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4h execution bars and daily bars (00:00 UTC)."""
    if sym == "BTC":
        m = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
        h4 = m.resample("4h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        old = pd.read_parquet(DATA / "BTC_bitstamp_4h_2012.parquet") if (DATA / "BTC_bitstamp_4h_2012.parquet").exists() else None
        if old is not None:
            h4 = pd.concat([old[old.index < h4.index[0]], h4])
    else:
        h4 = pd.read_csv(DATA / f"{sym}_4h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)
    d = h4.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return h4, d


def t1(sym: str, h4: pd.DataFrame, d: pd.DataFrame, exit_mode: str) -> tuple[list[dict], pd.Series]:
    O, H, L, C = (h4[k].values for k in ("o", "h", "l", "c")); t4 = h4.index
    dc = d.c.values; days = d.index; trades: list[dict] = []
    pnl = pd.Series(0.0, index=days)  # daily mark-to-market P&L in R, summed over components
    at8 = t4.searchsorted(days + pd.Timedelta(hours=8))  # first 4h bar at or after 08:00 UTC each day
    for N in LOOKBACKS:
        hi = pd.Series(dc).rolling(N).max().shift(1).values
        mx = pd.Series(dc).rolling(N).max().values; mn = pd.Series(dc).rolling(N).min().values
        mid = (mx + mn) / 2; i = N + 1
        while i < len(days) - 2:
            if not dc[i] >= hi[i]:
                i += 1; continue
            k = at8[i + 1]
            if k >= len(O) - 1:
                break
            entry = O[k] * (1 + SLIP); stop = mid[i]
            if entry <= stop or (entry - stop) / entry < MIN_STOP:
                i += 1; continue
            risk = entry - stop; stop_now = stop; j = i + 1; exit_px = None; kk = k
            while exit_px is None:
                k_next = at8[j + 1] if j + 1 < len(days) else len(O)
                if exit_mode == "INTRADAY":
                    for kk in range(max(k, at8[j]) if j > i + 1 else k, min(k_next, len(O))):
                        if L[kk] <= stop_now:
                            exit_px = min(stop_now, O[kk]) * (1 - SLIP); break
                if exit_px is None:
                    if j >= len(days) - 2:
                        exit_px = dc[j] * (1 - SLIP); kk = min(k_next, len(O)) - 1; break
                    if exit_mode == "CLOSE" and dc[j] < stop_now:
                        kk = min(k_next, len(O) - 1); exit_px = O[kk] * (1 - SLIP); break
                    stop_now = max(stop_now, mid[j]); j += 1
            held = max((t4[kk] - t4[k]).total_seconds() / 86400, 0.0)
            R = (exit_px - entry) / risk - (2 * TAKER) * entry / risk - FUND * held * entry / risk
            trades.append(dict(sym=sym, N=N, t=t4[k], exit=t4[kk], R=R, days=held, stop_pct=risk / entry * 100))
            # daily mark-to-market in R for the portfolio series
            span = (days > t4[k].normalize()) & (days <= t4[kk].normalize())
            marks = pd.Series(dc, index=days)[span]
            if len(marks):
                path = np.append(marks.values[:-1], exit_px)
                prev = np.append(entry, path[:-1])
                pnl[marks.index] += (path - prev) / risk
            pnl[t4[k].normalize()] -= TAKER * entry / risk
            i = max(int(days.searchsorted(t4[kk].normalize())), i + 1)
    return trades, pnl


def t3(sym: str, h4: pd.DataFrame, d: pd.DataFrame, kind: str) -> list[dict]:
    O, L = h4.o.values, h4.l.values; t4 = h4.index; dc = d.c.values; days = d.index
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values
    ref = pd.Series(dc).rolling(10).max().shift(1).values if kind == "MAX10" else pd.Series(dc).rolling(10).min().shift(1).values
    sig = dc >= ref if kind == "MAX10" else dc <= ref
    at8 = t4.searchsorted(days + pd.Timedelta(hours=8)); out = []; i = 15
    while i < len(days) - 3:
        if not sig[i]:
            i += 1; continue
        k = at8[i + 1]
        if k >= len(O) - 7:
            break
        entry = O[k]; first = True
        while True:  # one day segment per loop, rolled while the signal repeats
            stop = entry - atr[i]; k_end = min(at8[i + 2], len(O) - 1); ex = None
            for q in range(k, k_end):
                if L[q] <= stop:
                    ex = min(stop, O[q]) * (1 - SLIP); break
            roll = ex is None and sig[i + 1] and i + 1 < len(days) - 3
            if ex is None:
                ex = O[k_end] if roll else O[k_end] * (1 - SLIP)
            cost = (MAKER if first else 0.0) + (0.0 if roll else TAKER)
            out.append(dict(sym=sym, kind=kind, t=t4[k], R=(ex - entry) / atr[i] - cost * entry / atr[i] - FUND * entry / atr[i]))
            i += 1
            if not roll:
                break
            entry, k, first = ex, k_end, False
        i += 1
    return out


def summ(x: pd.Series) -> str:
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


def daily_t(p: pd.Series) -> str:
    p = p[p.index >= p.index[0]]
    sh = p.mean() / p.std() * np.sqrt(365) if p.std() > 0 else np.nan
    return f"days={len(p)} Sharpe={sh:.2f} t={p.mean() / p.std() * np.sqrt(len(p)):+.2f}"


def main() -> None:
    res: dict[str, list] = {"INTRADAY": [], "CLOSE": []}; pnls: dict[str, dict] = {"INTRADAY": {}, "CLOSE": {}}; r3 = []
    for sym in CORE + FRESH:
        h4, d = bars(sym)
        for mode in ("INTRADAY", "CLOSE"):
            tr, p = t1(sym, h4, d, mode); res[mode] += tr; pnls[mode][sym] = p
        for kind in ("MAX10", "MIN10"):
            r3 += t3(sym, h4, d, kind)
        print(f"{sym}: {d.index[0].date()} to {d.index[-1].date()}", flush=True)
    for mode in ("INTRADAY", "CLOSE"):
        T = pd.DataFrame(res[mode]); T.to_csv(Path(__file__).with_name(f"trend_t1_{mode.lower()}.csv"), index=False)
        core = T[T.sym.isin(CORE)]; des = core[core.t < HELD_FROM]; ho = core[core.t >= HELD_FROM]; fr = T[T.sym.isin(FRESH)]
        mid = des.t.min() + (des.t.max() - des.t.min()) / 2
        P = pd.DataFrame(pnls[mode]).fillna(0.0)
        pc = P[list(CORE)].sum(axis=1); pf = P[list(FRESH)].sum(axis=1)
        pd_ = pc[pc.index < HELD_FROM]; pd_ = pd_[pd_.index >= pd_.ne(0).idxmax()]
        pmid = pd_.index[0] + (pd_.index[-1] - pd_.index[0]) / 2
        print(f"\nT1 {mode}: trades design {summ(des.R)} | halves {summ(des[des.t < mid].R)} / {summ(des[des.t >= mid].R)} | "
              f"time held-out {summ(ho.R)} | fresh {summ(fr.R)}")
        print(f"   daily portfolio, design {daily_t(pd_)} | halves {daily_t(pd_[pd_.index < pmid])} / {daily_t(pd_[pd_.index >= pmid])}")
        print(f"   daily portfolio, time held-out {daily_t(pc[pc.index >= HELD_FROM])} | fresh {daily_t(pf[pf.index >= pf.ne(0).idxmax()])}")
        print("   by lookback (all sets): " + " | ".join(f"{n}: {T[T.N == n].R.mean():+.2f} ({len(T[T.N == n])})" for n in LOOKBACKS))
        print("   by coin: " + " | ".join(f"{s[:4]} {T[T.sym == s].R.mean():+.2f}" for s in CORE + FRESH))
        print(f"   win rate {(T.R > 0).mean() * 100:.1f}% | median hold {T.days.median():.1f} days | median stop {T.stop_pct.median():.1f}%")
    R3 = pd.DataFrame(r3)
    for kind in ("MAX10", "MIN10"):
        g = R3[R3.kind == kind]; core = g[g.sym.isin(CORE)]; des = core[core.t < HELD_FROM]
        mid = des.t.min() + (des.t.max() - des.t.min()) / 2
        print(f"\nT3 {kind}: design {summ(des.R)} | halves {summ(des[des.t < mid].R)} / {summ(des[des.t >= mid].R)} | "
              f"time held-out {summ(core[core.t >= HELD_FROM].R)} | fresh {summ(g[g.sym.isin(FRESH)].R)}")


if __name__ == "__main__":
    main()
