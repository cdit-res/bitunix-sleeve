"""Swing batch 5, pre-registered 23 Oct 2026 before running. The owner asked for longer moves at 3x to 5x as well.

SW10's rule on weekly bars: long at a weekly close at or above the prior ten weekly closes; stop 1 weekly ATR(14)
below that close; exit at a weekly close below the lowest of the prior three weekly closes, or after ten weekly
closes; stops on weekly lows (fill at the stop, or the weekly open if it gaps through). One position per market.
Placebo: a long every week with the same stop and exits. Counterparty: slow-moving capital that keeps buying a
trend for weeks (time-series momentum at one to four weeks; Liu and Tsyvinski, 2021).

E1 Crypto, weeks ending Sunday 24:00 UTC. Core BTC, ETH, SOL, XRP (design to 2023, held out 2024 on); fresh LINK,
   BNB, ADA, DOGE, AVAX, DOT. Costs 0.08% a side, longs pay 0.03% a calendar day.
E2 Index ETFs, weeks ending Friday. Design SPY and QQQ to 2016, held out 2017 on; fresh IWM, DIA, XLK, XLF, XLE,
   SMH, EWJ, EWZ, EWY, EWT, GLD, TLT. Costs 0.09% a side, funding 0.0105% a calendar day.
E3 The eight commodity futures, weeks ending Friday, design to 2016, held out 2017 on; no fresh set. Tradfi costs.
Gate as before; also printed: the median stop distance and the leverage it implies under the sleeve's rule
(55% of margin at the stop).

Usage: python swing_batch5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import swing_batch2 as B  # noqa: E402

CORE, FRESH = ("BTC", "ETH", "SOL", "XRP"), ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
COMMOD = ("GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "NG=F")
ETF_D, ETF_F = ("SPY", "QQQ"), ("IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "EWJ", "EWZ", "EWY", "EWT", "GLD", "TLT")


def weekly(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    w = d.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return w[w.index <= pd.Timestamp(B.END)]  # completed weeks only


def run(w: pd.DataFrame, side_cost: float, fund: float, placebo: bool = False, mult: float = 1.0) -> pd.DataFrame:
    c, lo, o = w.c.values, w.l.values, w.o.values; a = E.atr(w).values; n = len(w)
    sig = (w.c >= w.c.rolling(10).max().shift(1)).values; rows = []; k = 20
    while k < n - 1:
        if not (placebo or sig[k]) or not np.isfinite(a[k]) or a[k] <= 0:
            k += 1; continue
        e, st = c[k], c[k] - a[k]; ex = None; j = k + 1
        while j < n:
            if o[j] <= st or lo[j] <= st:
                ex = min(st, o[j]); break
            if c[j] < c[j - 3:j].min() or j >= k + 10:
                ex = c[j]; break
            j += 1
        if ex is None:
            break
        days = (w.index[j] - w.index[k]).days
        rows.append(dict(t=w.index[k], R=(ex - e) / a[k] - (2 * side_cost + fund * days) * e / a[k] * mult,
                         stop_pct=a[k] / e * 100, weeks=j - k))
        k = k + 1 if placebo else j + 1
    return pd.DataFrame(rows)


def series(T: pd.DataFrame) -> pd.Series:
    return pd.Series(T.R.values, index=pd.DatetimeIndex(T.t)) if len(T) else pd.Series(dtype=float)


def judge(name: str, data: dict[str, pd.DataFrame], design: tuple, fresh: tuple, end: str, side_cost: float, fund: float) -> bool:
    sig = {s: run(w, side_cost, fund) for s, w in data.items()}; plc = {s: run(w, side_cost, fund, True) for s, w in data.items()}
    dbl = {s: run(w, side_cost, fund, mult=2.0) for s, w in data.items()}
    x = pd.concat([series(sig[s]) for s in design]).sort_index(); r = B.report(name, x, end, B.fmtR)
    p = pd.concat([series(plc[s]) for s in design]).sort_index()
    des, _, _, held = B.halves(x, end); pdz, _, _, pheld = B.halves(p, end)
    print(f"    placebo: design {B.fmtR(pdz)}; held-out {B.fmtR(pheld)}")
    print(f"    signal minus placebo: design {des.mean() - pdz.mean():+.3f}R; held-out {held.mean() - pheld.mean():+.3f}R")
    fr_mean, ok_f = 1.0, True
    if fresh:
        fr = pd.concat([series(sig[s]) for s in fresh]); frp = pd.concat([series(plc[s]) for s in fresh])
        print(f"    fresh: {B.fmtR(fr)}; placebo {B.fmtR(frp)}; difference {fr.mean() - frp.mean():+.3f}R")
        fr_mean, ok_f = fr.mean(), fr.mean() > frp.mean()
    allT = pd.concat([sig[s] for s in data]); med = allT.stop_pct.median()
    lev = int(min(80, max(1, np.floor(55 / (med + 0.08)))))
    print(f"    median stop {med:.1f}% below entry, implying about {lev}x; median hold {allT.weeks.median():.0f} weeks; "
          f"{len(allT) / max(1, sum((w.index[-1] - w.index[20]).days for w in data.values()) / 365.25):.1f} trades a market-year")
    d2 = pd.concat([series(dbl[s]) for s in design]); d2 = d2[d2.index <= end]; print(f"    double costs (design): {B.fmtR(d2)}")
    ok_p = des.mean() > pdz.mean() and held.mean() > pheld.mean() and ok_f
    return B.verdict(name, r, fr_mean, ok_p, B.tstat(d2))


def main() -> None:
    res = {}
    print("E1 weekly MAX-10 swing, crypto")
    data = {s: weekly(B.crypto(s), "W-SUN") for s in CORE + FRESH}
    res["E1"] = judge("E1 crypto", data, CORE, FRESH, B.CRYPTO_DESIGN_END, B.SIDE_CR, B.F_CR)
    print("\nE2 weekly MAX-10 swing, index ETFs")
    data = {s: weekly(B.tradfi(s), "W-FRI") for s in ETF_D + ETF_F}
    res["E2"] = judge("E2 ETFs", data, ETF_D, ETF_F, B.DESIGN_END, B.SIDE_TF, B.F_TF)
    print("\nE3 weekly MAX-10 swing, commodity futures")
    data = {s: weekly(B.tradfi(s), "W-FRI") for s in COMMOD}
    res["E3"] = judge("E3 commodities", data, COMMOD, (), B.DESIGN_END, B.SIDE_TF, B.F_TF)
    print("\nPASSED: " + (", ".join(k for k, v in res.items() if v) or "none"))


if __name__ == "__main__":
    main()
