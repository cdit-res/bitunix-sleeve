"""Swing batch 3, pre-registered 23 Oct 2026 before running, after B3 (MAX-10 held as a swing) passed batch 2.

The B3 rule, unchanged: long at the close when it is at or above the prior ten closes; stop 1 daily ATR below that
close; exit at a close below the lowest of the prior three closes, or after ten closes; one position per market.
Placebo: a long on every day with the same stop and exits. Gate as batch 2.

C1 Extra fresh evidence for B3 (no new variant): the feed's other coins, NEAR, INJ, LTC, FET, RUNE, SUI, TAO
   (Binance spot daily from June 2022 or listing). Crypto costs (0.08% a side, longs pay 0.03% a day).
C2 B3 on the eight commodity futures (gold, silver, platinum, palladium, WTI, Brent, copper, natgas; Yahoo
   continuous, rows with a non-positive price dropped). Design to 2016, held out 2017 on; no fresh set. Tradfi costs
   (0.09% a side, funding 0.0105% a calendar day). Counterparty as for crypto MAX-10: late trend followers.
C3 B3 on index ETFs: design SPY and QQQ to 2016, held out 2017 on; fresh IWM, DIA, XLK, XLF, XLE, SMH, EWJ, EWZ,
   EWY, EWT, GLD, TLT. Tradfi costs.

Usage: python swing_batch3.py
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

COMMOD = ("GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "NG=F")
FEED_COINS = ("NEAR", "INJ", "LTC", "FET", "RUNE", "SUI", "TAO")


def swing(d: pd.DataFrame, side_cost: float, fund: float, placebo: bool = False, mult: float = 1.0) -> pd.Series:
    c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; out = {}; k = 20; n = len(d)
    while k < n - 1:
        if not (placebo or sig[k]) or not np.isfinite(a[k]) or a[k] <= 0:
            k += 1; continue
        e, st = c[k], c[k] - a[k]; ex = None; j = k + 1
        while j < n:
            if o[j] <= st:
                ex = o[j]; break
            if lo[j] <= st:
                ex = st; break
            if c[j] < c[j - 3:j].min() or j >= k + 10:
                ex = c[j]; break
            j += 1
        if ex is None:
            break
        days = (d.index[j] - d.index[k]).days
        out[d.index[k]] = (ex - e) / a[k] - (2 * side_cost + fund * days) * e / a[k] * mult
        k = k + 1 if placebo else j + 1
    return pd.Series(out, dtype=float)


def feed_coin(sym: str) -> pd.DataFrame:
    d = E.completed(E.bars(sym, "1d"), "1d")[["o", "h", "l", "c"]].astype(float); d.index = d.index.normalize()
    return d


def judge(name: str, sig: dict[str, pd.Series], plc: dict[str, pd.Series], dbl: dict[str, pd.Series],
          design: tuple[str, ...], fresh: tuple[str, ...], end: str) -> bool:
    x = pd.concat([sig[s] for s in design]).sort_index(); r = B.report(name, x, end, B.fmtR)
    p = pd.concat([plc[s] for s in design]).sort_index()
    des, _, _, held = B.halves(x, end); pdz, _, _, pheld = B.halves(p, end)
    line = f"    placebo: design {B.fmtR(pdz)}; held-out {B.fmtR(pheld)}"
    fr_mean, ok_fresh_p = 1.0, True
    if fresh:
        fr = pd.concat([sig[s] for s in fresh]); frp = pd.concat([plc[s] for s in fresh])
        line += f"; fresh {B.fmtR(frp)}"; print(line); print(f"    fresh: {B.fmtR(fr)}; minus placebo {fr.mean() - frp.mean():+.3f}R")
        fr_mean, ok_fresh_p = fr.mean(), fr.mean() > frp.mean()
    else:
        print(line)
    print(f"    signal minus placebo: design {des.mean() - pdz.mean():+.3f}R; held-out {held.mean() - pheld.mean():+.3f}R")
    d2 = pd.concat([dbl[s] for s in design]); d2 = d2[d2.index <= end]; print(f"    double costs (design): {B.fmtR(d2)}")
    ok_p = des.mean() > pdz.mean() and held.mean() > pheld.mean() and ok_fresh_p
    return B.verdict(name, r, fr_mean, ok_p, B.tstat(d2))


def main() -> None:
    print("C1 B3 on the feed's other coins (fresh evidence, June 2022 on)")
    for s in FEED_COINS:
        d = feed_coin(s); x = swing(d, B.SIDE_CR, B.F_CR); p = swing(d, B.SIDE_CR, B.F_CR, True)
        print(f"  {s}: signal {B.fmtR(x)}; placebo {B.fmtR(p)}; difference {x.mean() - p.mean():+.3f}R")
    X = pd.concat([swing(feed_coin(s), B.SIDE_CR, B.F_CR) for s in FEED_COINS])
    P = pd.concat([swing(feed_coin(s), B.SIDE_CR, B.F_CR, True) for s in FEED_COINS])
    print(f"  pooled: signal {B.fmtR(X)}; placebo {B.fmtR(P)}; difference {X.mean() - P.mean():+.3f}R")

    print("\nC2 B3 on commodity futures")
    data = {s: B.tradfi(s) for s in COMMOD}
    sig = {s: swing(d, B.SIDE_TF, B.F_TF) for s, d in data.items()}; plc = {s: swing(d, B.SIDE_TF, B.F_TF, True) for s, d in data.items()}
    dbl = {s: swing(d, B.SIDE_TF, B.F_TF, mult=2.0) for s, d in data.items()}
    ok2 = judge("C2 commodities", sig, plc, dbl, COMMOD, (), B.DESIGN_END)
    for s in COMMOD:
        print(f"      {s}: {B.fmtR(sig[s])}")

    print("\nC3 B3 on index ETFs")
    ds, fs = ("SPY", "QQQ"), ("IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "EWJ", "EWZ", "EWY", "EWT", "GLD", "TLT")
    data = {s: B.tradfi(s) for s in ds + fs}
    sig = {s: swing(d, B.SIDE_TF, B.F_TF) for s, d in data.items()}; plc = {s: swing(d, B.SIDE_TF, B.F_TF, True) for s, d in data.items()}
    dbl = {s: swing(d, B.SIDE_TF, B.F_TF, mult=2.0) for s, d in data.items()}
    ok3 = judge("C3 index ETFs", sig, plc, dbl, ds, fs, B.DESIGN_END)
    print("\nPASSED: " + (", ".join(k for k, v in (("C2", ok2), ("C3", ok3)) if v) or "none"))


if __name__ == "__main__":
    main()
