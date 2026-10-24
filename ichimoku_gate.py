"""Research gate for the four Ichimoku variants: realistic entry, both halves, held-out symbols, matched placebo.

Trade scoring and the matched placebo both come from engine.py, so this script and `engine.py ichimoku` cannot drift.
The difference here is the data source: tradfi daily bars come from the research feed via swing.daily rather than the
engine feed, and a result that does not survive that swap is not a result.

Usage: python ichimoku_gate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E
import swing as S

FIT = tuple(E.SCAN_CRYPTO) + tuple(E.SCAN_STOCKS)  # the set the rules were written against
FRESH = tuple(s for s in S.STOCK_POOL if s not in FIT) + tuple(s for s in S.ETFS if s not in FIT)


def rows(syms: tuple[str, ...], days: int = 540, delay: int = 1, seed: int = 20261024) -> pd.DataFrame:
    """engine.ichimoku_record on a given symbol set, but reading bars through swing.daily."""
    out = []
    orig = E.daily_done
    try:
        E.daily_done = lambda s: S.daily(s)  # same scoring code, research feed for tradfi
        out = E.ichimoku_record(days=days, delay=delay, syms=syms, seed=seed)
    finally:
        E.daily_done = orig
    return out


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"{'variant':14s} {'n':>5s} {'signal':>8s} {'placebo':>8s} {'delta':>7s} {'t':>6s} {'h1':>6s} {'h2':>6s} {'win%':>5s}")
    for v in E.ICHI_VARIANTS:
        g = df[df.edge == v]
        a, b = g[g.arm == "signal"].R, g[g.arm == "placebo"].R
        if not len(a):
            print(f"{v:14s} {0:5d}  no signals")
            continue
        hd = []
        for hh in (0, 1):
            ah, bh = g[(g.arm == "signal") & (g.half == hh)].R, g[(g.arm == "placebo") & (g.half == hh)].R
            hd.append(ah.mean() - bh.mean() if len(ah) and len(bh) else float("nan"))
        print(f"{v:14s} {len(a):5d} {a.mean():8.3f} {b.mean():8.3f} {a.mean() - b.mean():7.3f} "
              f"{E._welch_t(a, b):6.2f} {hd[0]:6.2f} {hd[1]:6.2f} {np.mean(a > 0) * 100:5.1f}")


if __name__ == "__main__":
    report(rows(FIT, delay=1), f"FIT SET, entry one bar after the signal, {len(FIT)} symbols")
    report(rows(FIT, delay=0), "FIT SET, entry at the signal close (optimistic, for comparison only)")
    report(rows(FRESH, delay=1), f"HELD OUT, fresh symbols never used to write the rules, {len(FRESH)} symbols")
    print("\nLive status needs t of 3 or more, positive in both halves, and holding on the held-out set.")
