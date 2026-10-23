"""Checks run after swing_search.py on 23 Oct 2026, both counted in LEDGER.md (SW1 to SW5 notes, SW6).

1. Diagnostic: the five specs gross of all costs, to separate signal from cost drag. Not a gate decision.
2. SW6 implementation check: MAX-10 on the four tested coins against the four plus the fresh six (LINK, BNB, ADA,
   DOGE, AVAX, DOT), under the sleeve cap of three MAX-10 rows a day (4% each inside the 15% crypto group), core coins
   first. Same trade rule and costs as engine.edges_record.

Usage: python swing_search_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import swing_search as S  # noqa: E402


def gross() -> None:
    saved = (S.C_TRADE, S.F_TRADFI, S.F_CRYPTO)
    S.C_TRADE = S.F_TRADFI = S.F_CRYPTO = 0.0
    P = S.Panel(S.H1_DESIGN, "calendar"); x = S.run(P, S.tsmom(P), "W"); s = S.split(x, S.DESIGN_END)
    print("H1 gross:", S.fmt(s["full"]), "| held-out", S.fmt(s["held"]))
    lo = S.run(P, S.tsmom(P, 1.0), "W"); print("H1 long-only gross:", S.fmt(S.stats(lo)))
    P = S.Panel(S.COMMOD, "business"); sc = S.score_on(P, lambda s, c: c.shift(21) / c.shift(252) - 1)
    x = S.run(P, S.ranked(P, sc, lambda n: 2, 6, "M"), "M"); s = S.split(x, S.DESIGN_END); print("H2 gross:", S.fmt(s["full"]), "| held-out", S.fmt(s["held"]))
    P = S.Panel(S.CORE, "calendar"); sc = S.score_on(P, lambda s, c: (c / c.shift(28) - 1).where(c.expanding().count() >= 120))
    x = S.run(P, S.ranked(P, sc, lambda n: max(2, n // 3), 6, "W"), "W"); s = S.split(x, S.CRYPTO_DESIGN_END); print("H3 gross:", S.fmt(s["full"]), "| held-out", S.fmt(s["held"]))
    P = S.Panel(S.H4_U, "business"); sc = S.score_on(P, lambda s, c: c / c.shift(126) - 1)
    x = S.run(P, S.ranked(P, sc, lambda n: min(3, n), 1, "M", long_only=True), "M"); s = S.split(x, S.DESIGN_END); print("H4 gross:", S.fmt(s["full"]), "| held-out", S.fmt(s["held"]))
    S.C_TRADE, S.F_TRADFI, S.F_CRYPTO = saved


def breadth() -> None:
    CORE4, FRESH6 = ("BTC", "ETH", "SOL", "XRP"), ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
    rows = []
    for s in CORE4 + FRESH6:
        d = E.completed(E.static_klines(s, "1d", "2016-01-01"), "1d"); c, lo = d.c, d.l; a = E.atr(d)
        sig = c >= c.rolling(10).max().shift(1)
        for k in range(20, len(d) - 1):
            if sig.iat[k]:
                e, st = c.iat[k], c.iat[k] - a.iat[k]; ex = st if lo.iat[k + 1] <= st else c.iat[k + 1]
                rows.append((d.index[k + 1].normalize(), s, s in CORE4, (ex - e) / a.iat[k] - 0.0008 * e / a.iat[k]))
    T = pd.DataFrame(rows, columns=["t", "sym", "core", "R"]); T = T[T.t >= "2020-10-01"]  # all ten coins live
    def book(df, cap=3):
        df = df.sort_values(["t", "core"], ascending=[True, False])
        return df.groupby("t").head(cap)
    for name, df in (("four tested coins", T[T.core]), ("four plus fresh six", T)):
        b = book(df); daily = b.groupby("t").R.sum().reindex(pd.date_range("2020-10-01", "2026-10-22"), fill_value=0.0)
        yrs = len(daily) / 365.25; sr = daily.mean() / daily.std() * np.sqrt(365)
        eq = (1 + 0.04 * daily).cumprod(); dd = float((eq / eq.cummax() - 1).min())
        print(f"{name}: {len(b)} trades, {len(b) / yrs:.0f} a year, mean {b.R.mean():+.3f}R, {b.R.sum() / yrs:+.1f}R a year, "
              f"daily SR {sr:.2f}, t {daily.mean() / daily.std() * np.sqrt(len(daily)):+.2f}, worst drawdown at 4% a row {dd * 100:.0f}%")
        for y, g in b.groupby(b.t.dt.year): print(f"   {y}: n {len(g)}, {g.R.sum():+.1f}R")


if __name__ == "__main__":
    gross()
    breadth()
