"""Book-level check of D3 (volume filter) and D5 (pyramid add-ons), run 23 Oct 2026 after both passed batch 4.

A refinement is adopted only if it improves the capped book, not just the mean trade: at most three rows at once,
4% risk each, core coins first, marked to market daily, October 2020 on. Variants:
  SW10 as live; D3 as a hard filter; D3 as a priority (high-volume signals take free slots first, no filter);
  D5 (add-ons allowed, each taking a slot); D5 with D3 priority.
Usage: python swing_batch4_checks.py
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
import swing_batch4 as D  # noqa: E402

START, RISK = "2020-10-01", 0.04


def book(cands: list[tuple], data: dict[str, pd.DataFrame], slots: int = 3) -> pd.Series:
    """cands: (t, priority, sym, k, j). Marked to market; stop, exit-close and gap handling as SW10."""
    cands = sorted(cands, key=lambda x: (x[0], x[1])); days = pd.date_range(START, B.END)
    pnl = pd.Series(0.0, index=days); open_until: list[pd.Timestamp] = []; atr = {s: E.atr(d).values for s, d in data.items()}
    for t, _, s, k, j in cands:
        if t < pd.Timestamp(START):
            continue
        open_until = [u for u in open_until if u > t]
        if len(open_until) >= slots:
            continue
        d = data[s]; c, lo, o = d.c.values, d.l.values, d.o.values; a = atr[s][k]; e, st = c[k], c[k] - a; prev = e
        for x in range(k + 1, j + 1):
            px = o[x] if o[x] <= st else (st if lo[x] <= st else c[x])
            if d.index[x] in pnl.index:
                pnl[d.index[x]] += (px - prev) / a - B.F_CR * e / a
            prev = px
            if px != c[x]:
                break
        for x in (k, j):
            if d.index[x] in pnl.index:
                pnl[d.index[x]] -= B.SIDE_CR * e / a
        open_until.append(d.index[j])
    return pnl


def stats(name: str, pnl: pd.Series) -> None:
    yrs = len(pnl) / 365.25; eq = (1 + RISK * pnl).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    by = pnl.groupby(pnl.index.year).sum()
    print(f"  {name}: {pnl.sum() / yrs:+.1f}R a year, daily SR {pnl.mean() / pnl.std() * np.sqrt(365):.2f}, worst drawdown {dd * 100:.0f}%, "
          f"x{eq.iat[-1]:.1f}; by year " + ", ".join(f"{y} {v:+.1f}R" for y, v in by.items()))


def main() -> None:
    for label, coins in (("core four", D.CORE), ("core four plus fresh six", D.CORE + D.FRESH)):
        data = {s: D.crypto_v(s) for s in coins}
        hv = {s: (data[s].v > data[s].v.rolling(20).mean().shift(1)).values for s in coins}
        base, filt, prio, pyr, pyr_prio = [], [], [], [], []
        for s in coins:
            core = s in D.CORE
            T = D.sw(data[s]); spans = [(int(r.k), int(r.j)) for r in T.itertuples()]
            for r in T.itertuples():
                t = data[s].index[r.k]; h = bool(hv[s][r.k])
                base.append((t, (not core,), s, r.k, r.j)); prio.append((t, (not h, not core), s, r.k, r.j))
                pyr.append((t, (not core,), s, r.k, r.j)); pyr_prio.append((t, (not h, not core), s, r.k, r.j))
            for r in D.sw(data[s], mask=hv[s]).itertuples():
                filt.append((data[s].index[r.k], (not core,), s, r.k, r.j))
            for r in D.sw(data[s], overlap=True, only_inside=spans).itertuples():
                t = data[s].index[r.k]; h = bool(hv[s][r.k])
                pyr.append((t, (not core, 1), s, r.k, r.j)); pyr_prio.append((t, (not h, not core, 1), s, r.k, r.j))
        print(f"Book, three rows at most, 4% each, {label}, Oct 2020 on")
        for name, cands in (("SW10 as live", base), ("D3 as a filter", filt), ("D3 as a priority", prio),
                            ("D5 add-ons", pyr), ("D5 add-ons with D3 priority", pyr_prio)):
            stats(name, book(cands, data))


if __name__ == "__main__":
    main()
