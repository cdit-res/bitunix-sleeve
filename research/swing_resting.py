"""Resting orders for SW10, pre-registered 23 Oct 2026 before running (hourly data, 2022 on, ten coins).

Baseline: SW10 as the 08:30 run trades it: signal at the 00:00 UTC daily close, fill at the 08:00 UTC open, exit at
the 08:00 UTC open after a daily close below the prior three closes or after ten closes, stop 1 daily ATR below the
signal close on hourly lows (swing_b3_checks.exec_0830).
H1a Resting buy-stop: each morning at 08:00 UTC, with no position and the last close below the prior ten-close high
    H, rest a buy-stop at H for 24 hours; fill at max(H, hourly open) plus 0.02% slippage, taker fees. Stop 1 daily
    ATR below H. Exits as the baseline. (If the close is already at or above H, the baseline entry applies.)
H1b Resting sell-stop exit: entries as the baseline; while in a position, a sell-stop rests at the lowest of the
    three latest closes (refreshed each morning), filled on hourly lows; the 1 ATR stop and ten-close limit remain.
Each judged against the baseline: mean R per trade (risk = daily ATR), and R a year in a book of three rows at most,
core first. Placebo for H1a: a buy-stop at a random level between the last close and H (same exits).
Costs: 0.08% a side, longs pay 0.03% a day.

Usage: python swing_resting.py [hourly_cache_dir]
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
import swing_exec_compare as X  # noqa: E402

COINS = ("BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
CORE = {"BTC", "ETH", "SOL", "XRP"}
rng = np.random.default_rng(20261023)


def sim(sym: str, mode: str) -> pd.DataFrame:
    """mode: base, h1a, h1a_placebo, h1b. Returns trades with entry time, exit time and R."""
    d = B.crypto(sym); h = X.hourly(sym); a = E.atr(d).values; c = d.c.values; dix = d.index
    hix, ho, hh, hl = h.index, h.o.values, h.h.values, h.l.values
    hi10 = d.c.rolling(10).max().shift(1).values  # prior ten closes, known at each daily close
    rows = []; pos = None; k0 = int(np.searchsorted(dix, hix[0])) + 20
    for k in range(k0, len(d) - 2):
        t8 = dix[k] + pd.Timedelta(days=1, hours=8); p = hix.searchsorted(t8); p2 = hix.searchsorted(t8 + pd.Timedelta(days=1))
        if p2 >= len(hix):
            break
        if pos is not None:  # manage an open position through the next 24 hours
            e, st, a0, kin, t_in = pos
            exit_rule = c[k] < c[k - 3:k].min() or k >= kin + 10
            trail = c[k - 2:k + 1].min() if mode == "h1b" else -np.inf  # three latest closes, refreshed at the run
            if exit_rule:
                hit = next((x for x in range(q_last, p) if hl[x] <= max(st, trail)), None)
                ex = min(max(st, trail), ho[hit]) if hit is not None else ho[p]
                rows.append(dict(sym=sym, t=t_in, R=(ex - e) / a0 - (2 * B.SIDE_CR + B.F_CR * (k - kin + 1)) * e / a0)); pos = None
            else:
                lvl = max(st, trail); hit = next((x for x in range(q_last, p2) if hl[x] <= lvl), None)
                if hit is not None:
                    ex = min(lvl, ho[hit]); rows.append(dict(sym=sym, t=t_in, R=(ex - e) / a0 - (2 * B.SIDE_CR + B.F_CR * (k - kin + 1)) * e / a0)); pos = None
                q_last = p2
            if pos is not None:
                continue
        if not np.isfinite(a[k]) or not np.isfinite(hi10[k]):
            continue
        H = hi10[k]
        if c[k] >= H:  # the daily signal fired: baseline entry at 08:00
            e, st = ho[p], c[k] - a[k]
            if e > st:
                pos = (e, st, a[k], k, dix[k]); q_last = p
            continue
        if mode in ("h1a", "h1a_placebo"):
            lvl = H if mode == "h1a" else c[k] + rng.random() * (H - c[k])
            fill = next((x for x in range(p, p2) if hh[x] >= lvl), None)
            if fill is not None:
                e = max(lvl, ho[fill]) * 1.0002; st = lvl - a[k]
                hit = next((x for x in range(fill, p2) if hl[x] <= st), None)
                if hit is not None:
                    rows.append(dict(sym=sym, t=dix[k], R=(min(st, ho[hit]) - e) / a[k] - 2 * B.SIDE_CR * e / a[k]))
                else:
                    pos = (e, st, a[k], k, dix[k]); q_last = p2
    return pd.DataFrame(rows)


def book(T: pd.DataFrame, slots: int = 3) -> float:
    """R a year from trades booked on entry day, three at most at once, core first (holds approximated as 5 days)."""
    T = T.assign(core=T.sym.isin(CORE)).sort_values(["t", "core"], ascending=[True, False]); taken, ends = [], []
    for r in T.itertuples():
        ends = [u for u in ends if u > r.t]
        if len(ends) < slots:
            taken.append(r.R); ends.append(r.t + pd.Timedelta(days=5))
    yrs = (T.t.max() - T.t.min()).days / 365.25
    return sum(taken) / yrs


def main() -> None:
    res = {}
    for mode in ("base", "h1a", "h1a_placebo", "h1b"):
        T = pd.concat([sim(s, mode) for s in COINS]); res[mode] = T
        core = T[T.sym.isin(CORE)]
        print(f"{mode}: all {B.fmtR(T.R)}; core {B.fmtR(core.R)}; capped book about {book(T):+.1f}R a year")
        for y, g in T.groupby(T.t.dt.year):
            print(f"   {y}: {B.fmtR(g.R)}")
    b, a1, a1p, b1 = (res[m].R for m in ("base", "h1a", "h1a_placebo", "h1b"))
    print(f"\nH1a minus baseline: {a1.mean() - b.mean():+.3f}R a trade; H1a minus its placebo {a1.mean() - a1p.mean():+.3f}R")
    print(f"H1b minus baseline: {b1.mean() - b.mean():+.3f}R a trade")


if __name__ == "__main__":
    main()
