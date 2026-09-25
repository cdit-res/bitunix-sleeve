"""Hypothesis RD: range deviation reclaim, and RT: trend-aligned range fade. Pre-registered 25 Sep 2026 before running.

Why: range_grid.py showed resting orders at range edges lose (the edge usually breaks); range_break.py showed breaks
rarely follow through (with the stop just inside, 96% returned into the range). Range traders' actual setup uses both
facts: wait for the sweep beyond the edge, and enter when price closes back inside.
Range: exactly as range_grid.py (N = 36 1h bars, 3 to 8 ATR wide, at least 1% of price, 3 or more alternations).
RD (both sides): while a range is valid, a sweep is any execution bar trading beyond L (or H). If, within 6 hours of
  the first sweep, an execution bar closes back inside the range, enter at the next bar's open (taker plus slippage).
  Stop: 0.1 ATR(1h) beyond the most extreme price since the sweep. Targets, two variants counted: MID (L + 0.5 W) and
  FAR (H - 0.1 W; mirror for shorts). Exit at market after 48h.
RD-T: RD, taking only longs when the coin's 14 and 28-day returns are both positive and only shorts when both negative.
RT: range_grid.py's fade (limit at L + 0.1 W, stop 0.25 ATR beyond, MID target) taking only the side the same
  14/28-day trend allows.
Execution, costs, stop-first rule, splits and gate as range_grid.py and range_break.py; fresh coins LINK, BNB, ADA,
DOGE, AVAX, DOT. Four variants counted: RD-MID, RD-FAR, RD-T (MID), RT.

Usage: SLEEVE_DATA=/path/to/data python range_deviation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import range_grid as G

FRESH = ("LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")
WINDOW_H, BEYOND = 6, 0.1


def trend_sign(sig: pd.DataFrame) -> pd.Series:
    """+1 when the 14 and 28-day returns are both positive, -1 when both negative, else 0; known at each hour."""
    d = sig.c.resample("1D").last().dropna()
    s = (np.sign(d / d.shift(14) - 1) + np.sign(d / d.shift(28) - 1)) / 2
    s = s.where(s.abs() == 1, 0.0); s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(sig.index, method="ffill").fillna(0.0)


def run_rd(sym, sig, ex, rg, side, target, trend=None) -> list[dict]:
    T = sig.index; eo, eh, el, ec = (ex[k].values for k in ("o", "h", "l", "c"))
    starts = ex.index.searchsorted(T); ends = ex.index.searchsorted(T + pd.Timedelta(hours=1))
    step = pd.Timedelta(ex.index[1] - ex.index[0]); hold = int(pd.Timedelta(hours=G.HOLD_H) / step)
    win = int(pd.Timedelta(hours=WINDOW_H) / step)
    V, H, L, W, A = (rg[k].values for k in ("valid", "H", "L", "W", "atr")); out = []; t = G.N + 14
    while t < len(T) - 2:
        if not V[t] or (trend is not None and trend.iat[t] != side):
            t += 1; continue
        edge = L[t] if side == 1 else H[t]
        k0, k1 = starts[t + 1], ends[t + 1]
        sweep = np.nonzero(el[k0:k1] < edge)[0] if side == 1 else np.nonzero(eh[k0:k1] > edge)[0]
        if not len(sweep):
            t += 1; continue
        ks = k0 + int(sweep[0]); ext = el[ks] if side == 1 else eh[ks]; kr = None
        for q in range(ks, min(ks + win, len(ec) - 2)):
            ext = min(ext, el[q]) if side == 1 else max(ext, eh[q])
            if (side == 1 and ec[q] > edge) or (side == -1 and ec[q] < edge):
                kr = q; break
        if kr is None:
            t = int(T.searchsorted(ex.index[min(ks + win, len(ec) - 1)], side="right")); continue
        k = kr + 1; entry = eo[k] * (1 + side * G.SLIP); stop = ext - side * BEYOND * A[t]
        tgt = (L[t] + 0.5 * W[t]) if side == 1 else (H[t] - 0.5 * W[t])
        if target == "FAR":
            tgt = (H[t] - 0.1 * W[t]) if side == 1 else (L[t] + 0.1 * W[t])
        risk = side * (entry - stop)
        if risk <= 0 or side * (tgt - entry) <= 0:
            t = int(T.searchsorted(ex.index[k], side="right")); continue
        loss = -1 - (G.TAKER + G.TAKER + G.SLIP) * entry / risk
        win_r = side * (tgt - entry) / risk - (G.TAKER + G.MAKER) * entry / risk
        R = None; q = k; end = min(k + hold, len(ec) - 1)
        for q in range(k, end + 1):
            if (side == 1 and el[q] <= stop) or (side == -1 and eh[q] >= stop):
                R, how = loss, "stop"; break
            if q > k and ((side == 1 and eh[q] >= tgt) or (side == -1 and el[q] <= tgt)):
                R, how = win_r, "target"; break
        if R is None:
            R, how = side * (ec[q] - entry) / risk - (G.TAKER + G.TAKER + G.SLIP) * entry / risk, "time"
        out.append(dict(sym=sym, side=side, t=ex.index[k], R=R, how=how, stop_pct=risk / entry * 100,
                        gross_rr=side * (tgt - entry) / risk))
        t = max(int(T.searchsorted(ex.index[q], side="right")) - 1, t) + 1
    return out


def main() -> None:
    rows = []
    for sym in tuple(G.SPLIT) + FRESH:
        sig, ex = G.load(sym); rg = G.ranges(sig); tr = trend_sign(sig)
        for side in (1, -1):
            for target in ("MID", "FAR"):
                rows += [dict(r, variant=f"RD-{target}") for r in run_rd(sym, sig, ex, rg, side, target)]
            rows += [dict(r, variant="RD-T") for r in run_rd(sym, sig, ex, rg, side, "MID", tr)]
            rt = G.run_side(sym, sig, ex, rg, side, "MID")
            rows += [dict(r, variant="RT") for r in rt if tr.asof(r["t"]) == side]
        print(sym, flush=True)
    T = pd.DataFrame(rows)
    split = {**G.SPLIT, **{s: "2100-01-01" for s in FRESH}}
    T["per"] = ["fresh" if s in FRESH else ("held-out" if t >= pd.Timestamp(split[s]) else "design") for s, t in zip(T.sym, T.t)]
    T.to_csv(Path(__file__).with_name("range_deviation_trades.csv"), index=False)
    for v in ("RD-MID", "RD-FAR", "RD-T", "RT"):
        g = T[T.variant == v]; d = g[g.per == "design"]; mid = d.t.min() + (d.t.max() - d.t.min()) / 2
        print(f"\n{v}: design {G.summ(d.R)} | halves {G.summ(d[d.t < mid].R)} / {G.summ(d[d.t >= mid].R)} | "
              f"held-out {G.summ(g[g.per == 'held-out'].R)} | fresh {G.summ(g[g.per == 'fresh'].R)}")
        print(f"   win rate {(g.how == 'target').mean() * 100:.1f}% | median stop {g.stop_pct.median():.2f}% | "
              f"median gross R:R {g.gross_rr.median() if 'gross_rr' in g else float('nan'):.2f} | "
              f"longs {G.summ(g[g.side == 1].R)} | shorts {G.summ(g[g.side == -1].R)}")


if __name__ == "__main__":
    main()
