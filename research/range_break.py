"""Hypothesis RB: break of an established range, pre-registered 25 Sep 2026 after range_grid.py failed.

Why: range_grid.py found that fading the edges of a tested 1h range loses 0.4R to 0.5R per trade, worse than a
random walk implies (MID wins 21.7% against about 28% expected), in design and held-out alike. Tested edges tend to
break. This checks whether trading the break pays after costs. The time held-out for BTC, ETH, SOL and XRP has
now been seen through the fade, so the clean test is six symbols never used here: LINK, BNB, ADA, DOGE, AVAX, DOT.

Range: exactly as range_grid.py (N = 36 1h bars, 3 to 8 ATR wide, at least 1% of price, 3 or more alternations).
Entry: while the range is valid, a buy stop at H + 0.25 ATR and a sell stop at L - 0.25 ATR, re-set each hour.
  Taker fill at the level (or the bar open if beyond it) plus 0.02% slippage.
Stop, two variants, both counted: IN = back inside the range at H - 0.25 ATR (short: L + 0.25 ATR);
  MID = the range midpoint.
Target 3R, maker. Exit at market after 48h. One position per side at a time; the side re-arms after the exit.
Execution, stop-first and split as range_grid.py. Fresh symbols run on 1h bars, 2022-01 to 2026-09, all out of sample.
Gate: design t >= 3, the same sign in both design halves, time held-out mean above zero, and fresh symbols above zero.

Usage: SLEEVE_DATA=/path/to/data python range_break.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import range_grid as G

FRESH = ("LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")
TARGET_R, BEYOND = 3.0, 0.25


def run_side(sym: str, sig: pd.DataFrame, ex: pd.DataFrame, rg: pd.DataFrame, side: int, variant: str) -> list[dict]:
    T = sig.index; eo, eh, el, ec = (ex[k].values for k in ("o", "h", "l", "c"))
    starts = ex.index.searchsorted(T); ends = ex.index.searchsorted(T + pd.Timedelta(hours=1))
    step = pd.Timedelta(ex.index[1] - ex.index[0]); hold = int(pd.Timedelta(hours=G.HOLD_H) / step)
    V, H, L, W, A, C = (rg[k].values for k in ("valid", "H", "L", "W", "atr", "c"))
    out: list[dict] = []; t = G.N + 14
    while t < len(T) - 2:
        if not V[t]:
            t += 1; continue
        if side == 1:
            lvl = H[t] + BEYOND * A[t]; stop = H[t] - BEYOND * A[t] if variant == "IN" else L[t] + 0.5 * W[t]
            if C[t] >= lvl:
                t += 1; continue
        else:
            lvl = L[t] - BEYOND * A[t]; stop = L[t] + BEYOND * A[t] if variant == "IN" else H[t] - 0.5 * W[t]
            if C[t] <= lvl:
                t += 1; continue
        k0, k1 = starts[t + 1], ends[t + 1]
        hit = np.nonzero(eh[k0:k1] >= lvl)[0] if side == 1 else np.nonzero(el[k0:k1] <= lvl)[0]
        if not len(hit):
            t += 1; continue
        k = k0 + int(hit[0])
        entry = (max(lvl, eo[k]) if side == 1 else min(lvl, eo[k])) * (1 + side * G.SLIP)
        risk = side * (entry - stop)
        if risk <= 0:
            t += 1; continue
        tgt = entry + side * TARGET_R * risk
        loss = -1 - (G.TAKER + G.TAKER + G.SLIP) * entry / risk  # entry slippage is in the fill price
        win = TARGET_R - (G.TAKER + G.MAKER) * entry / risk
        if (side == 1 and el[k] <= stop) or (side == -1 and eh[k] >= stop):
            R, q, how = loss, k, "stop"
        else:
            R = None; q = k; end = min(k + hold, len(ec) - 1)
            for q in range(k + 1, end + 1):
                if (side == 1 and el[q] <= stop) or (side == -1 and eh[q] >= stop):
                    R, how = loss, "stop"; break
                if (side == 1 and eh[q] >= tgt) or (side == -1 and el[q] <= tgt):
                    R, how = win, "target"; break
            if R is None:
                R, how = side * (ec[q] - entry) / risk - (G.TAKER + G.TAKER + G.SLIP) * entry / risk, "time"
        out.append(dict(sym=sym, side=side, variant=variant, t=ex.index[k], R=R, how=how, stop_pct=risk / entry * 100))
        t = max(int(T.searchsorted(ex.index[q], side="right")) - 1, t) + 1
    return out


def main() -> None:
    rows, fade = [], []
    for sym in tuple(G.SPLIT) + FRESH:
        sig, ex = G.load(sym); rg = G.ranges(sig)
        for variant in ("IN", "MID"):
            for side in (1, -1):
                rows += run_side(sym, sig, ex, rg, side, variant)
        if sym in FRESH:  # the fade's own out-of-sample check on symbols it never saw
            for side in (1, -1):
                fade += G.run_side(sym, sig, ex, rg, side, "MID")
    T = pd.DataFrame(rows)
    split = {**G.SPLIT, **{s: "2100-01-01" for s in FRESH}}
    T["per"] = ["fresh" if s in FRESH else ("held-out" if t >= pd.Timestamp(split[s]) else "design") for s, t in zip(T.sym, T.t)]
    T.to_csv(Path(__file__).with_name("range_break_trades.csv"), index=False)
    for variant in ("IN", "MID"):
        g = T[T.variant == variant]; d = g[g.per == "design"]; mid = d.t.min() + (d.t.max() - d.t.min()) / 2
        print(f"\n{variant}: design {G.summ(d.R)} | halves {G.summ(d[d.t < mid].R)} / {G.summ(d[d.t >= mid].R)} | "
              f"held-out {G.summ(g[g.per == 'held-out'].R)} | fresh {G.summ(g[g.per == 'fresh'].R)}")
        print(f"   win rate {(g.how == 'target').mean() * 100:.1f}% | median stop {g.stop_pct.median():.2f}%")
        print("   by symbol: " + " | ".join(f"{s[:4]} {g[g.sym == s].R.mean():+.2f}" for s in tuple(G.SPLIT) + FRESH))
        print("   longs " + G.summ(g[g.side == 1].R) + " | shorts " + G.summ(g[g.side == -1].R))
    F = pd.DataFrame(fade)
    print(f"\nFade (range_grid MID) on the six fresh symbols: {G.summ(F.R)} | win rate {(F.how == 'target').mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
