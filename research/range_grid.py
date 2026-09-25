"""Hypothesis R: range fade ('grid'), pre-registered 25 Sep 2026 before running.

Mechanism claimed: inside an established range, resting liquidity and stop runs make price revert from the edges.
Signal, 1h bars, information to the close of bar t:
  window N = 36 bars; H and L are the highest high and lowest low; W = H - L; ATR = ATR(14) on 1h.
  Valid range: 3 ATR <= W <= 8 ATR, W >= 1.0% of price, and at least 3 alternations inside the window
  between the upper zone (high >= H - 0.15 W) and the lower zone (low <= L + 0.15 W).
Orders, re-set each hour while the range stays valid: long limit at L + 0.1 W, short limit at H - 0.1 W,
  only where price is on the far side of the order. Stop 0.25 ATR beyond the range edge.
Targets, two variants, both counted: MID = L + 0.5 W (short: H - 0.5 W); FAR = H - 0.1 W (short: L + 0.1 W).
Exit at market after 48h if neither is hit. One position per side at a time; the side re-arms after the exit.
Costs: maker 0.02% on entry and target; taker 0.06% plus 0.02% slippage on stops; taker 0.06% on time exits.
Execution: BTC on 5m bars (Bitstamp, 2016-06 to 2026-09); ETH, SOL and XRP on 1h bars (Binance, 2022-26).
Stop first when a bar spans stop and target; a fill bar that also reaches the stop is a loss.
Split: BTC design to 2023-12-31, held out from 2024-01-01; alts design to 2024-12-31, held out from 2025-01-01.
Gate: design t >= 3, the same sign in both design halves, held-out mean above zero; held-out run once.

Usage: SLEEVE_DATA=/path/to/data python range_grid.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("SLEEVE_DATA", Path(__file__).resolve().parent.parent / "data"))
MAKER, TAKER, SLIP = 0.0002, 0.0006, 0.0002
N, ZONE, ENTRY_IN, STOP_ATR, HOLD_H = 36, 0.15, 0.10, 0.25, 48
SPLIT = {"BTC": "2024-01-01", "ETHUSDT": "2025-01-01", "SOLUSDT": "2025-01-01", "XRPUSDT": "2025-01-01"}


def load(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal bars (1h) and execution bars (5m for BTC, 1h otherwise)."""
    if sym == "BTC":
        ex = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
        sig = ex.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        return sig, ex
    sig = pd.read_csv(DATA / f"{sym}_1h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)
    return sig, sig


def ranges(sig: pd.DataFrame) -> pd.DataFrame:
    h, l, c = sig.h.values, sig.l.values, sig.c.values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    H = pd.Series(h).rolling(N).max().values; L = pd.Series(l).rolling(N).min().values; W = H - L
    valid = np.zeros(len(sig), bool)
    for t in range(N + 14, len(sig)):
        w = W[t]
        if not (3 * atr[t] <= w <= 8 * atr[t] and w >= 0.01 * c[t]):
            continue
        up = h[t - N + 1:t + 1] >= H[t] - ZONE * w; dn = l[t - N + 1:t + 1] <= L[t] + ZONE * w
        seq = [1 if u else -1 for u, d in zip(up, dn) if u != d]
        alt = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        valid[t] = alt >= 3
    return pd.DataFrame({"H": H, "L": L, "W": W, "atr": atr, "c": c, "valid": valid}, index=sig.index)


def run_side(sym: str, sig: pd.DataFrame, ex: pd.DataFrame, rg: pd.DataFrame, side: int, variant: str) -> list[dict]:
    T = sig.index; eo, eh, el, ec = (ex[k].values for k in ("o", "h", "l", "c"))
    starts = ex.index.searchsorted(T); ends = ex.index.searchsorted(T + pd.Timedelta(hours=1))
    step = pd.Timedelta(ex.index[1] - ex.index[0]) if len(ex) > 1 else pd.Timedelta(hours=1)
    hold = int(pd.Timedelta(hours=HOLD_H) / step)
    V, H, L, W, A, C = (rg[k].values for k in ("valid", "H", "L", "W", "atr", "c"))
    out: list[dict] = []; t = N + 14
    while t < len(T) - 2:
        if not V[t]:
            t += 1; continue
        if side == 1:
            entry, stop = L[t] + ENTRY_IN * W[t], L[t] - STOP_ATR * A[t]
            tgt = L[t] + 0.5 * W[t] if variant == "MID" else H[t] - ENTRY_IN * W[t]
            if C[t] <= entry:
                t += 1; continue
        else:
            entry, stop = H[t] - ENTRY_IN * W[t], H[t] + STOP_ATR * A[t]
            tgt = H[t] - 0.5 * W[t] if variant == "MID" else L[t] + ENTRY_IN * W[t]
            if C[t] >= entry:
                t += 1; continue
        k0, k1 = starts[t + 1], ends[t + 1]
        hit = np.nonzero(el[k0:k1] <= entry)[0] if side == 1 else np.nonzero(eh[k0:k1] >= entry)[0]
        if not len(hit):
            t += 1; continue
        k = k0 + int(hit[0]); risk = side * (entry - stop)
        loss = -1 - (MAKER + TAKER + SLIP) * entry / risk
        win = side * (tgt - entry) / risk - 2 * MAKER * entry / risk
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
                R, how = side * (ec[q] - entry) / risk - (MAKER + TAKER) * entry / risk, "time"
        out.append(dict(sym=sym, side=side, variant=variant, t=ex.index[k], R=R, how=how,
                        stop_pct=risk / entry * 100, gross_rr=side * (tgt - entry) / risk))
        t = max(int(T.searchsorted(ex.index[q], side="right")) - 1, t) + 1
    return out


def summ(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 6:
        return f"n={len(x)}"
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}"


def main() -> None:
    rows = []
    for sym in SPLIT:
        sig, ex = load(sym); rg = ranges(sig)
        print(f"{sym}: {rg.valid.mean() * 100:.1f}% of hours in a valid range", flush=True)
        for variant in ("MID", "FAR"):
            for side in (1, -1):
                rows += run_side(sym, sig, ex, rg, side, variant)
    T = pd.DataFrame(rows)
    T["per"] = ["held-out" if t >= pd.Timestamp(SPLIT[s]) else "design" for s, t in zip(T.sym, T.t)]
    T.to_csv(Path(__file__).with_name("range_grid_trades.csv"), index=False)
    for variant in ("MID", "FAR"):
        g = T[T.variant == variant]; d = g[g.per == "design"]; ho = g[g.per == "held-out"]
        mid = d.t.min() + (d.t.max() - d.t.min()) / 2
        print(f"\n{variant}: design {summ(d.R)} | halves {summ(d[d.t < mid].R)} / {summ(d[d.t >= mid].R)} | held-out {summ(ho.R)}")
        print(f"   win rate {(g.how == 'target').mean() * 100:.1f}% | median stop {g.stop_pct.median():.2f}% | median gross R:R {g.gross_rr.median():.2f}")
        print("   by symbol: " + " | ".join(f"{s}: {summ(d[d.sym == s].R)} / {summ(ho[ho.sym == s].R)}" for s in SPLIT))
        print("   longs " + summ(d[d.side == 1].R) + " / " + summ(ho[ho.side == 1].R) + " | shorts " + summ(d[d.side == -1].R) + " / " + summ(ho[ho.side == -1].R))


if __name__ == "__main__":
    main()
