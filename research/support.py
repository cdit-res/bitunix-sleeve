"""Hypothesis F, tested-level method: enter at a tested support or a rising trendline, at the level only;
stop just beyond the invalidating level; after a stop, re-enter at the next support down.
Mirror for shorts at resistance. Levels from 4h bars, execution on 1h bars, stop first.

Pre-registered settings (25 Sep 2026): pivots k=3; supports need 2+ pivot touches within 0.5 ATR(4h)
in the last 120 bars and no 4h close through them since; stop 0.25 ATR beyond the zone (0.5 ATR for
trendlines); target 3R; order valid 30 bars; cancel if price runs 4 ATR away; hold up to 240h; 3 rungs.
Usage: python support.py design|holdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
MAKER, TAKER, SLIP = 0.0002, 0.0006, 0.0002
WINDOW, K, TOL, BUF_H, BUF_T = 120, 3, 0.5, 0.25, 0.5
VALID, RUNAWAY, HOLD, MAX_RUNGS, TARGET_R = 30, 4.0, 240, 3, 3.0
SYMS = ("BTC", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def load(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = DATA / ("BTC_bitstamp_1h.csv.gz" if sym == "BTC" else f"{sym}_1h.csv.gz")
    h1 = pd.read_csv(f, index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)
    h4 = h1.resample("4h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return h1, h4


def btc_trend() -> pd.Series:
    h1, _ = load("BTC")
    d = h1.c.resample("1D").last().dropna(); e = d.ewm(span=50, adjust=False).mean()
    lab = np.where((d > e) & (e > e.shift(5)), "up", np.where((d < e) & (e < e.shift(5)), "down", "flat"))
    return pd.Series(lab, index=d.index + pd.Timedelta(days=1))


def pivots(v: np.ndarray, side: int) -> list[int]:
    out = []
    for i in range(K, len(v) - K):
        w = v[i - K:i + K + 1]
        if (side == 1 and v[i] == w.min()) or (side == -1 and v[i] == w.max()):
            out.append(i)
    return out


class Market:
    def __init__(self, sym: str):
        self.h1, self.h4 = load(sym)
        h4 = self.h4
        tr = pd.concat([h4.h - h4.l, (h4.h - h4.c.shift()).abs(), (h4.l - h4.c.shift()).abs()], axis=1).max(axis=1)
        self.atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values
        self.H, self.L, self.C = h4.h.values, h4.l.values, h4.c.values
        self.plo, self.phi = pivots(self.L, 1), pivots(self.H, -1)
        self.pos = self.h1.index.searchsorted(h4.index)  # first 1h row of each 4h bar
        self.o1, self.h1v, self.l1, self.c1 = (self.h1[c].values for c in ("o", "h", "l", "c"))

    def levels(self, i: int, side: int) -> list[tuple[float, float]]:
        """Tested, unbroken levels on the entry side of price, nearest first: (level, stop)."""
        vals, piv, a = (self.L, self.plo, self.atr[i]) if side == 1 else (self.H, self.phi, self.atr[i])
        pts = sorted((vals[p], p) for p in piv if i - WINDOW <= p <= i - K)
        clusters: list[list[tuple[float, int]]] = []
        for v, p in pts:
            if clusters and abs(v - clusters[-1][-1][0]) <= TOL * a:
                clusters[-1].append((v, p))
            else:
                clusters.append([(v, p)])
        out = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            lvl = float(np.mean([v for v, _ in cl])); last = max(p for _, p in cl)
            edge = min(v for v, _ in cl) if side == 1 else max(v for v, _ in cl)
            closes = self.C[last:i + 1]
            if (side == 1 and (closes < edge).any()) or (side == -1 and (closes > edge).any()):
                continue
            if (side == 1 and lvl < self.C[i]) or (side == -1 and lvl > self.C[i]):
                out.append((lvl, edge - side * BUF_H * a))
        out.sort(key=lambda x: -x[0] * side)
        return out

    def trendline(self, i: int, side: int):
        vals, piv = (self.L, self.plo) if side == 1 else (self.H, self.phi)
        ps = [p for p in piv if i - WINDOW <= p <= i - K]
        if len(ps) < 2:
            return None
        p1, p2 = ps[-2], ps[-1]
        if p2 - p1 < 5 or (side == 1 and vals[p2] <= vals[p1]) or (side == -1 and vals[p2] >= vals[p1]):
            return None
        s = (vals[p2] - vals[p1]) / (p2 - p1)
        js = np.arange(p1, i + 1); line = vals[p1] + s * (js - p1)
        if (side == 1 and (self.C[js] < line).any()) or (side == -1 and (self.C[js] > line).any()):
            return None
        return lambda j: vals[p1] + s * (j - p1)


def new_order(m: Market, i: int, side: int, variant: str, rung: int, cap: float | None) -> dict | None:
    a = m.atr[i]
    if variant == "T" and rung == 1:
        tl = m.trendline(i, side)
        if tl is not None and ((side == 1 and tl(i + 1) < m.C[i]) or (side == -1 and tl(i + 1) > m.C[i])):
            return dict(kind="trendline", line=tl, gap=BUF_T * a, placed=i, rung=rung)
    lv = m.levels(i, side)
    if cap is not None:
        lv = [x for x in lv if (side == 1 and x[0] < cap) or (side == -1 and x[0] > cap)]
    if not lv:
        return None
    return dict(kind="support" if side == 1 else "resistance", level=lv[0][0], stop=lv[0][1], placed=i, rung=rung)


def run(sym: str, side: int, variant: str) -> list[dict]:
    m = Market(sym); trades = []; order = None; i = 200; n4 = len(m.h4)
    while i < n4 - 2:
        if order is None:
            order = new_order(m, i, side, variant, 1, None)
            if order is None:
                i += 1; continue
        j = i + 1
        if j - order["placed"] > VALID:
            order = None; i = j; continue
        if order["kind"] == "trendline":
            level = order["line"](j); stop = level - side * order["gap"]
        else:
            level, stop = order["level"], order["stop"]
        if (side == 1 and m.C[i] > level + RUNAWAY * m.atr[i]) or (side == -1 and m.C[i] < level - RUNAWAY * m.atr[i]):
            order = None; i = j; continue
        k0, k1 = m.pos[j], (m.pos[j + 1] if j + 1 < n4 else len(m.c1))
        fill = None
        for k in range(k0, k1):
            if (side == 1 and m.o1[k] < level) or (side == -1 and m.o1[k] > level):
                fill = (k, m.o1[k], TAKER); break
            if (side == 1 and m.l1[k] <= level) or (side == -1 and m.h1v[k] >= level):
                fill = (k, level, MAKER); break
        if fill is None:
            i = j; continue
        k, entry, fee_in = fill
        risk = side * (entry - stop)
        if risk <= 0:  # gapped through the stop: level already broken
            order = new_order(m, j, side, variant, order["rung"] + 1, stop) if order["rung"] < MAX_RUNGS else None
            i = j; continue
        tgt = entry + side * TARGET_R * risk; end = min(k + HOLD, len(m.c1)); r = None; q = end - 1
        for q in range(k, end):
            if (side == 1 and m.l1[q] <= stop) or (side == -1 and m.h1v[q] >= stop):
                r = -1 - (fee_in + TAKER + SLIP) * entry / risk; out = "stop"; break
            if q > k and ((side == 1 and m.h1v[q] >= tgt) or (side == -1 and m.l1[q] <= tgt)):
                r = TARGET_R - (fee_in + MAKER) * entry / risk; out = "target"; break
        if r is None:
            r = side * (m.c1[q] - entry) / risk - (fee_in + TAKER) * entry / risk; out = "time"
        t = m.h1.index[k]
        trades.append(dict(sym=sym, side=side, variant=variant, rung=order["rung"], kind=order["kind"], t=t, R=r,
                           out=out, stop_pct=risk / entry * 100))
        jx = int(m.h4.index.searchsorted(m.h1.index[q], side="right") - 1)
        if out == "stop" and order["rung"] < MAX_RUNGS:
            order = new_order(m, jx, side, variant, order["rung"] + 1, stop)
        else:
            order = None
        i = max(jx, j)
    return trades


def summ(x: pd.Series) -> str:
    x = x.dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "design"
    tr = btc_trend()
    T = pd.DataFrame([t for s in SYMS for side in (1, -1) for v in ("H", "T") for t in run(s, side, v)])
    T["btc"] = [tr.asof(x) for x in T.t]
    T = T[T.t < "2025-01-01"] if period == "design" else T[T.t >= "2025-01-01"]
    half = T.t.min() + (T.t.max() - T.t.min()) / 2
    print(f"period: {period}, {T.t.min().date()} to {T.t.max().date()}")
    for v, lab in (("H", "horizontal support ladder"), ("T", "trendline first, then support ladder")):
        for side in (1, -1):
            g = T[(T.variant == v) & (T.side == side)]
            print(f"\n{lab}, {'longs at support' if side == 1 else 'shorts at resistance'}: all {summ(g.R)}")
            print(f"   halves: {summ(g[g.t < half].R)} | {summ(g[g.t >= half].R)}")
            print("   by rung: " + " | ".join(f"rung {r}: {summ(g[g.rung == r].R)}" for r in (1, 2, 3)))
            print("   by symbol: " + " | ".join(f"{s}: {summ(g[g.sym == s].R)}" for s in SYMS))
            print("   by BTC trend: " + " | ".join(f"{b}: {summ(g[g.btc == b].R)}" for b in ("up", "flat", "down")))
            print("   stops >= 0.5%: " + summ(g[g.stop_pct >= 0.5].R) + f" | median stop {g.stop_pct.median():.2f}%")
            if v == "T":
                print("   trendline entries only: " + summ(g[g.kind == "trendline"].R))
