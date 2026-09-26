"""Placebo controls for the two passing crypto edges (disproof attempt, 26 Sep 2026).

T01: the same trailing midpoint exit, entered on random days with the same 0.5% stop room, matched in count per
lookback (five seeds). T03: the same 1 ATR stop and next-morning exit, entered every day (unconditional long).
An edge in the entry signal shows as signal minus control above zero in every set.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import harness as H  # noqa: E402
import trend_ensemble as TE  # noqa: E402
import batch4_crypto as B4  # noqa: E402


def t01_control(sym: str, seed: int) -> list[dict]:
    h4, d = TE.bars(sym); dc = d.c.values; lo = d.l.values; op = d.o.values; rng = np.random.default_rng(seed); out = []
    for N in TE.LOOKBACKS:
        mx = pd.Series(dc).rolling(N).max().values; mn = pd.Series(dc).rolling(N).min().values; mid = (mx + mn) / 2
        hi = pd.Series(dc).rolling(N).max().shift(1).values
        brk = (dc >= hi) & ((dc - mid) / dc >= 0.005); elig = (dc - mid) / dc >= 0.005
        p = brk[N + 1:].mean() / max(elig[N + 1:].mean(), 1e-9); i = N + 1
        while i < len(d) - 2:
            if not (elig[i] and rng.random() < p):
                i += 1; continue
            entry = op[i + 1] * (1 + TE.SLIP); s = mid[i]; risk = entry - s
            if risk <= 0 or risk / entry < 0.005:
                i += 1; continue
            j = i + 1; ex = None
            while j < len(d):
                if op[j] <= s and j > i + 1: ex = op[j]; break
                if lo[j] <= s: ex = s; break
                s = max(s, mid[j]); j += 1
            if ex is None: break
            days = (d.index[j] - d.index[i + 1]).days
            cost = (2 * TE.TAKER + TE.FUND * days) * entry / risk; gross = (ex * (1 - TE.SLIP) - entry) / risk
            out.append(dict(sym=sym.replace("USDT", ""), t=d.index[i + 1], t_exit=d.index[j], gross=gross, cost=cost, net=gross - cost))
            i = j + 1
    return out


def t03_control(sym: str) -> list[dict]:
    h4, d = TE.bars(sym); O, L = h4.o.values, h4.l.values; t4 = h4.index; days = d.index
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values; at8 = t4.searchsorted(days + pd.Timedelta(hours=8)); out = []
    for i in range(15, len(days) - 3):
        k = at8[i + 1]; k_end = min(at8[i + 2], len(O) - 1)
        if k >= len(O) - 7: break
        entry = O[k]; stop = entry - atr[i]; ex = None
        for q in range(k, k_end):
            if L[q] <= stop: ex = min(stop, O[q]) * (1 - TE.SLIP); break
        if ex is None: ex = O[k_end] * (1 - TE.SLIP)
        cost = (TE.MAKER + TE.TAKER + TE.FUND) * entry / atr[i]; gross = (ex - entry) / atr[i]
        out.append(dict(sym=sym.replace("USDT", ""), t=t4[k], t_exit=t4[k_end], gross=gross, cost=cost, net=gross - cost))
    return out


def compare(name: str, sig: pd.DataFrame, ctl: pd.DataFrame) -> None:
    def sets(T):
        core = T[T.sym.isin(B4.CORE)]
        return {"design": core[core.t < B4.HELD].net, "held-out": core[core.t >= B4.HELD].net, "fresh": T[T.sym.isin(B4.FRESH)].net}
    a, b = sets(sig), sets(ctl)
    print(f"{name}:")
    for k in a:
        diff = a[k].mean() - b[k].mean(); se = np.sqrt(a[k].var() / len(a[k]) + b[k].var() / len(b[k]))
        print(f"   {k}: signal {H.fmt(a[k])} | control {H.fmt(b[k])} | difference {diff:+.3f}R (t {diff / se:+.1f})")


if __name__ == "__main__":
    syms = TE.CORE + TE.FRESH
    compare("T01 trend ensemble vs random entry days, same exit", B4.t01_trades(),
            pd.DataFrame([r for k in range(5) for s in syms for r in t01_control(s, k)]))
    compare("T03 MAX-10 vs long every day, same stop and hold", B4.t03_trades(), pd.DataFrame([r for s in syms for r in t03_control(s)]))
