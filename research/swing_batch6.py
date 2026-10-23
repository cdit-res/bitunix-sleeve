"""Swing batch 6, pre-registered 23 Oct 2026 before running: the bigger crypto cross-section the perp archive allows.

Universe: the 50 perps in research/perps, a coin counting only on days it trades (seven-day median quote volume
>= $1m, 60-day volatility >= 20%; see swing_funding.panel). Weekly, decided on Sunday's close, held from the next
close; perp returns plus funding as posted; 0.08% a side on turnover. Halves by alphabetical position: A design
(2020 to 2023) and held out (2024 on), B fresh. Placebo: 200 draws of random ranks. Gate as before.

G1 Cross-sectional momentum on 50 perps (Liu, Tsyvinski and Wu, 2022): rank on the 28-day return; long the top
   fifth, short the bottom fifth, equal risk; at least 20 eligible. H3 failed on ten coins; this asks whether a
   wider cross-section changes that.
G2 Cross-sectional low volatility (lottery demand; Bali, Cakici and Whitelaw, 2011): long the fifth with the lowest
   60-day volatility, short the highest fifth, equal risk; at least 20 eligible. Counterparty: buyers who overpay
   for lottery-like coins.
Check (not a variant): SW10 in a book of three rows at 4% across all 50 perps, core four first, against the core
four alone, October 2020 on, marked to market.

Usage: python swing_batch6.py
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
import swing_funding as F  # noqa: E402

TV, SIDE, N_PLACEBO, DESIGN_END = 0.10, 0.0008, 200, "2023-12-31"
rng = np.random.default_rng(20261023)


def book_ls(syms: list[str], score: str, shuffle: bool = False, mult: float = 1.0) -> pd.Series:
    R, V, Fd, F7 = F.panel(syms); idx = R.index
    C = pd.concat({s: F.bars(s).c for s in syms}, axis=1).reindex(idx)
    elig = F7.notna()  # live, 60 days of funding (panel's rule)
    sc = (C / C.shift(28) - 1) if score == "mom" else -V  # higher is better for the long leg
    sc = sc.where(elig)
    reb = ~pd.Series(idx.to_period("W-SUN"), index=idx).duplicated(); W = pd.DataFrame(np.nan, index=idx, columns=syms)
    for i in np.flatnonzero(reb.values):
        if i < 1:
            continue
        t = idx[i - 1]; s = sc.loc[t].dropna()
        if len(s) < 20:
            W.iloc[i] = 0.0; continue
        order = list(s.sort_values(ascending=False).index)
        if shuffle:
            order = list(rng.permutation(order))
        k = len(order) // 5; row = pd.Series(0.0, index=syms)
        for x in order[:k]:
            row[x] = TV / max(V.loc[t, x], 0.2) / (2 * k)
        for x in order[-k:]:
            row[x] = -TV / max(V.loc[t, x], 0.2) / (2 * k)
        W.iloc[i] = row.values
    W = W.ffill().fillna(0.0).clip(-3, 3); H = W.shift(1).fillna(0.0)
    net = (H * R.fillna(0.0)).sum(axis=1) - (H * Fd.fillna(0.0)).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0.0) * SIDE * mult
    live = W.abs().sum(axis=1) > 0
    return net[idx >= (live.idxmax() if live.any() else idx[-1])]


def judge(name: str, A: list[str], Bs: list[str], score: str) -> bool:
    x = book_ls(A, score); des = x[x.index <= DESIGN_END]; held = x[x.index > DESIGN_END]
    mid = des.index[0] + (des.index[-1] - des.index[0]) / 2
    print(f"  {name}, A: design {F.sr(des)} [halves {F.sr(des[des.index < mid])} / {F.sr(des[des.index >= mid])}]; held-out {F.sr(held)}")
    xb = book_ls(Bs, score); print(f"    fresh B: {F.sr(xb)}")
    pl = [book_ls(A, score, shuffle=True).mean() for _ in range(N_PLACEBO)]; p95 = np.percentile(pl, 95)
    print(f"    placebo: real {x.mean() * 1e4:+.2f}bp a day vs 95th percentile {p95 * 1e4:+.2f}bp")
    dd = book_ls(A, score, mult=2.0); dd = dd[dd.index <= DESIGN_END]; print(f"    double costs (design): {F.sr(dd)}")
    r = dict(des=B.tstat(des), h1=B.tstat(des[des.index < mid]), h2=B.tstat(des[des.index >= mid]), held=held.mean())
    return B.verdict(name, r, xb.mean(), x.mean() > p95, B.tstat(dd))


def sw_book(syms: list[str], core: set[str], start: str = "2020-10-01", slots: int = 3, risk: float = 0.04) -> None:
    cands, data, atr = [], {}, {}
    for s in syms:
        d = B.crypto(s) if s in core else F.bars(s); data[s] = d; a = E.atr(d).values; atr[s] = a
        c, lo, o = d.c.values, d.l.values, d.o.values; sig = (d.c >= d.c.rolling(10).max().shift(1)).values
        if s not in core:
            q = F.qvol(s).reindex(d.index); live = (q.rolling(7, min_periods=5).median() >= 1e6).values
        else:
            live = np.ones(len(d), bool)
        k = 20
        while k < len(d) - 1:
            if not sig[k] or not live[k] or not np.isfinite(a[k]) or a[k] / c[k] < 0.005:
                k += 1; continue
            st = c[k] - a[k]; j = k + 1
            while j < len(d):
                if o[j] <= st or lo[j] <= st or c[j] < c[j - 3:j].min() or j >= k + 10:
                    break
                j += 1
            if j >= len(d):
                break
            cands.append((d.index[k], s not in core, s, k, j)); k = j + 1
    cands.sort(key=lambda x: (x[0], x[1], x[2])); days = pd.date_range(start, B.END); pnl = pd.Series(0.0, index=days); open_until = []
    for t, _, s, k, j in cands:
        if t < pd.Timestamp(start):
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
    yrs = len(pnl) / 365.25; eq = (1 + risk * pnl).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    print(f"    {len(syms)} coins: {pnl.sum() / yrs:+.1f}R a year, daily SR {pnl.mean() / pnl.std() * np.sqrt(365):.2f}, worst drawdown {dd * 100:.0f}%; "
          + ", ".join(f"{y} {v:+.1f}R" for y, v in pnl.groupby(pnl.index.year).sum().items()))


def main() -> None:
    syms = list(F.perps()); A, Bs = syms[0::2], syms[1::2]; res = {}
    print("G1 cross-sectional momentum, 50 perps"); res["G1"] = judge("G1", A, Bs, "mom")
    print("\nG2 cross-sectional low volatility, 50 perps"); res["G2"] = judge("G2", A, Bs, "lowvol")
    print("\nCheck: SW10 book, three rows at 4%, core four first")
    core = {"BTC", "ETH", "SOL", "XRP"}
    sw_book(sorted(core), core); sw_book(sorted(core) + [s for s in syms if s not in core], core)
    print("\nPASSED: " + (", ".join(k for k, v in res.items() if v) or "none"))


if __name__ == "__main__":
    main()
