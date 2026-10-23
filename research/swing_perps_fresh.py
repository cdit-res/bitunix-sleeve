"""F5, pre-registered 23 Oct 2026 before the data arrived: SW10 on every perp in research/perps outside the ten coins
already used (BTC, ETH, SOL, XRP and the fresh six), on Binance perp daily bars from 2020, delisted coins included.
Fresh evidence for SW10, not a new variant. Funding charged as posted. Placebo: a long every day with the same exits.
Reported pooled, by coin, by year, and for delisted coins separately.

Usage: python swing_perps_fresh.py
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

USED = {"BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT"}
DELISTED = {"LUNA", "FTT", "SRM", "MATIC"}


def run(sym: str, placebo: bool = False) -> pd.DataFrame:
    d = F.bars(sym); c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values; n = len(d)
    f = F.funding(sym); fday = f.groupby(f.index.normalize()).sum().reindex(d.index).fillna(0.0).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; rows = []; k = 20
    while k < n - 1:
        if not (placebo or sig[k]) or not np.isfinite(a[k]) or a[k] <= 0:
            k += 1; continue
        e, st = c[k], c[k] - a[k]; ex = None; j = k + 1
        while j < n:
            if o[j] <= st or lo[j] <= st:
                ex = min(st, o[j]); break
            if c[j] < c[j - 3:j].min() or j >= k + 10:
                ex = c[j]; break
            j += 1
        if ex is None:
            break
        R = (ex - e) / a[k] - (2 * B.SIDE_CR + fday[k + 1:j + 1].sum()) * e / a[k]
        rows.append(dict(sym=sym, t=d.index[k], R=R))
        k = k + 1 if placebo else j + 1
    return pd.DataFrame(rows)


def main() -> None:
    syms = [s for s in F.perps() if s not in USED]
    S = pd.concat([run(s) for s in syms]); P = pd.concat([run(s, True) for s in syms])
    print(f"F5 SW10 on {len(syms)} further perps (2020 on): signal {B.fmtR(S.R)}; placebo {B.fmtR(P.R)}; difference {S.R.mean() - P.R.mean():+.3f}R")
    beat = sum(S[S.sym == s].R.mean() > P[P.sym == s].R.mean() for s in syms if (S.sym == s).sum() >= 5)
    print(f"  coins where the signal beats its placebo: {beat} of {sum((S.sym == s).sum() >= 5 for s in syms)}")
    for y, g in S.groupby(S.t.dt.year):
        print(f"  {y}: signal {B.fmtR(g.R)}; placebo {B.fmtR(P[P.t.dt.year == y].R)}")
    dl = S[S.sym.isin(DELISTED)]; pdl = P[P.sym.isin(DELISTED)]
    print(f"  delisted or renamed ({', '.join(sorted(DELISTED & set(syms)))}): signal {B.fmtR(dl.R)}; placebo {B.fmtR(pdl.R)}")
    for s in syms:
        g = S[S.sym == s]
        if len(g):
            print(f"    {s}: {B.fmtR(g.R)}; placebo mean {P[P.sym == s].R.mean():+.3f}R")


if __name__ == "__main__":
    main()
