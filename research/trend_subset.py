"""Robustness variant (counted): the trend ensemble with three lookbacks (20, 60, 150) instead of nine, the version
The owner could run by hand. Same rules, rail, data and sets as trend_ensemble.py with MIN_STOP=0.005."""
import os
os.environ.setdefault("MIN_STOP", "0.005")
import numpy as np, pandas as pd
import trend_ensemble as TE
TE.LOOKBACKS = (20, 60, 150)
res, pnl = [], {}
for s in TE.CORE + TE.FRESH:
    h4, d = TE.bars(s); tr, p = TE.t1(s, h4, d, "INTRADAY"); res += tr; pnl[s] = p
T = pd.DataFrame(res); core = T[T.sym.isin(TE.CORE)]; des = core[core.t < TE.HELD_FROM]
mid = des.t.min() + (des.t.max() - des.t.min()) / 2
print("3 lookbacks: design", TE.summ(des.R), "| halves", TE.summ(des[des.t < mid].R), "/", TE.summ(des[des.t >= mid].R),
      "| held-out", TE.summ(core[core.t >= TE.HELD_FROM].R), "| fresh", TE.summ(T[T.sym.isin(TE.FRESH)].R))
P = pd.DataFrame(pnl).fillna(0.0); pc = P[list(TE.CORE)].sum(axis=1); pf = P[list(TE.FRESH)].sum(axis=1)
pdz = pc[pc.index < TE.HELD_FROM]; pdz = pdz[pdz.index >= pdz.ne(0).idxmax()]
print("   daily: design", TE.daily_t(pdz), "| held-out", TE.daily_t(pc[pc.index >= TE.HELD_FROM]), "| fresh", TE.daily_t(pf[pf.index >= pf.ne(0).idxmax()]))
print(f"   trades per coin-year: {len(T) / sum((TE.bars(s)[1].index[-1] - TE.bars(s)[1].index[0]).days / 365 for s in TE.CORE + TE.FRESH):.1f} | win rate {(T.R > 0).mean() * 100:.0f}% | median hold {T.days.median():.1f} d | median stop {T.stop_pct.median():.1f}%")
