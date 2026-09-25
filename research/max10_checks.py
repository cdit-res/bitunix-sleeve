"""Robustness checks on T3 MAX-10 (not new hypotheses): per coin and year, taker entry, no stop, lookback
plateau, and the continuation short (SHORT10, counted as a variant). Same data and costs as trend_ensemble.py."""
from __future__ import annotations
import numpy as np, pandas as pd
import trend_ensemble as TE

def run(sym, h4, d, n=10, side=1, entry_fee=TE.MAKER, stop_mult=1.0):
    O, L, Hh = h4.o.values, h4.l.values, h4.h.values; t4 = h4.index; dc = d.c.values; days = d.index
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values
    if side == 1:
        sig = dc >= pd.Series(dc).rolling(n).max().shift(1).values
    else:
        sig = dc <= pd.Series(dc).rolling(n).min().shift(1).values
    at8 = t4.searchsorted(days + pd.Timedelta(hours=8)); out = []; i = n + 5
    while i < len(days) - 3:
        if not sig[i]:
            i += 1; continue
        k = at8[i + 1]
        if k >= len(O) - 7:
            break
        entry = O[k]; first = True
        while True:
            stop = entry - side * stop_mult * atr[i] if stop_mult else None
            k_end = min(at8[i + 2], len(O) - 1); ex = None
            if stop is not None:
                for q in range(k, k_end):
                    if (side == 1 and L[q] <= stop) or (side == -1 and Hh[q] >= stop):
                        ex = (min(stop, O[q]) if side == 1 else max(stop, O[q])) * (1 - side * TE.SLIP); break
            roll = ex is None and sig[i + 1] and i + 1 < len(days) - 3
            if ex is None:
                ex = O[k_end] if roll else O[k_end] * (1 - side * TE.SLIP)
            cost = (entry_fee if first else 0.0) + (0.0 if roll else TE.TAKER)
            fund = TE.FUND if side == 1 else -0.0  # shorts usually receive funding; counted as zero to stay conservative
            R = side * (ex - entry) / atr[i] - cost * entry / atr[i] - fund * entry / atr[i]
            out.append(dict(sym=sym, t=t4[k], R=R))
            i += 1
            if not roll:
                break
            entry, k, first = ex, k_end, False
        i += 1
    return pd.DataFrame(out)

def sets(T):
    core = T[T.sym.isin(TE.CORE)]
    return core[core.t < TE.HELD_FROM], core[core.t >= TE.HELD_FROM], T[T.sym.isin(TE.FRESH)]

data = {s: TE.bars(s) for s in TE.CORE + TE.FRESH}
def allrun(**kw):
    return pd.concat([run(s, *data[s], **kw) for s in data])

base = allrun()
des, ho, fr = sets(base)
print("BASE MAX10 (maker entry, 1 ATR stop):", TE.summ(des.R), "|", TE.summ(ho.R), "|", TE.summ(fr.R))
oos = pd.concat([ho, fr]); print("  pooled out-of-sample (time + fresh):", TE.summ(oos.R))
print("  by coin (design / time held-out or fresh):")
for s in TE.CORE + TE.FRESH:
    g = base[base.sym == s]
    print(f"    {s:9s} {TE.summ(g[g.t < TE.HELD_FROM].R):32s} {TE.summ(g[g.t >= TE.HELD_FROM].R)}")
base["yr"] = base.t.dt.year
print("  by year, all coins: " + " | ".join(f"{y}: {g.R.mean():+.3f} ({len(g)})" for y, g in base.groupby("yr")))
tk = allrun(entry_fee=TE.TAKER + TE.SLIP); a, b, c = sets(tk)
print("TAKER ENTRY:", TE.summ(a.R), "|", TE.summ(b.R), "|", TE.summ(c.R))
ns = allrun(stop_mult=0); a, b, c = sets(ns)
print("NO STOP (R per ATR):", TE.summ(a.R), "|", TE.summ(b.R), "|", TE.summ(c.R))
for n in (5, 8, 12, 15, 20, 30):
    x = allrun(n=n); a, b, c = sets(x)
    print(f"LOOKBACK {n:2d}:", TE.summ(a.R), "|", TE.summ(b.R), "|", TE.summ(c.R))
sh = allrun(side=-1); a, b, c = sets(sh)
print("SHORT10 (sell a 10-day closing low, counted variant):", TE.summ(a.R), "|", TE.summ(b.R), "|", TE.summ(c.R))
# portfolio view: core coins, 5% risk per trade, daily P&L as % of equity (simple, non-compounding)
core = base[base.sym.isin(TE.CORE)].copy(); core["day"] = core.t.dt.normalize()
daily = core.groupby("day").R.sum() * 5.0
idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D"); daily = daily.reindex(idx, fill_value=0.0)
for lab, p in (("design", daily[daily.index < TE.HELD_FROM]), ("held-out", daily[daily.index >= TE.HELD_FROM])):
    sharpe = p.mean() / p.std() * np.sqrt(365)
    cum = p.cumsum(); dd = (cum - cum.cummax()).min()
    print(f"PORTFOLIO core 4 at 5% risk, {lab}: mean {p.mean():+.3f}%/day, Sharpe {sharpe:.2f}, worst peak-to-trough {dd:.1f} points, trades/day {len(core[core.t < TE.HELD_FROM]) / max(len(p),1):.2f}")
same = core.groupby("day").size(); print("  days with 1/2/3/4 coins signalling:", same.value_counts().sort_index().to_dict())
# one bet per day: 5% risk split across the core coins signalling that day, compounding
def book(frame):
    f = frame.copy(); f["day"] = f.t.dt.normalize(); n = f.groupby("day").R.transform("size")
    f["pct"] = f.R * 5.0 / n
    dly = f.groupby("day").pct.sum(); idx = pd.date_range(dly.index.min(), dly.index.max(), freq="D")
    dly = dly.reindex(idx, fill_value=0.0); eq = (1 + dly / 100).cumprod(); dd = (eq / eq.cummax() - 1).min() * 100
    yrs = len(dly) / 365; cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    return dly.mean() / dly.std() * np.sqrt(365), cagr, dd
for lab, f in (("design", core[core.t < TE.HELD_FROM]), ("held-out", core[core.t >= TE.HELD_FROM])):
    sh, cg, dd = book(f); print(f"ONE BET PER DAY (5% split), {lab}: Sharpe {sh:.2f}, CAGR {cg:.0f}%, max drawdown {dd:.0f}%")
