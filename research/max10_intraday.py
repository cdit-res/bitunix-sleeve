"""MAX-10 as a day trade, two variants pre-registered 25 Sep 2026 before running (Cole: day trading on 15m).
D1: on a MAX-10 signal, long at the 08:00 UTC open (maker), stop 1.0 daily ATR, exit at 21:00 UTC the same day
    (US close area) at market. No rolling; each signal day is one day trade.
D2: on a MAX-10 signal, a maker limit at the 08:00 UTC open minus 0.25 daily ATR (a 15m pullback entry), valid until
    20:00 UTC; if filled, stop 1.0 daily ATR below the fill, exit at the next 08:00 UTC open.
Execution on 1h bars where available (BTC Bitstamp from 2016; alts 2022 onward), stop first. Sets and gate as trend_ensemble.py,
with the alt design ending 2023-12-31 as elsewhere for daily-signal tests."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
os.environ.setdefault("MIN_STOP", "0.005")
import trend_ensemble as TE
import remaining_lines as RL
DATA = TE.DATA

def hourly(sym):
    if sym == "BTC":
        return pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet").resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return pd.read_csv(DATA / f"{sym}_1h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)

def run(sym, variant):
    h = hourly(sym); d = h.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values; dc = d.c.values
    sig = dc >= pd.Series(dc).rolling(10).max().shift(1).values
    O, H, L, C = (h[k].values for k in ("o", "h", "l", "c")); idx = h.index; out = []
    for i in range(15, len(d) - 2):
        if not sig[i]:
            continue
        day = d.index[i + 1]
        k8 = idx.searchsorted(day + pd.Timedelta(hours=8)); k20 = idx.searchsorted(day + pd.Timedelta(hours=20))
        k21 = idx.searchsorted(day + pd.Timedelta(hours=21)); k_next8 = idx.searchsorted(day + pd.Timedelta(hours=32))
        if k_next8 >= len(idx) or idx[k8] != day + pd.Timedelta(hours=8):
            continue
        a = atr[i]
        if variant == "D1":
            entry = O[k8]; stop = entry - a; end = k21; fee_in = TE.MAKER; k = k8
        else:
            lvl = O[k8] - 0.25 * a; k = None
            for q in range(k8, k20):
                if L[q] <= lvl:
                    k = q; break
            if k is None:
                continue
            entry = min(lvl, O[k]); stop = entry - a; end = k_next8; fee_in = TE.MAKER
        R = None
        for q in range(k, end):
            if L[q] <= stop:
                R = -1 - (fee_in + TE.TAKER + TE.SLIP) * entry / a; break
        if R is None:
            ex = O[end] * (1 - TE.SLIP); days = (idx[end] - idx[k]).total_seconds() / 86400
            R = (ex - entry) / a - (fee_in + TE.TAKER) * entry / a - TE.FUND * days * entry / a
        out.append(dict(sym=sym, t=idx[k], R=R))
    return out

for v in ("D1", "D2"):
    T = pd.DataFrame([r for s in TE.CORE + TE.FRESH if s == "BTC" or (DATA / f"{s}_1h.csv.gz").exists() for r in run(s, v)])
    core = T[T.sym.isin(TE.CORE)]; des = core[core.t < TE.HELD_FROM]
    print(f"{v}: design {RL.summ(des.R)} | halves {RL.halves(des)} | held-out {RL.summ(core[core.t >= TE.HELD_FROM].R)} | fresh {RL.summ(T[T.sym.isin(TE.FRESH)].R)}")
