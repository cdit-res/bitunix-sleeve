"""Batch 5, stock CFD side, pre-registered 25 Sep 2026 before running.

The single-stock tests in stocks_cfd.py draw on today's Bitunix list, which is a list of survivors, so any long-only
stock result carries survivorship bias (SC3 passed its gate but random entries did better). Index, country, sector
and commodity ETFs do not have that problem to the same degree, and Bitunix lists SPY, QQQ, IWM, SMH, XLE, XBI, GDX,
EWJ, EWT, EWY, EWZ, URNM, XAU, XAG, CL, BZ, NATGAS and COPPER perps.
Costs and entry timing as stocks_cfd.py. Design SPY, QQQ, GLD, SLV to 2016; held out 2017 on; fresh IWM, DIA, XLK, XLF,
XLE, SMH, XBI, GDX, EWJ, EWT, EWY, EWZ, USO, URNM (all dates). Three variants counted.

SC9  Trend ensemble on ETFs (T01 rules, lookbacks 20, 60, 150, 250, midpoint stop trailed daily), long only (SC9a)
     and long plus short (SC9b; shorts mirror the rule on closing lows, stop at the midpoint trailed down, no
     funding charged on shorts).
K08  Overnight drift after a down session (Boyarchenko, Larsen and Whelan, 2023): when a session's open-to-close
     return is below -0.5%, long at that close (the 20:00 UK run proxy), exit at the next open (the 08:30 UK run is
     earlier, 03:30 New York; the open is the nearest observable print); stop 1 x ATR(14), gap-aware at the open.
     Costs: maker in, taker plus slippage out, one day's funding.

Usage: FEED=/path/to/feed python batch5_etf.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import harness as H
import stocks_cfd as S

DESIGN = ("SPY", "QQQ", "GLD", "SLV")
FRESH = ("IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI", "GDX", "EWJ", "EWT", "EWY", "EWZ", "USO", "URNM")


def trend(sym: str, d: pd.DataFrame, side: int) -> list[dict]:
    c = d.c; L, Hh, O = d.l.values, d.h.values, d.o.values; out = []
    for n in (20, 60, 150, 250):
        mx, mn = c.rolling(n).max(), c.rolling(n).min(); mid = ((mx + mn) / 2).values
        ext = (mx if side == 1 else mn).shift(1).values; i = n + 1
        while i < len(d) - 1:
            ok = (c.iat[i] >= ext[i]) if side == 1 else (c.iat[i] <= ext[i])
            if not (ok and side * (c.iat[i] - mid[i]) / c.iat[i] >= 0.005):
                i += 1; continue
            px = c.iat[i]; s = mid[i]; j = i + 1; ex = None
            while j < len(d):
                if side * (O[j] - s) <= 0:
                    ex = O[j]; break
                if (side == 1 and L[j] <= s) or (side == -1 and Hh[j] >= s):
                    ex = s; break
                s = max(s, mid[j]) if side == 1 else min(s, mid[j]); j += 1
            if ex is None:
                break
            risk = side * (px - mid[i]); days = (d.index[j] - d.index[i]).days
            gross = side * (ex - px) / risk
            cost = (S.MAKER + S.TAKER + S.SLIP + (S.FUND * days if side == 1 else 0.0)) * px / risk
            out.append(dict(sym=sym, t=d.index[i], t_exit=d.index[j], side=side, gross=gross, cost=cost, net=gross - cost,
                            variant="base", N=n))
            i = j + 1
    return out


def k08(sym: str, d: pd.DataFrame, thr: float = -0.005) -> list[dict]:
    a = S.atr(d); out = []
    for i in range(20, len(d) - 1):
        if not d.c.iat[i] / d.o.iat[i] - 1 < thr:
            continue
        px = d.c.iat[i]; risk = a.iat[i]; o = d.o.iat[i + 1]
        ex = min(o, px - risk) if o <= px - risk else o  # a gap through the stop fills at the open
        days = (d.index[i + 1] - d.index[i]).days
        gross = (ex - px) / risk
        cost = (S.MAKER + S.TAKER + S.SLIP + S.FUND * days) * px / risk
        out.append(dict(sym=sym, t=d.index[i], t_exit=d.index[i + 1], side=1, gross=gross, cost=cost, net=gross - cost,
                        variant="base", move=(o / px - 1) * 100))
    return out


def main() -> None:
    data = {s: S.load(s) for s in DESIGN + FRESH}
    rows, gates = [], []

    def run(name, T):
        T = pd.DataFrame(T); g = H.gate(T, S.HELD, FRESH)
        rows.append(H.evaluate(name, T, asset="SPY, QQQ, GLD, SLV; fresh 14 ETFs", held_from=S.HELD, fresh=FRESH, ann=252))
        gates.append({"strategy": name, **g}); print(name, g, flush=True)
        return T

    run("SC9a trend ensemble long, ETFs", [r for s, d in data.items() for r in trend(s, d, 1)])
    Tb = run("SC9b trend ensemble long and short, ETFs", [r for s, d in data.items() for side in (1, -1) for r in trend(s, d, side)])
    sh = Tb[Tb.side == -1]; core = sh[~sh.sym.isin(FRESH)]
    print(f"   shorts alone: design {H.fmt(core[core.t < S.HELD].net)} | held-out {H.fmt(core[core.t >= S.HELD].net)} | fresh {H.fmt(sh[sh.sym.isin(FRESH)].net)}")
    T8 = run("K08 overnight drift after a down session, ETFs", [r for s, d in data.items() for r in k08(s, d)])
    core = T8[~T8.sym.isin(FRESH)]
    print(f"   mean overnight move {T8.move.mean():+.3f}% (design {core[core.t < S.HELD].move.mean():+.3f}%, held-out {core[core.t >= S.HELD].move.mean():+.3f}%); cost per trade {((S.MAKER + S.TAKER + S.SLIP + S.FUND) * 100):.3f}% before funding days")
    all_nights = pd.DataFrame([dict(sym=s, t=d.index[i], move=(d.o.iat[i + 1] / d.c.iat[i] - 1) * 100) for s, d in data.items() for i in range(20, len(d) - 1)])
    print(f"   control, every night: mean {all_nights.move.mean():+.3f}% (n {len(all_nights)})")
    print(); H.show(rows); print(); print(pd.DataFrame(gates).to_string(index=False))


if __name__ == "__main__":
    main()
