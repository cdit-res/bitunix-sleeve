"""Stock CFD (Bitunix stock perp) batch, pre-registered 25 Sep 2026 before any result was seen.

Data: Yahoo daily bars since listing, split and dividend adjusted, from the repo's data feed (data branch,
research/daily). The run fires at 15:00 New York (the 20:00 UK run); the daily close stands in for that price for
both the signal and the entry, which shifts every trade by the same hour and so does not bias the test, except
through last-hour seasonality.
Costs, per side on the underlying: maker 0.02% on entries, taker 0.06% plus 0.03% slippage on exits and stops;
funding 0.03% a calendar day on longs, none on shorts. Stops are gap-aware: an open through the stop fills at
the open (no guaranteed stop on stock perps).
Sets. Index strategies: design SPY and QQQ to 2016, held out 2017 on; fresh IWM, DIA, XLK, XLF, XLE, SMH, EWJ,
EWZ, EWY, EWT, GLD, TLT (all dates). Stock strategies: design = stocks listed before 2000, to 2016; held out 2017
on; fresh = stocks listed from 2000 (all dates).
Gate as elsewhere: design t >= 3, the same sign in both halves, held-out and fresh above zero. Variants below
are all counted (11).

SC1 Turn of the month (Lakonishok and Smidt, 1988; McConnell and Xu, 2008; Etula et al., 2020): long at the
    close of the month's last trading day, exit at the close of the third trading day; stop 2 x ATR(14).
SC2 Short-vol carry (Simon and Campasano, 2014): short UVXY at the close when VIX/VIX3M < 0.95 (SC2a) or < 0.90
    (SC2b); exit at the close when the ratio reaches 1.0, or after 20 days (re-entered if the signal holds);
    stop 2.5 x ATR(14) above entry. Design Oct 2011 to 2018, held out 2019 on. No fresh set exists.
SC3 Trend ensemble long on stocks (T01 rules; lookbacks 20, 60, 150, 250; midpoint stop trailed daily).
SC4 MAX-10 on stocks (T03 rules: close at or above the prior 10 closes, long at the close, stop 1 ATR,
    exit at the next close unless the signal repeats).
SC5 Weekly short-term reversal (Jegadeesh, 1990; Lehmann, 1990): each Friday close, long the 3 worst and short
    the 3 best 5-day performers among the design stocks, hold to the next Friday close, stop 2 x ATR per leg.
    SC5a unconditional; SC5b only when VIX closes at or above 20 (Nagel, 2012).
SC6 Index RSI-2 dips (Connors): close above SMA200 and RSI(2) below 10; long at the close; exit at the close
    once above SMA5, or after 10 days; stop 1.5 x ATR.
SC7 Index limit dip: when the close is above SMA200, a buy limit at close - 0.5 x ATR for the next session;
    if filled, stop 1 x ATR below the fill, exit at the next day's close.
SC8 Event-gap continuation (post-announcement drift; Chan, Jegadeesh and Lakonishok, 1996): an open gap of at
    least max(4%, 3 x the 60-day median absolute daily return) on volume at least 2.5 x its 60-day average, with
    the day closing beyond its open in the gap's direction; enter at that close, stop beyond the day's extreme
    (+0.1 ATR), exit at the close ten sessions later. SC8a longs only (up gaps); SC8b both sides.

Usage: FEED=/path/to/feed python stocks_cfd.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import harness as H

FEED = Path(os.environ.get("FEED", "/tmp/feed"))
MAKER, TAKER, SLIP, FUND = 0.0002, 0.0006, 0.0003, 0.0003
HELD = "2017-01-01"
IDX_DESIGN, IDX_FRESH = ("SPY", "QQQ"), ("IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "EWJ", "EWZ", "EWY", "EWT", "GLD", "TLT")
NON_STOCK = set(IDX_DESIGN + IDX_FRESH) | {"SOXL", "SOXS", "TQQQ", "SQQQ", "TMF", "TBT", "UVXY", "SVXY", "VXX", "NVDL", "TSLL",
                                          "KORU", "BITO", "GDX", "URNM", "XBI", "SLV", "USO"}


def load(t: str) -> pd.DataFrame:
    f = FEED / "research" / "daily" / f"{t.replace('^', 'IDX_').replace('=', '_').replace('.', '_')}.csv.gz"
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    d = d[d.index < pd.Timestamp.now().normalize()]  # completed sessions only
    d = d[(d.c > 0) & (d.h >= d.l)].dropna(subset=["o", "h", "l", "c"])
    return d


def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def stocks() -> tuple[list[str], list[str]]:
    names = sorted(p.name.split(".csv")[0] for p in (FEED / "research" / "daily").glob("*.csv.gz"))
    names = [n for n in names if not n.startswith("IDX_") and "_F" not in n and "_KS" not in n and "_HK" not in n
             and n not in NON_STOCK and n != "DX-Y_NYB"]
    design, fresh = [], []
    for n in names:
        first = pd.read_csv(FEED / "research" / "daily" / f"{n}.csv.gz", nrows=1, index_col=0, parse_dates=True).index[0]
        (design if first < pd.Timestamp("2000-01-01") else fresh).append(n)
    return design, fresh


def trade(sym: str, d: pd.DataFrame, i: int, side: int, stop: float, exit_at, variant: str = "base", entry: float | None = None) -> dict | None:
    """Enter at bar i's close (or `entry`), walk forward; exit_at(j) True means exit at bar j's close."""
    O, Hh, L, C = d.o.values, d.h.values, d.l.values, d.c.values
    px = C[i] if entry is None else entry; risk = side * (px - stop)
    if not np.isfinite(risk) or risk <= 0 or risk / px < 0.002:
        return None
    j = i + 1; ex = None
    while j < len(d):
        if side == 1 and O[j] <= stop or side == -1 and O[j] >= stop:
            ex = O[j]; break
        if side == 1 and L[j] <= stop or side == -1 and Hh[j] >= stop:
            ex = stop; break
        if exit_at(j):
            ex = C[j]; break
        j += 1
    if ex is None:
        return None
    days = (d.index[j] - d.index[i]).days
    gross = side * (ex - px) / risk
    cost = (MAKER + TAKER + SLIP + (FUND * days if side == 1 else 0.0)) * px / risk
    return dict(sym=sym, t=d.index[i], t_exit=d.index[j], side=side, gross=gross, cost=cost, net=gross - cost, variant=variant)


# ---------- strategies ----------
def sc1(sym: str, d: pd.DataFrame, control: bool = False) -> list[dict]:
    a = atr(d); idx = d.index; out = []
    last = idx.to_series().groupby([idx.year, idx.month]).transform("max") == idx.to_series()
    pos = {t: k for k, t in enumerate(idx)}
    days = idx[~last.values][::4] if control else idx[last.values][:-1]
    for t in days:
        i = pos[t]
        if i < 20 or i + 3 >= len(d):
            continue
        r = trade(sym, d, i, 1, d.c.iat[i] - 2 * a.iat[i], lambda j, i=i: j >= i + 3)
        if r: out.append(r)
    return out


def sc2(threshold: float, variant: str) -> list[dict]:
    u = load("UVXY"); v = load("^VIX").c; v3 = load("^VIX3M").c
    ratio = (v / v3).reindex(u.index).ffill(); a = atr(u); out = []; i = 20
    while i < len(u) - 1:
        if not ratio.iat[i] < threshold:
            i += 1; continue
        r = trade("UVXY", u, i, -1, u.c.iat[i] + 2.5 * a.iat[i], lambda j, i=i: ratio.iat[j] >= 1.0 or j >= i + 20, variant)
        if r is None:
            i += 1; continue
        out.append(r); i = u.index.get_loc(r["t_exit"])
    return out


def sc3(sym: str, d: pd.DataFrame, control: bool = False, seed: int = 0) -> list[dict]:
    """Trend ensemble component trades. control=True enters on random eligible days instead of breakouts."""
    c = d.c; L = d.l.values; O = d.o.values; out = []; rng = np.random.default_rng(seed)
    for n in (20, 60, 150, 250):
        mx, mn = c.rolling(n).max(), c.rolling(n).min(); mid = ((mx + mn) / 2).values; prev = mx.shift(1).values
        brk = (c.values >= prev) & ((c.values - mid) / c.values >= 0.005)
        elig = (c.values - mid) / c.values >= 0.005
        rate = brk[n + 1:].mean() if len(brk) > n + 1 else 0.0
        i = n + 1
        while i < len(d) - 1:
            hit = (elig[i] and rng.random() < rate / max(elig[n + 1:].mean(), 1e-9)) if control else brk[i]
            if not hit:
                i += 1; continue
            px = c.iat[i]; s = mid[i]; j = i + 1; exit_px = None
            while j < len(d):
                if O[j] <= s:
                    exit_px = O[j]; break
                if L[j] <= s:
                    exit_px = s; break
                s = max(s, mid[j]); j += 1
            if exit_px is None:
                break
            risk = px - mid[i]; days = (d.index[j] - d.index[i]).days
            gross = (exit_px - px) / risk; cost = (MAKER + TAKER + SLIP + FUND * days) * px / risk
            out.append(dict(sym=sym, t=d.index[i], t_exit=d.index[j], side=1, gross=gross, cost=cost, net=gross - cost,
                            variant="base", N=n))
            i = j + 1
    return out


def sc4(sym: str, d: pd.DataFrame, control: bool = False) -> list[dict]:
    c = d.c; a = atr(d); sig = (c >= c.rolling(10).max().shift(1)).values; out = []; i = 15
    if control:
        sig = np.ones(len(d), dtype=bool)
    while i < len(d) - 1:
        if not sig[i]:
            i += 1; continue
        r = trade(sym, d, i, 1, c.iat[i] - a.iat[i], lambda j: True)  # exit at the next close unless stopped
        if r:
            if sig[d.index.get_loc(r["t_exit"])] and r["t_exit"] != r["t"]:
                r["cost"] -= (TAKER + SLIP) * c.iat[i] / (a.iat[i])  # a roll does not pay the exit
                r["net"] = r["gross"] - r["cost"]
            out.append(r)
        i += 1
    return out


def sc5(universe: list[str], data: dict, vix: pd.Series, start: str = "1995-01-01") -> list[dict]:
    frames = {s: data[s] for s in universe}; atrs = {s: atr(f) for s, f in frames.items()}
    closes = pd.DataFrame({s: f.c for s, f in frames.items()})
    fridays = closes.index[closes.index.dayofweek == 4]
    ret5 = closes / closes.shift(5) - 1; out = []
    for t in fridays:
        if t < pd.Timestamp(start):
            continue
        r = ret5.loc[t].dropna()
        r = r[[s for s in r.index if frames[s].index.get_loc(t) > 200]]
        if len(r) < 10:
            continue
        cond = bool(vix.asof(t) >= 20)
        for side, names in ((1, r.nsmallest(3).index), (-1, r.nlargest(3).index)):
            for s in names:
                d = frames[s]; i = d.index.get_loc(t); a = atrs[s].iat[i]; nxt = d.index[i] + pd.Timedelta(days=7)
                row = trade(s, d, i, side, d.c.iat[i] - side * 2 * a, lambda j, d=d, nxt=nxt: d.index[j] >= nxt)
                if row:
                    out.append({**row, "variant": "a"})
                    if cond:
                        out.append({**row, "variant": "b"})
    return out


def rsi(c: pd.Series, n: int) -> pd.Series:
    x = c.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def sc6(sym: str, d: pd.DataFrame) -> list[dict]:
    c = d.c; r2 = rsi(c, 2); s200 = c.rolling(200).mean(); s5 = c.rolling(5).mean(); a = atr(d); out = []; i = 201
    while i < len(d) - 1:
        if not (c.iat[i] > s200.iat[i] and r2.iat[i] < 10):
            i += 1; continue
        r = trade(sym, d, i, 1, c.iat[i] - 1.5 * a.iat[i], lambda j, i=i: c.iat[j] > s5.iat[j] or j >= i + 10)
        if r is None:
            i += 1; continue
        out.append(r); i = d.index.get_loc(r["t_exit"]) + 1
    return out


def sc7(sym: str, d: pd.DataFrame) -> list[dict]:
    c = d.c; s200 = c.rolling(200).mean(); a = atr(d); out = []; i = 201
    while i < len(d) - 2:
        if not c.iat[i] > s200.iat[i]:
            i += 1; continue
        lim = c.iat[i] - 0.5 * a.iat[i]; k = i + 1
        if d.l.iat[k] > lim:
            i += 1; continue
        fill = min(lim, d.o.iat[k]); stop = fill - a.iat[i]
        # stop checked from the fill day's remaining range (conservative: the fill day's low counts)
        if d.l.iat[k] <= stop:
            gross = (stop - fill) / (fill - stop); ex_t = d.index[k]
        else:
            k2 = k + 1
            if d.o.iat[k2] <= stop:
                ex = d.o.iat[k2]
            elif d.l.iat[k2] <= stop:
                ex = stop
            else:
                ex = d.c.iat[k2]
            gross = (ex - fill) / (fill - stop); ex_t = d.index[k2]
        risk = fill - stop; days = (ex_t - d.index[k]).days + 1
        cost = (MAKER + TAKER + SLIP + FUND * days) * fill / risk
        out.append(dict(sym=sym, t=d.index[k], t_exit=ex_t, side=1, gross=gross, cost=cost, net=gross - cost, variant="base"))
        i = d.index.get_loc(ex_t) + 1
    return out


def sc8_control(sym: str, d: pd.DataFrame, n: int, seed: int = 0) -> list[dict]:
    a = atr(d); rng = np.random.default_rng(seed); out = []
    if len(d) < 80 or n == 0:
        return out
    for i in sorted(rng.choice(np.arange(61, len(d) - 11), size=min(n, len(d) - 72), replace=False)):
        r = trade(sym, d, i, 1, d.l.iat[i] - 0.1 * a.iat[i], lambda j, i=i: j >= i + 10, "control")
        if r: out.append(r)
    return out


def sc8(sym: str, d: pd.DataFrame, both: bool) -> list[dict]:
    if "v" not in d or d.v.fillna(0).eq(0).mean() > 0.2:
        return []
    c, o = d.c, d.o; a = atr(d); pc = c.shift(1); gap = o / pc - 1
    med = (c.pct_change().abs()).rolling(60).median().shift(1); vavg = d.v.rolling(60).mean().shift(1); out = []; i = 61
    while i < len(d) - 11:
        g = gap.iat[i]
        big = abs(g) >= max(0.04, 3 * med.iat[i]) and d.v.iat[i] >= 2.5 * vavg.iat[i]
        side = 1 if g > 0 else -1
        if not big or (side == -1 and not both) or not (side * (c.iat[i] - o.iat[i]) > 0):
            i += 1; continue
        stop = d.l.iat[i] - 0.1 * a.iat[i] if side == 1 else d.h.iat[i] + 0.1 * a.iat[i]
        r = trade(sym, d, i, side, stop, lambda j, i=i: j >= i + 10, "b" if both else "a")
        if r is None:
            i += 1; continue
        out.append(r); i = d.index.get_loc(r["t_exit"]) + 1
    return out


def main() -> None:
    rows = []; gates = []
    idx_syms = IDX_DESIGN + IDX_FRESH
    idx = {s: load(s) for s in idx_syms}
    design, fresh = stocks()
    data = {s: load(s) for s in design + fresh}
    print(f"stocks: {len(design)} design ({', '.join(design)}), {len(fresh)} fresh", flush=True)

    def ctl(name, T, fresh_set):
        T = pd.DataFrame(T); core = T[~T.sym.isin(set(fresh_set))]
        print(f"   {name}: design {H.fmt(core[core.t < HELD].net)} | held-out {H.fmt(core[core.t >= HELD].net)} | fresh {H.fmt(T[T.sym.isin(set(fresh_set))].net)}", flush=True)

    def run(name, T, asset, fresh_set, held=HELD, primary=None):
        T = pd.DataFrame(T)
        g = H.gate(T[T.variant == primary] if primary else T, held, fresh_set)
        rows.append(H.evaluate(name, T, asset=asset, held_from=held, fresh=fresh_set, ann=252, primary=primary))
        gates.append({"strategy": name, **g}); print(name, g, flush=True)

    run("SC1 turn of the month", [r for s, d in idx.items() for r in sc1(s, d)], "SPY, QQQ; fresh 12 ETFs", IDX_FRESH)
    ctl("SC1 control: same rule on other days", [r for s, d in idx.items() for r in sc1(s, d, True)], IDX_FRESH)
    T2 = sc2(0.95, "a") + sc2(0.90, "b")
    run("SC2a short UVXY, VIX/VIX3M < 0.95", T2, "UVXY", (), "2019-01-01", primary="a")
    run("SC2b short UVXY, VIX/VIX3M < 0.90", T2, "UVXY", (), "2019-01-01", primary="b")
    run("SC3 trend ensemble long, stocks", [r for s, d in data.items() for r in sc3(s, d)], f"{len(design)} + {len(fresh)} stocks", fresh)
    ctl("SC3 control: random entry days", [r for k in range(3) for s, d in data.items() for r in sc3(s, d, True, k)], fresh)
    run("SC4 MAX-10, stocks", [r for s, d in data.items() for r in sc4(s, d)], f"{len(design)} + {len(fresh)} stocks", fresh)
    ctl("SC4 control: long every day", [r for s, d in data.items() for r in sc4(s, d, True)], fresh)
    vix = load("^VIX").c
    T5 = sc5(design, data, vix) + sc5(fresh, data, vix, "2012-01-01")
    run("SC5a weekly reversal", T5, f"{len(design)} + {len(fresh)} stocks", fresh, primary="a")
    run("SC5b weekly reversal, VIX >= 20", T5, f"{len(design)} + {len(fresh)} stocks", fresh, primary="b")
    run("SC6 RSI-2 dips, indices", [r for s, d in idx.items() for r in sc6(s, d)], "SPY, QQQ; fresh 12 ETFs", IDX_FRESH)
    run("SC7 limit dip, indices", [r for s, d in idx.items() for r in sc7(s, d)], "SPY, QQQ; fresh 12 ETFs", IDX_FRESH)
    T8 = [r for s, d in data.items() for r in sc8(s, d, False)] + [r for s, d in data.items() for r in sc8(s, d, True)]
    run("SC8a event-gap drift, longs", T8, f"{len(design)} + {len(fresh)} stocks", fresh, primary="a")
    run("SC8b event-gap drift, both sides", T8, f"{len(design)} + {len(fresh)} stocks", fresh, primary="b")
    na = pd.DataFrame(T8).query("variant == 'a'").groupby("sym").size()
    ctl("SC8 control: random days, same stop and hold", [r for s, d in data.items() for r in sc8_control(s, d, int(na.get(s, 0)) * 3)], fresh)
    print(); H.show(rows); print(); print(pd.DataFrame(gates).to_string(index=False))


if __name__ == "__main__":
    main()
