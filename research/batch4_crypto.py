"""Batch 4, crypto, pre-registered 25 Sep 2026 before running. Every variant counted (7 plus 2 improvements).

Windows are the ticket windows: morning 08:00 to 19:00 UTC (the 08:30 run to the 20:00 run) and evening 19:00 to
08:00 UTC. Costs: maker 0.02%, taker 0.06%, slippage 0.02%, funding 0.03% a day on longs (0.015% a window).
Sets: BTC, ETH, SOL, XRP design to 2023, held out 2024 on; LINK, BNB, ADA, DOGE, AVAX, DOT fresh.

N1 Taker-flow continuation (order-flow persistence; Binance taker buy volume, 1h, 2022-26): at each window start
   the 24h taker buy share is ranked against the previous 180 window starts; top 20% long, bottom 20% short.
   Entry at the window open (maker), target 2R (maker), else exit at the window end.
   N1a stop 0.25 x daily ATR(14); N1b stop 0.5 x ATR; N1c the reverse side (exhaustion), stop 0.25 x ATR.
N2 Round-number fade (Osler, 2003: take-profit orders cluster at round numbers): at each window start, a buy limit
   at the first round level below price and a sell limit at the first above (step = 10^(digits - 2), e.g. 1,000 on
   BTC at 84,000), each at least 0.25 x ATR away and within 1.5 x ATR; stop 0.5 x ATR beyond, target 1 x ATR (2R),
   exit at the window end. Control, not a variant: the same at the half-step levels (83,500, 84,500).
   BTC on Bitstamp 5m (2017 on); other coins on 1h.
N4 Asia-range bracket at the 08:30 run: the 00:00 to 07:00 UTC range; buy stop at its high, sell stop at its low,
   each stop at the range midpoint, target 2R, exit 19:00 UTC; skipped when half the range is under 0.5%.
   Taker entries.
N5 Turtle soup (Raschke and Connors, 1995): a new 20-day low with the previous 20-day low at least four days old;
   next day a buy stop at that previous low, stop 0.1 x ATR under the new low, target 2R, else exit at the third
   close. Shorts mirror. Daily signals, 4h bars for fills.
C3 Funding filter on MAX-10 (T03): skip a signal when the coin's funding averaged over the prior three days is in
   the top fifth of its prior 180 days. BTC and ETH only (funding 2020-23 and Aug 2025 on).
C4 Volatility-scaled trend ensemble (Moreira and Muir, 2017): each T01 trade weighted by the coin's 180-day median
   daily volatility over its 30-day volatility at entry, capped 0.25 to 2. Compared on Sharpe with T01 as is.

Usage: SLEEVE_DATA=/path/to/data python batch4_crypto.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import harness as H  # noqa: E402
import trend_ensemble as TE  # noqa: E402

DATA = TE.DATA; MAKER, TAKER, SLIP, FUND = TE.MAKER, TE.TAKER, TE.SLIP, TE.FUND
CORE = ("BTC", "ETH", "SOL", "XRP"); FRESH = ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
HELD = "2024-01-01"


def taker(sym: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "taker" / f"{sym}USDT_1h_taker.parquet")


def daily_atr(h: pd.DataFrame) -> pd.Series:
    d = h.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / 14, adjust=False).mean()
    a.index = a.index + pd.Timedelta(days=1)  # known from the next day's 00:00 UTC
    return a


def window_trade(sym, bars, k0, k1, side, entry, stop, tgt, variant, t0, entry_fee=MAKER, open_entry=None):
    """Stop checked before target in every bar; on a limit or stop fill bar the target is not credited, since
    the bar's extreme may have come before the fill. Gaps through the stop fill at the open."""
    Hh, L, O = bars.h.values, bars.l.values, bars.o.values
    open_entry = (entry_fee == MAKER and variant in ("a", "b", "c")) if open_entry is None else open_entry
    risk = side * (entry - stop); R = None; k_exit = k1
    for q in range(k0, k1):
        if (side == 1 and L[q] <= stop) or (side == -1 and Hh[q] >= stop):
            px = stop if q == k0 else (min(stop, O[q]) if side == 1 else max(stop, O[q]))
            R = (side * (px - entry) / risk, entry_fee + TAKER + SLIP); k_exit = q; break
        if (q > k0 or open_entry) and ((side == 1 and Hh[q] >= tgt) or (side == -1 and L[q] <= tgt)):
            R = (side * (tgt - entry) / risk, entry_fee + MAKER); k_exit = q; break
    if R is None:
        ex = O[k1] if k1 < len(O) else bars.c.values[-1]
        R = (side * (ex - entry) / risk, entry_fee + TAKER + SLIP)
    hours = (bars.index[min(k_exit, len(bars) - 1)] - bars.index[k0]).total_seconds() / 3600
    gross, fee = R
    cost = fee * entry / risk + (FUND * hours / 24 * entry / risk if side == 1 else 0.0)
    return dict(sym=sym, t=t0, t_exit=bars.index[min(k_exit, len(bars) - 1)], side=side, gross=gross, cost=cost,
                net=gross - cost, variant=variant)


def windows(idx: pd.DatetimeIndex):
    days = pd.date_range(idx[0].normalize() + pd.Timedelta(days=2), idx[-1].normalize() - pd.Timedelta(days=1), freq="D")
    for day in days:
        for lab, a, b in (("morning", 8, 19), ("evening", 19, 32)):
            yield lab, day + pd.Timedelta(hours=a), day + pd.Timedelta(hours=b)


# ---------- N1 taker flow ----------
def n1(sym: str) -> list[dict]:
    h = taker(sym); atr = daily_atr(h); idx = h.index; out = []
    share = (h.tb.rolling(24).sum() / h.v.rolling(24).sum()).shift(1)  # known at the bar's open
    starts = []
    for lab, t0, t1 in windows(idx):
        k0, k1 = idx.searchsorted(t0), idx.searchsorted(t1)
        if k1 >= len(idx) or idx[k0] != t0:
            continue
        starts.append((t0, k0, k1, share.iat[k0]))
    vals = np.array([s[3] for s in starts])
    for n, (t0, k0, k1, s) in enumerate(starts):
        if n < 180 or not np.isfinite(s):
            continue
        rank = (vals[n - 180:n] < s).mean(); a = atr.asof(t0)
        if not np.isfinite(a) or (0.2 < rank < 0.8):
            continue
        side = 1 if rank >= 0.8 else -1; entry = h.o.iat[k0]
        for var, mult, sgn in (("a", 0.25, 1), ("b", 0.5, 1), ("c", 0.25, -1)):
            sd = side * sgn; risk = mult * a
            if risk / entry < 0.005:
                continue
            out.append(window_trade(sym, h, k0, k1, sd, entry, entry - sd * risk, entry + sd * 2 * risk, var, t0))
    return out


# ---------- N2 round numbers ----------
def n2_bars(sym: str) -> pd.DataFrame:
    if sym == "BTC":
        return pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
    return taker(sym)[["o", "h", "l", "c"]]


def n2(sym: str, control: bool) -> list[dict]:
    b = n2_bars(sym); atr = daily_atr(b.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna())
    idx = b.index; out = []
    for lab, t0, t1 in windows(idx):
        k0, k1 = idx.searchsorted(t0), idx.searchsorted(t1)
        if k1 >= len(idx) or idx[k0] != t0:
            continue
        p = b.o.iat[k0]; a = atr.asof(t0)
        if not np.isfinite(a) or p <= 0:
            continue
        step = 10 ** (np.floor(np.log10(p)) - 1); off = step / 2 if control else 0.0
        lo = np.floor((p - off) / step) * step + off; hi = np.ceil((p - off) / step) * step + off
        while p - lo < 0.25 * a: lo -= step
        while hi - p < 0.25 * a: hi += step
        for side, lvl in ((1, lo), (-1, hi)):
            if abs(lvl - p) > 1.5 * a or 0.5 * a / lvl < 0.005:
                continue
            L, Hh = b.l.values, b.h.values
            fill = next((q for q in range(k0, k1) if (L[q] <= lvl if side == 1 else Hh[q] >= lvl)), None)
            if fill is None:
                continue
            out.append(window_trade(sym, b, fill, k1, side, lvl, lvl - side * 0.5 * a, lvl + side * 1.0 * a,
                                    "control" if control else "base", t0))
    return out


# ---------- N4 Asia bracket ----------
def n4(sym: str) -> list[dict]:
    b = n2_bars(sym); idx = b.index; out = []
    for day in pd.date_range(idx[0].normalize() + pd.Timedelta(days=1), idx[-1].normalize() - pd.Timedelta(days=1), freq="D"):
        a0, a1, r0, r1 = (day + pd.Timedelta(hours=x) for x in (0, 7, 7.5, 19))
        ka0, ka1 = idx.searchsorted(a0), idx.searchsorted(a1); kr0, kr1 = idx.searchsorted(r0), idx.searchsorted(r1)
        if kr1 >= len(idx) or ka1 - ka0 < 6 or kr1 <= kr0:
            continue
        hi, lo = b.h.values[ka0:ka1].max(), b.l.values[ka0:ka1].min(); mid = (hi + lo) / 2
        if (hi - mid) / mid < 0.005:
            continue
        for side, lvl in ((1, hi), (-1, lo)):
            L, Hh = b.l.values, b.h.values
            fill = next((q for q in range(kr0, kr1) if (Hh[q] >= lvl if side == 1 else L[q] <= lvl)), None)
            if fill is None:
                continue
            entry = max(lvl, b.o.values[fill]) if side == 1 else min(lvl, b.o.values[fill])
            entry *= (1 + side * SLIP); risk = side * (entry - mid)
            if risk <= 0:
                continue
            out.append(window_trade(sym, b, fill, kr1, side, entry, mid, entry + side * 2 * risk, "base", day + pd.Timedelta(hours=7.5),
                                    entry_fee=TAKER))
    return out


# ---------- N5 turtle soup ----------
def n5(sym: str) -> list[dict]:
    h4, d = TE.bars(sym if sym == "BTC" else f"{sym}USDT"); t4 = h4.index; out = []
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values; L, Hh = d.l.values, d.h.values
    for i in range(25, len(d) - 5):
        for side in (1, -1):
            win = L[i - 20:i] if side == 1 else Hh[i - 20:i]
            prev = win.min() if side == 1 else win.max(); age = 20 - (int(np.argmin(win)) if side == 1 else int(np.argmax(win)))
            new = L[i] < prev if side == 1 else Hh[i] > prev
            if not (new and age >= 4):
                continue
            stop = L[i] - 0.1 * atr[i] if side == 1 else Hh[i] + 0.1 * atr[i]
            k0 = t4.searchsorted(d.index[i + 1]); k1 = t4.searchsorted(d.index[i + 4])
            if k1 >= len(t4):
                continue
            fill = next((q for q in range(k0, t4.searchsorted(d.index[i + 2])) if (h4.h.values[q] >= prev if side == 1 else h4.l.values[q] <= prev)), None)
            if fill is None:
                continue
            entry = (max(prev, h4.o.values[fill]) if side == 1 else min(prev, h4.o.values[fill])) * (1 + side * SLIP)
            risk = side * (entry - stop)
            if risk <= 0 or risk / entry < 0.005:
                continue
            out.append(window_trade(sym, h4, fill, k1, side, entry, stop, entry + side * 2 * risk, "base", d.index[i + 1], entry_fee=TAKER))
    return out


# ---------- T01 and T03 in the table, C3 and C4 ----------
def t01_trades() -> pd.DataFrame:
    rows = []
    for s in TE.CORE + TE.FRESH:
        h4, d = TE.bars(s); tr, _ = TE.t1(s, h4, d, "INTRADAY")
        for r in tr:
            cost = (2 * TAKER + FUND * r["days"]) * 100 / r["stop_pct"]
            rows.append(dict(sym=s.replace("USDT", ""), t=r["t"], t_exit=r["exit"], side=1, gross=r["R"] + cost, cost=cost,
                             net=r["R"], variant="base", N=r["N"]))
    return pd.DataFrame(rows)


def t03_trades(filt: dict | None = None) -> pd.DataFrame:
    rows = []
    for s in TE.CORE + TE.FRESH:
        h4, d = TE.bars(s); O, L = h4.o.values, h4.l.values; t4 = h4.index; dc = d.c.values; days = d.index
        tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values
        sig = dc >= pd.Series(dc).rolling(10).max().shift(1).values
        if filt is not None:
            f = filt.get(s.replace("USDT", ""))
            if f is None:
                continue
            ok = np.array([not bool(f.asof(t + pd.Timedelta(days=1))) if t + pd.Timedelta(days=1) >= f.index[0] else True for t in days])
            sig = sig & ok
        at8 = t4.searchsorted(days + pd.Timedelta(hours=8)); i = 15
        while i < len(days) - 3:
            if not sig[i]:
                i += 1; continue
            k = at8[i + 1]
            if k >= len(O) - 7:
                break
            entry = O[k]; first = True
            while True:
                stop = entry - atr[i]; k_end = min(at8[i + 2], len(O) - 1); ex = None
                for q in range(k, k_end):
                    if L[q] <= stop:
                        ex = min(stop, O[q]) * (1 - SLIP); break
                roll = ex is None and sig[i + 1] and i + 1 < len(days) - 3
                if ex is None:
                    ex = O[k_end] if roll else O[k_end] * (1 - SLIP)
                fee = (MAKER if first else 0.0) + (0.0 if roll else TAKER)
                cost = (fee + FUND) * entry / atr[i]; gross = (ex - entry) / atr[i]
                rows.append(dict(sym=s.replace("USDT", ""), t=t4[k], t_exit=t4[k_end], side=1, gross=gross, cost=cost,
                                 net=gross - cost, variant="base"))
                i += 1
                if not roll:
                    break
                entry, k, first = ex, k_end, False
            i += 1
    return pd.DataFrame(rows)


def funding_flags() -> dict:
    out = {}
    for s in ("BTC", "ETH"):
        a = pd.read_csv(DATA / f"supervik_{s}_binance.csv"); a["t"] = pd.to_datetime(a.Date) - pd.Timedelta(hours=3)
        a = a.set_index("t")["Funding Rate"].astype(float)
        b = pd.read_parquet(DATA / "funding_binance.parquet"); b = b[b.canonical_symbol == f"{s}-USDT-PERP"]
        b = b.set_index(pd.to_datetime(b.settlement_ts, utc=True).dt.tz_localize(None)).rate_raw.astype(float)
        f = pd.concat([a, b]).sort_index(); f = f[~f.index.duplicated()]
        dly = f.resample("1D").mean().dropna(); m3 = dly.rolling(3).mean()
        q80 = m3.rolling(180, min_periods=90).quantile(0.8).shift(1)
        flag = (m3 > q80) & q80.notna()
        flag.index = flag.index + pd.Timedelta(days=1)
        out[s] = flag
    return out


def main() -> None:
    rows, gates = [], []

    def run(name, T, primary=None, fresh=FRESH, held=HELD, asset="BTC, ETH, SOL, XRP; fresh six", ann=365):
        T = pd.DataFrame(T)
        if T.empty:
            print(name, "no trades"); return T
        g = H.gate(T[T.variant == primary] if primary else T, held, fresh)
        rows.append(H.evaluate(name, T, asset=asset, held_from=held, fresh=fresh, ann=ann, primary=primary))
        gates.append({"strategy": name, **g}); print(name, g, flush=True); return T

    T1 = t01_trades(); run("T01 trend ensemble (baseline)", T1)
    T3 = t03_trades(); run("T03 MAX-10 (baseline)", T3)
    N1 = [r for s in CORE + FRESH for r in n1(s)]
    for v, lab in (("a", "N1a taker flow, 0.25 ATR stop"), ("b", "N1b taker flow, 0.5 ATR stop"), ("c", "N1c taker flow reversed")):
        run(lab, N1, primary=v)
    N2 = [r for s in CORE + FRESH for r in n2(s, False)]; run("N2 round-number fade", N2)
    C2 = pd.DataFrame([r for s in CORE + FRESH for r in n2(s, True)])
    core = C2[C2.sym.isin(CORE)]
    print(f"   N2 control, half-step levels: design {H.fmt(core[core.t < HELD].net)} | held-out {H.fmt(core[core.t >= HELD].net)} | fresh {H.fmt(C2[C2.sym.isin(FRESH)].net)}")
    run("N4 Asia-range bracket", [r for s in CORE + FRESH for r in n4(s)])
    run("N5 turtle soup", [r for s in CORE + FRESH for r in n5(s)])
    ff = funding_flags(); T3f = t03_trades(ff)
    T3bf = T3[T3.sym.isin(["BTC", "ETH"])]
    for lab, T in (("BTC, ETH unfiltered", T3bf), ("BTC, ETH funding filter", T3f)):
        print(f"   C3 {lab}: 2020-23 {H.fmt(T[(T.t >= '2020-01-01') & (T.t < '2024-01-01')].net)} | Aug 2025 on {H.fmt(T[T.t >= '2025-08-22'].net)}")
    # C4: volatility-scaled T01
    w = []
    for s in T1.sym.unique():
        h4, d = TE.bars(s if s == "BTC" else f"{s}USDT"); r = d.c.pct_change()
        v30 = r.rolling(30).std(); med = v30.rolling(180, min_periods=90).median()
        wt = (med / v30).clip(0.25, 2.0); wt.index = wt.index + pd.Timedelta(days=1)
        g = T1[T1.sym == s]; w += [float(wt.asof(t)) if np.isfinite(wt.asof(t)) else 1.0 for t in g.t]
    T1w = T1.copy(); T1w[["gross", "cost", "net"]] = T1w[["gross", "cost", "net"]].mul(np.array(w), axis=0)
    for lab, T in (("T01 as is", T1), ("C4 volatility-scaled", T1w)):
        core = T[T.sym.isin(CORE)]
        print(f"   {lab}: Sharpe all {H.sharpe(T, 365):.2f} | design {H.sharpe(core[core.t < HELD], 365):.2f} | held-out {H.sharpe(core[core.t >= HELD], 365):.2f} | fresh {H.sharpe(T[T.sym.isin(FRESH)], 365):.2f} | max DD {H.max_dd(T):.1f}R")
    print(); H.show(rows); print(); print(pd.DataFrame(gates).to_string(index=False))


if __name__ == "__main__":
    main()
