"""Swing batch 2, pre-registered 23 Oct 2026 before running, after batch 1 (swing_search.py) found nothing.

Lesson from batch 1: on perps a hold of weeks pays funding on the whole notional, so a slow signal must beat simply
holding the same markets by more than the funding. Batch 2 therefore looks for short holds driven by forced flows.

B1 Month-end rebalancing front-run (Harvey, Mazzoleni and Melone, 2025, NBER w33554). Counterparty: pension and
   balanced funds that rebalance a stock-bond mix to target at month end, selling the asset that outperformed.
   Signal: a 60/40 SPY/TLT mix reset to target at each month's last trading day; at the close five sessions before
   month end, dev = equity weight - 0.6; z = dev / its expanding standard deviation across prior months (24 needed).
   Position p = -z capped at +-2, entered at the next open (four sessions before month end), exited at the month's
   last close (five sessions held).
   B1a: trade SPY alone.  B1b: SPY minus TLT, the TLT leg sized to SPY's 60-day volatility.
   Fresh: the same p traded on QQQ and on IWM. Placebo: 200 draws, each month a random five-session window that ends
   before the last five sessions, with p from the same drift signal the day before; must beat the 95th percentile.
B2 Option-expiration week (Stivers and Sun, 2013). Counterparty: option dealers unwinding delta hedges into expiry.
   Long from the close before the week of the third Friday to the close of the expiry day (the last trading day
   on or before the third Friday). Design SPY and QQQ; fresh IWM and DIA. Placebo: 200 draws of a random non-expiry
   week in the same year for each event; must beat the 95th percentile.
B3 MAX-10 as a swing hold (owner asked for swing). Long at the close at a 10-day closing high, stop 1 daily ATR,
   exit at a close below the prior three closes or after ten days; one position per coin. Core BTC, ETH, SOL, XRP
   (design to 2023, held out 2024 on); fresh LINK, BNB, ADA, DOGE, AVAX, DOT. Placebo: a long on every day with the
   same stop and exits; the signal must beat it in every set.
B4 RSI-2 index dips (SC6) entered at the next open, the price the 08:30 run can actually get. Close above SMA200 and
   RSI(2) below 10; enter at the next open; stop 1.5 ATR; exit at the close once above SMA5, or after ten days.
   Design SPY and QQQ to 2016, held out 2017 on; fresh IWM, DIA, XLK, XLF, XLE, SMH, EWJ, EWZ, EWY, EWT, GLD, TLT.
   Placebo: every close above SMA200, same entry and exits; the signal must beat it in every set.

Costs: tradfi taker 0.06% plus 0.03% slippage each side, funding 0.0105% a calendar day on notional, longs and shorts
alike; crypto 0.08% each side, longs pay 0.03% a day. Gate: design t >= 3 (event or trade level); the same sign in
both design halves; held-out and fresh above zero; the placebo test above; t >= 2 at double costs.

Usage: python swing_batch2.py
"""
from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E  # noqa: E402

END = "2026-10-22"
DESIGN_END, CRYPTO_DESIGN_END = "2016-12-31", "2023-12-31"
SIDE_TF, F_TF = 0.0009, 0.000105
SIDE_CR, F_CR = 0.0008, 0.0003
N_PLACEBO = 200
rng = np.random.default_rng(20261023)


@lru_cache(maxsize=None)
def tradfi(sym: str) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/daily/{E.safe(sym)}.csv.gz")), compression="gzip",
                    index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)
    d = d[(d.c > 0) & (d.o > 0) & (d.h >= d.l)].dropna(); d.index = d.index.normalize()
    return d[~d.index.duplicated()].sort_index().loc[:END]


@lru_cache(maxsize=None)
def crypto(sym: str) -> pd.DataFrame:
    d = E.completed(E.static_klines(sym, "1d", "2016-01-01"), "1d")[["o", "h", "l", "c"]].astype(float)
    d.index = d.index.normalize()
    return d.loc[:END]


def tstat(x) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else float("nan")


def fmt(x) -> str:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"n {len(x)}, mean {x.mean() * 100:+.3f}%, t {tstat(x):+.2f}" if len(x) > 5 else f"n {len(x)}"


def fmtR(x) -> str:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"n {len(x)}, mean {x.mean():+.3f}R, t {tstat(x):+.2f}" if len(x) > 5 else f"n {len(x)}"


def halves(s: pd.Series, end: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    des = s[s.index <= end]; held = s[s.index > end]
    mid = des.index[0] + (des.index[-1] - des.index[0]) / 2
    return des, des[des.index < mid], des[des.index >= mid], held


def report(name: str, s: pd.Series, end: str, fn=fmt) -> dict:
    des, h1, h2, held = halves(s.sort_index(), end)
    print(f"  {name}: design {fn(des)} [halves {fn(h1)} / {fn(h2)}]; held-out {fn(held)}")
    return dict(des=tstat(des), h1=tstat(h1), h2=tstat(h2), held=held.mean() if len(held) else np.nan)


def verdict(name: str, r: dict, fresh: float, placebo_ok: bool, dbl_t: float) -> bool:
    checks = {"design t >= 3": r["des"] >= 3, "both halves positive": r["h1"] > 0 and r["h2"] > 0,
              "held-out positive": r["held"] > 0, "fresh positive": fresh > 0, "placebo beaten": placebo_ok,
              "double costs t >= 2": dbl_t >= 2}
    ok = all(checks.values())
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} ({'; '.join(k for k, v in checks.items() if not v) or 'all checks met'})")
    return ok


# ---------- B1 month-end rebalancing ----------
def b1_frame() -> pd.DataFrame:
    s, b = tradfi("SPY"), tradfi("TLT"); idx = s.index.intersection(b.index)
    s, b = s.loc[idx], b.loc[idx]
    last = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("max") == pd.Series(idx, index=idx)
    ends = list(np.flatnonzero(last.values))
    rows = []
    for m in range(1, len(ends)):
        T, Tp = ends[m], ends[m - 1]
        if T - Tp < 12:
            continue
        def dev(k: int) -> float:
            ge, gb = s.c.iat[k] / s.c.iat[Tp], b.c.iat[k] / b.c.iat[Tp]
            return 0.6 * ge / (0.6 * ge + 0.4 * gb) - 0.6
        rows.append(dict(t=idx[T], T=T, Tp=Tp, sig=T - 5, dev=dev(T - 5),
                         alt=[(k, dev(k - 1)) for k in range(Tp + 2, T - 8)]))  # placebo windows k..k+4 end before T-4
    F = pd.DataFrame(rows).set_index("t")
    sd = F.dev.expanding().std().shift(1); F["z"] = F.dev / sd; F.loc[F.index[:24], "z"] = np.nan
    F["p"] = (-F.z).clip(-2, 2)
    F["sd"] = sd
    return F


def window_ret(d: pd.DataFrame, idx: pd.DatetimeIndex, k0: int, k1: int) -> tuple[float, int]:
    """Open of session k0 to close of session k1 on the shared index; calendar days held."""
    t0, t1 = idx[k0], idx[k1]
    return d.c.loc[t1] / d.o.loc[t0] - 1, (t1 - t0).days + 1


def b1(F: pd.DataFrame, trade: str, mult: float = 1.0, spread: bool = False) -> pd.Series:
    s, b = tradfi("SPY"), tradfi("TLT"); idx = s.index.intersection(b.index); d = tradfi(trade)
    vs = s.c.pct_change().rolling(60).std(); vb = b.c.pct_change().rolling(60).std(); out = {}
    for t, r in F.dropna(subset=["p"]).iterrows():
        k0, k1 = r["T"] - 4, r["T"]
        if idx[k0] not in d.index or idx[k1] not in d.index:
            continue
        x, days = window_ret(d, idx, k0, k1); p = r["p"]
        net = p * x - abs(p) * (2 * SIDE_TF + F_TF * days) * mult
        if spread:
            k = vs.loc[idx[r["sig"]]] / vb.loc[idx[r["sig"]]]; y, _ = window_ret(b, idx, k0, k1)
            net += -p * k * y - abs(p) * k * (2 * SIDE_TF + F_TF * days) * mult
        out[t] = net
    return pd.Series(out)


def b1_placebo(F: pd.DataFrame, spread: bool) -> list[float]:
    s, b = tradfi("SPY"), tradfi("TLT"); idx = s.index.intersection(b.index)
    vs = s.c.pct_change().rolling(60).std(); vb = b.c.pct_change().rolling(60).std(); means = []
    G = F.dropna(subset=["p"])
    for _ in range(N_PLACEBO):
        vals = []
        for t, r in G.iterrows():
            if not r["alt"]:
                continue
            k, dv = r["alt"][rng.integers(len(r["alt"]))]; p = float(np.clip(-dv / r["sd"], -2, 2))
            x, days = window_ret(s, idx, k, k + 4); net = p * x - abs(p) * (2 * SIDE_TF + F_TF * days)
            if spread:
                kk = vs.loc[idx[k - 1]] / vb.loc[idx[k - 1]]; y, _ = window_ret(b, idx, k, k + 4)
                net += -p * kk * y - abs(p) * kk * (2 * SIDE_TF + F_TF * days)
            vals.append(net)
        means.append(np.mean(vals))
    return means


# ---------- B2 option-expiration week ----------
@lru_cache(maxsize=None)
def oe_events(sym: str) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    d = tradfi(sym); out = []; idx = d.index
    for (y, m), _ in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        days = pd.date_range(f"{y}-{m:02d}-15", f"{y}-{m:02d}-21")
        fri = [x for x in days if x.dayofweek == 4][0]
        exp = idx[idx <= fri]
        if not len(exp):
            continue
        exp = exp[-1]; monday = fri - pd.Timedelta(days=4)
        before = idx[idx < monday]
        if len(before) and exp > before[-1]:
            out.append((before[-1], exp))
    return tuple(out)


@lru_cache(maxsize=None)
def week_pool(sym: str) -> dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Friday-to-Friday weeks (Thursday where Friday is a holiday) that do not end on an expiry day, by year."""
    d = tradfi(sym); ends = {e for _, e in oe_events(sym)}; idx = set(d.index)
    fr = [t for t in d.index if t.dayofweek == 4 or (t.dayofweek == 3 and (t + pd.Timedelta(days=1)) not in idx)]
    pool: dict[int, list] = {}
    for a, b in zip(fr[:-1], fr[1:]):
        if b not in ends and (b - a).days <= 8:
            pool.setdefault(b.year, []).append((a, b))
    return pool


def b2(sym: str, mult: float = 1.0, placebo: bool = False) -> pd.Series:
    d = tradfi(sym); ev = oe_events(sym); out = {}; wp = week_pool(sym) if placebo else {}
    for s0, e0 in ev:
        if placebo:
            pool = wp.get(e0.year, [])
            if not pool:
                continue
            s0, e1 = pool[rng.integers(len(pool))]
        else:
            e1 = e0
        x = d.c.loc[e1] / d.c.loc[s0] - 1; days = (e1 - s0).days
        out[e0] = x - (2 * SIDE_TF + F_TF * days) * mult
    return pd.Series(out)


# ---------- B3 MAX-10 swing ----------
def b3(sym: str, placebo: bool = False, mult: float = 1.0) -> pd.Series:
    d = crypto(sym); c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; out = {}; k = 20; n = len(d)
    while k < n - 1:
        if not (placebo or sig[k]) or not np.isfinite(a[k]):
            k += 1; continue
        e, st = c[k], c[k] - a[k]; ex = None; j = k + 1
        while j < n:
            if o[j] <= st:
                ex = o[j]; break
            if lo[j] <= st:
                ex = st; break
            if c[j] < min(c[j - 1], c[j - 2], c[j - 3]):
                ex = c[j]; break
            if j >= k + 10:
                ex = c[j]; break
            j += 1
        if ex is None:
            break
        days = j - k
        out[d.index[k]] = (ex - e) / a[k] - (2 * SIDE_CR + F_CR * days) * e / a[k] * mult
        k = k + 1 if placebo else j + 1
    return pd.Series(out)


# ---------- B4 RSI-2 dips, next-open entry ----------
def rsi(c: pd.Series, n: int) -> pd.Series:
    x = c.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def b4(sym: str, placebo: bool = False, mult: float = 1.0, close_entry: bool = False) -> pd.Series:
    d = tradfi(sym); c = d.c; s200 = c.rolling(200).mean().values; s5 = c.rolling(5).mean().values
    r2 = rsi(c, 2).values; a = E.atr(d).values; C, O, L = c.values, d.o.values, d.l.values; out = {}; i = 201; n = len(d)
    while i < n - 2:
        cond = C[i] > s200[i] and (placebo or r2[i] < 10)
        if not cond:
            i += 1; continue
        k = i if close_entry else i + 1
        e = C[i] if close_entry else O[k]; st = e - 1.5 * a[i]; risk = e - st; ex = None
        j = k if not close_entry else k + 1
        while j < n:
            if j > k and O[j] <= st:
                ex = O[j]; break
            if L[j] <= st:
                ex = st; break
            if C[j] > s5[j] or j >= i + 10:
                ex = C[j]; break
            j += 1
        if ex is None:
            break
        days = (d.index[j] - d.index[k]).days + 1
        cost = ((0.0002 + SIDE_TF) if close_entry else 2 * SIDE_TF) + F_TF * days
        out[d.index[i]] = (ex - e) / risk - cost * e / risk * mult
        i = i + 1 if placebo else j + 1
    return pd.Series(out)


def main() -> None:
    res = {}
    print("B1 month-end rebalancing front-run (SPY/TLT 60/40 drift, last five sessions)")
    F = b1_frame()
    for tag, spread in (("B1a SPY", False), ("B1b SPY minus TLT", True)):
        x = b1(F, "SPY", spread=spread); r = report(tag, x, DESIGN_END)
        post = x[x.index > "2023-03-17"]; print(f"    after the paper's sample (Mar 2023 on): {fmt(post)}")
        q = x[x.index.month.isin([3, 6, 9, 12])]; print(f"    diagnostic, quarter ends only: {fmt(q)}")
        pl = b1_placebo(F, spread); p95 = np.percentile(pl, 95); pct = (np.array(pl) < x.mean()).mean() * 100
        print(f"    placebo: real mean {x.mean() * 100:+.3f}% vs 95th percentile {p95 * 100:+.3f}% (above {pct:.0f}% of draws)")
        fq, fi = b1(F, "QQQ", spread=spread), b1(F, "IWM", spread=spread)
        print(f"    fresh: QQQ {fmt(fq)}; IWM {fmt(fi)}")
        dbl = b1(F, "SPY", 2.0, spread); dbl = dbl[dbl.index <= DESIGN_END]; print(f"    double costs: {fmt(dbl)} (design)")
        res[tag] = verdict(tag, r, min(fq.mean(), fi.mean()), x.mean() > p95, tstat(dbl))

    print("\nB2 option-expiration week, long")
    x = pd.concat([b2("SPY"), b2("QQQ")]); r = report("SPY and QQQ", x, DESIGN_END)
    pl = [pd.concat([b2("SPY", placebo=True), b2("QQQ", placebo=True)]).mean() for _ in range(N_PLACEBO)]
    p95 = np.percentile(pl, 95); print(f"    placebo: real mean {x.mean() * 100:+.3f}% vs 95th percentile of random weeks {p95 * 100:+.3f}%")
    fr = pd.concat([b2("IWM"), b2("DIA")]); print(f"    fresh IWM and DIA: {fmt(fr)}")
    dbl = pd.concat([b2("SPY", 2.0), b2("QQQ", 2.0)]); dbl = dbl[dbl.index <= DESIGN_END]; print(f"    double costs: {fmt(dbl)} (design)")
    res["B2"] = verdict("B2", r, fr.mean(), x.mean() > p95, tstat(dbl))

    print("\nB3 MAX-10 as a swing hold (3-day closing-low exit, 10 days max)")
    core, fresh = ("BTC", "ETH", "SOL", "XRP"), ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
    x = pd.concat([b3(s) for s in core]); r = report("core coins", x, CRYPTO_DESIGN_END, fmtR)
    pz = pd.concat([b3(s, placebo=True) for s in core])
    fr = pd.concat([b3(s) for s in fresh]); frp = pd.concat([b3(s, placebo=True) for s in fresh])
    des, _, _, held = halves(x.sort_index(), CRYPTO_DESIGN_END); pdz, _, _, pheld = halves(pz.sort_index(), CRYPTO_DESIGN_END)
    print(f"    placebo (long every day, same exits): design {fmtR(pdz)}; held-out {fmtR(pheld)}; fresh {fmtR(frp)}")
    print(f"    signal minus placebo: design {des.mean() - pdz.mean():+.3f}R; held-out {held.mean() - pheld.mean():+.3f}R; fresh {fr.mean() - frp.mean():+.3f}R")
    print(f"    fresh coins: {fmtR(fr)}")
    m1 = pd.concat([E_max10(s) for s in core]); print(f"    diagnostic, MAX-10 one-day hold on the same coins: {fmtR(m1)}")
    dbl = pd.concat([b3(s, mult=2.0) for s in core]); print(f"    double costs: {fmtR(dbl[dbl.index <= CRYPTO_DESIGN_END])} (design)")
    ok_p = des.mean() > pdz.mean() and held.mean() > pheld.mean() and fr.mean() > frp.mean()
    res["B3"] = verdict("B3", r, fr.mean(), ok_p, tstat(dbl[dbl.index <= CRYPTO_DESIGN_END]))

    print("\nB4 RSI-2 index dips, next-open entry")
    ds, fs = ("SPY", "QQQ"), ("IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "EWJ", "EWZ", "EWY", "EWT", "GLD", "TLT")
    base = pd.concat([b4(s, close_entry=True) for s in ds]); print(f"    reference, SC6 close entry at Bitunix funding: {fmtR(base)}")
    x = pd.concat([b4(s) for s in ds]); r = report("SPY and QQQ", x, DESIGN_END, fmtR)
    pz = pd.concat([b4(s, placebo=True) for s in ds]); fr = pd.concat([b4(s) for s in fs]); frp = pd.concat([b4(s, placebo=True) for s in fs])
    des, _, _, held = halves(x.sort_index(), DESIGN_END); pdz, _, _, pheld = halves(pz.sort_index(), DESIGN_END)
    print(f"    placebo (every uptrend day, same entry and exits): design {fmtR(pdz)}; held-out {fmtR(pheld)}; fresh {fmtR(frp)}")
    print(f"    signal minus placebo: design {des.mean() - pdz.mean():+.3f}R; held-out {held.mean() - pheld.mean():+.3f}R; fresh {fr.mean() - frp.mean():+.3f}R")
    print(f"    fresh ETFs: {fmtR(fr)}")
    dbl = pd.concat([b4(s, mult=2.0) for s in ds]); print(f"    double costs: {fmtR(dbl[dbl.index <= DESIGN_END])} (design)")
    ok_p = des.mean() > pdz.mean() and held.mean() > pheld.mean() and fr.mean() > frp.mean()
    res["B4"] = verdict("B4", r, fr.mean(), ok_p, tstat(dbl[dbl.index <= DESIGN_END]))
    print("\nPASSED: " + (", ".join(k for k, v in res.items() if v) or "none"))


def E_max10(sym: str) -> pd.Series:
    d = crypto(sym); c, lo = d.c, d.l; a = E.atr(d); sig = c >= c.rolling(10).max().shift(1); out = {}
    for k in range(20, len(d) - 1):
        if sig.iat[k]:
            e, st = c.iat[k], c.iat[k] - a.iat[k]; ex = st if lo.iat[k + 1] <= st else c.iat[k + 1]
            out[d.index[k]] = (ex - e) / a.iat[k] - (2 * SIDE_CR + F_CR) * e / a.iat[k]
    return pd.Series(out)


if __name__ == "__main__":
    main()
