"""Swing batch 4, pre-registered 23 Oct 2026 before running, after SW10 (MAX-10 held as a swing) was promoted.

Two structural ideas outside crypto momentum, then three ways to get more from SW10.

D1 Short both legs of a leveraged ETF pair (TQQQ and SQQQ), equal notional, rebalanced weekly. Counterparty:
   holders of daily-reset leveraged ETFs, who bear the fund's fees and financing; on perps the short pays no borrow.
   Design 2010 to 2016, held out 2017 on; fresh SOXL and SOXS.
D2 Short a leveraged ETF, long L times its underlying, rebalanced weekly: harvests the fund's fees and financing.
   Design TQQQ against 3 x QQQ; fresh SOXL/3 x SMH, TMF/3 x TLT, KORU/3 x EWY, NVDL/2 x NVDA, TSLL/2 x TSLA.
   D1 and D2 report the gross result too: on perps the trade is worth whatever the ETFs' internal costs exceed the
   perp funding by, and the funding rate assumed here (0.0105% a day on every leg, shorts paying) is a guess.
D3 SW10 only when the signal day's volume is above its 20-day mean (attention). A filter: it must beat unfiltered
   SW10 in design, held-out and fresh, and beat its placebo.
D4 Short mirror of SW10: short at a close at or below the prior ten closes, stop 1 ATR above, cover at a close above
   the highest of the prior three closes or after ten closes. D4a always; D4b only when BTC is in a daily downtrend
   (close below a falling EMA50). Shorts earn no funding here. Placebo: a short every day with the same exits.
D5 Pyramid: a MAX-10 signal while an SW10 position on the same coin is open adds a unit traded by the same rule.
   Judged on the add-on units alone against placebo add-ons (a unit added on every open day).
Crypto sets as SW10: core BTC, ETH, SOL, XRP (design to 2023, held out 2024 on); fresh LINK, BNB, ADA, DOGE, AVAX, DOT.
Costs: tradfi 0.09% a side on turnover, funding 0.0105% a calendar day on every leg; crypto 0.08% a side, longs pay
0.03% a day. Gate: design t >= 3, both halves positive, held-out and fresh positive, placebo or base beaten, t >= 2
at double costs (design).

Usage: python swing_batch4.py
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

CORE, FRESH = ("BTC", "ETH", "SOL", "XRP"), ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")


# ---------- D1 and D2: leveraged ETF structures, weekly ----------
def weekly_book(legs: list[tuple[str, float]], mult: float = 1.0, gross_only: bool = False) -> pd.Series:
    """Daily return of a book of signed notional weights, reset weekly (first session of each week), drifting between."""
    cl = pd.concat({s: B.tradfi(s).c for s, _ in legs}, axis=1).dropna()
    r = cl.pct_change().fillna(0.0); idx = cl.index
    reset = ~pd.Series(idx.to_period("W-SUN"), index=idx).duplicated()
    w0 = np.array([w for _, w in legs]); w = w0.copy(); out = []; prev_t = idx[0]
    for t in idx:
        day = float((w * r.loc[t].values).sum())
        w = w * (1 + r.loc[t].values)  # notionals drift with prices
        cost = 0.0
        if reset.loc[t]:
            cost = np.abs(w0 - w).sum() * B.SIDE_TF; w = w0.copy()
        fund = np.abs(w).sum() * B.F_TF * (t - prev_t).days
        out.append(day if gross_only else day - (cost + fund) * mult); prev_t = t
    s = pd.Series(out, index=idx); return s.iloc[1:]


def year_stats(x: pd.Series) -> str:
    sr = x.mean() / x.std() * np.sqrt(252); return f"annual {x.mean() * 252 * 100:+.1f}%, SR {sr:+.2f}, t {B.tstat(x):+.2f}"


def judge_daily(name: str, x: pd.Series, fresh: list[pd.Series], dbl: pd.Series, end: str = B.DESIGN_END) -> bool:
    des, h1, h2, held = B.halves(x, end)
    print(f"  {name}: design {year_stats(des)} [halves {year_stats(h1)} / {year_stats(h2)}]; held-out {year_stats(held)}")
    fm = [f.mean() for f in fresh]
    print(f"    fresh: " + "; ".join(f"{f.name} {year_stats(f)}" for f in fresh))
    d2 = dbl[dbl.index <= end]; print(f"    double costs (design): {year_stats(d2)}")
    r = dict(des=B.tstat(des), h1=B.tstat(h1), h2=B.tstat(h2), held=held.mean())
    return B.verdict(name, r, min(fm) if fm else 1.0, True, B.tstat(d2))


# ---------- crypto ----------
def crypto_v(sym: str) -> pd.DataFrame:
    d = E.completed(E.static_klines(sym, "1d", "2016-01-01"), "1d")[["o", "h", "l", "c", "v"]].astype(float)
    d.index = d.index.normalize(); return d.loc[:B.END]


def btc_down() -> pd.Series:
    c = crypto_v("BTC").c; e = c.ewm(span=50, adjust=False).mean()
    return (c < e) & (e < e.shift(5))


def sw(d: pd.DataFrame, side: int = 1, placebo: bool = False, mask: np.ndarray | None = None, mult: float = 1.0,
       overlap: bool = False, only_inside: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """SW10 (side 1) or its short mirror (side -1). mask: allowed signal days. overlap: take every qualifying day."""
    c, lo, hi, o = d.c.values, d.l.values, d.h.values, d.o.values; a = E.atr(d).values; n = len(d)
    ext = d.c.rolling(10).max().shift(1).values if side == 1 else d.c.rolling(10).min().shift(1).values
    sig = (c >= ext) if side == 1 else (c <= ext)
    inside = np.zeros(n, bool)
    if only_inside is not None:
        for k0, j0 in only_inside:
            inside[k0 + 1:j0] = True
    rows = []; k = 20
    while k < n - 1:
        ok = (placebo or sig[k]) and np.isfinite(a[k]) and (mask is None or mask[k]) and (only_inside is None or inside[k])
        if not ok:
            k += 1; continue
        e, st = c[k], c[k] - side * a[k]; ex = None; j = k + 1
        while j < n:
            if side == 1 and (o[j] <= st or lo[j] <= st):
                ex = min(st, o[j]); break
            if side == -1 and (o[j] >= st or hi[j] >= st):
                ex = max(st, o[j]); break
            if (side == 1 and c[j] < c[j - 3:j].min()) or (side == -1 and c[j] > c[j - 3:j].max()) or j >= k + 10:
                ex = c[j]; break
            j += 1
        if ex is None:
            break
        fund = B.F_CR * (j - k) if side == 1 else 0.0
        R = side * (ex - e) / a[k] - (2 * B.SIDE_CR + fund) * e / a[k] * mult
        rows.append(dict(t=d.index[k], k=k, j=j, R=R))
        k = k + 1 if (placebo or overlap) else j + 1
    return pd.DataFrame(rows)


def sets(fn) -> dict[str, pd.Series]:
    out = {}
    for s in CORE + FRESH:
        T = fn(s)
        out[s] = pd.Series(T.R.values, index=pd.DatetimeIndex(T.t)) if len(T) else pd.Series(dtype=float)
    return out


def split3(m: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series, pd.Series]:
    core = pd.concat([m[s] for s in CORE]).sort_index(); fresh = pd.concat([m[s] for s in FRESH])
    return core[core.index <= B.CRYPTO_DESIGN_END], core[core.index > B.CRYPTO_DESIGN_END], fresh


def judge_trades(name: str, sig: dict, ref: dict, dbl: dict, ref_name: str) -> bool:
    x = pd.concat([sig[s] for s in CORE]).sort_index(); r = B.report(name, x, B.CRYPTO_DESIGN_END, B.fmtR)
    sd, sh, sf = split3(sig); rd, rh, rf = split3(ref); dd, _, _ = split3(dbl)
    print(f"    {ref_name}: design {B.fmtR(rd)}; held-out {B.fmtR(rh)}; fresh {B.fmtR(rf)}")
    print(f"    fresh: {B.fmtR(sf)}; minus {ref_name}: design {sd.mean() - rd.mean():+.3f}R, held-out {sh.mean() - rh.mean():+.3f}R, fresh {sf.mean() - rf.mean():+.3f}R")
    print(f"    double costs (design): {B.fmtR(dd)}")
    ok_ref = sd.mean() > rd.mean() and sh.mean() > rh.mean() and sf.mean() > rf.mean()
    return B.verdict(name, r, sf.mean(), ok_ref, B.tstat(dd))


def main() -> None:
    res = {}
    print("D1 short both TQQQ and SQQQ, equal notional, weekly")
    pair = [("TQQQ", -0.5), ("SQQQ", -0.5)]; fpair = [("SOXL", -0.5), ("SOXS", -0.5)]
    x = weekly_book(pair); g = weekly_book(pair, gross_only=True)
    print(f"    gross of perp costs: full {year_stats(g)}; held-out {year_stats(g[g.index > B.DESIGN_END])}")
    f = weekly_book(fpair); f.name = "SOXL and SOXS"
    res["D1"] = judge_daily("D1", x, [f], weekly_book(pair, 2.0))

    print("\nD2 short a leveraged ETF, long L x its underlying, weekly")
    x = weekly_book([("TQQQ", -1.0), ("QQQ", 3.0)]); g = weekly_book([("TQQQ", -1.0), ("QQQ", 3.0)], gross_only=True)
    print(f"    gross of perp costs: full {year_stats(g)}; held-out {year_stats(g[g.index > B.DESIGN_END])}")
    fresh = []
    for etf, und, L in (("SOXL", "SMH", 3.0), ("TMF", "TLT", 3.0), ("KORU", "EWY", 3.0), ("NVDL", "NVDA", 2.0), ("TSLL", "TSLA", 2.0)):
        s = weekly_book([(etf, -1.0), (und, L)]); s.name = f"{etf}/{und}"; fresh.append(s)
        gs = weekly_book([(etf, -1.0), (und, L)], gross_only=True); print(f"      {etf}/{und} gross {year_stats(gs)}")
    res["D2"] = judge_daily("D2", x, fresh, weekly_book([("TQQQ", -1.0), ("QQQ", 3.0)], 2.0))

    data = {s: crypto_v(s) for s in CORE + FRESH}
    base = sets(lambda s: sw(data[s]))
    print("\nD3 SW10 with signal-day volume above its 20-day mean")
    vm = {s: (data[s].v > data[s].v.rolling(20).mean().shift(1)).values for s in data}
    sig = sets(lambda s: sw(data[s], mask=vm[s])); dbl = sets(lambda s: sw(data[s], mask=vm[s], mult=2.0))
    res["D3 vs SW10"] = judge_trades("D3 volume filter", sig, base, dbl, "unfiltered SW10")
    plc = sets(lambda s: sw(data[s], placebo=True, mask=vm[s]))
    sd, sh, sf = split3(sig); pd_, ph, pf = split3(plc)
    print(f"    placebo on high-volume days: signal minus placebo design {sd.mean() - pd_.mean():+.3f}R, held-out {sh.mean() - ph.mean():+.3f}R, fresh {sf.mean() - pf.mean():+.3f}R")

    print("\nD4 short mirror of SW10")
    down = btc_down()
    for tag, use_mask in (("D4a always", False), ("D4b BTC downtrend only", True)):
        mk = {s: down.reindex(data[s].index).fillna(False).values for s in data} if use_mask else {s: None for s in data}
        sig = sets(lambda s: sw(data[s], -1, mask=mk[s])); plc = sets(lambda s: sw(data[s], -1, placebo=True, mask=mk[s]))
        dbl = sets(lambda s: sw(data[s], -1, mask=mk[s], mult=2.0))
        res[tag] = judge_trades(tag, sig, plc, dbl, "placebo")

    print("\nD5 pyramid add-ons inside open SW10 positions")
    spans = {s: [(int(r.k), int(r.j)) for r in sw(data[s]).itertuples()] for s in data}
    sig = sets(lambda s: sw(data[s], overlap=True, only_inside=spans[s])); plc = sets(lambda s: sw(data[s], placebo=True, only_inside=spans[s]))
    dbl = sets(lambda s: sw(data[s], overlap=True, only_inside=spans[s], mult=2.0))
    res["D5"] = judge_trades("D5 add-ons", sig, plc, dbl, "placebo add-ons")
    print("\nPASSED: " + (", ".join(k for k, v in res.items() if v) or "none"))


if __name__ == "__main__":
    main()
