"""Funding batch, pre-registered 23 Oct 2026 before the data arrived (Binance USD-M archives via the feed,
research/perps, 2020 on).

F1 Cross-sectional funding carry (Schmeling, Schrimpf and Todorov, 2023; Koijen et al., 2018). Counterparty:
   leveraged longs who pay to hold crowded coins. Each Monday (decided on Sunday's close, held from the next close),
   rank perps by mean funding over the prior seven days; long the lowest third, short the highest third, equal
   risk (10% / 60-day vol per leg, split across names); at least nine eligible (60 days of funding and prices).
   P&L: perp close-to-close returns, plus funding (longs pay the posted rate on notional at each settlement,
   shorts receive it; signs flip when the rate is negative), less 0.08% a side on turnover.
   Universe split by alphabetical position: A (even) is design 2020 to 2023 and held out 2024 on; B (odd) is fresh.
   Placebo: 200 draws of random ranks. Components (price, funding) reported, not judged.
F2 SW10 skips a signal when the coin's seven-day mean funding is above the 80th percentile of its prior year
   (120 days needed). Must beat unfiltered SW10 in design (2020 to 2023), held-out and fresh, as SW10's sets.
F3 Diagnostic: SW10 charged the funding actually posted during each hold instead of a flat 0.03% a day.
F4 Diagnostic: pooled rank correlation of seven-day mean funding with the next seven days' perp return.
Gate for F1 and F2 as before (design t >= 3, halves, held-out, fresh, placebo or base, double costs t >= 2).

Usage: python swing_funding.py
"""
from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import swing_batch2 as B  # noqa: E402

CORE, FRESH = ("BTC", "ETH", "SOL", "XRP"), ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
DESIGN_END = "2023-12-31"; TV = 0.10; SIDE = 0.0008; N_PLACEBO = 200
rng = np.random.default_rng(20261023)


@lru_cache(maxsize=None)
def perps() -> tuple[str, ...]:
    return tuple(sorted(E.manifest().get("perps", {}).keys()))


@lru_cache(maxsize=None)
def funding(sym: str) -> pd.Series:
    f = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/perps/{sym}_funding.csv.gz")), compression="gzip", index_col=0, parse_dates=True)
    return f.rate.astype(float).sort_index()


@lru_cache(maxsize=None)
def bars(sym: str) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/perps/{sym}_1d.csv.gz")), compression="gzip", index_col=0, parse_dates=True)
    d = d[["o", "h", "l", "c", "v"]].astype(float); d.index = d.index.normalize()
    return d[~d.index.duplicated()].sort_index().loc[:B.END]


def panel(syms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _panel(tuple(syms))


@lru_cache(maxsize=None)
def _panel(syms: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily close returns, 60-day vol, daily funding sum and 7-day mean funding on a common calendar."""
    C = pd.concat({s: bars(s).c for s in syms}, axis=1); idx = pd.date_range(C.index.min(), C.index.max(), freq="D")
    C = C.reindex(idx); R = C.pct_change()
    V = R.rolling(60, min_periods=40).std() * np.sqrt(365)
    Fd = pd.concat({s: funding(s).groupby(funding(s).index.normalize()).sum() for s in syms}, axis=1).reindex(idx)
    F7 = pd.concat({s: funding(s).rolling("7D").mean().groupby(funding(s).index.normalize()).last() for s in syms}, axis=1).reindex(idx)
    age = pd.concat({s: pd.Series(1.0, index=funding(s).index.normalize().unique()) for s in syms}, axis=1).reindex(idx).notna().cumsum()
    F7 = F7.where((age >= 60) & C.notna() & V.notna())
    return R, V, Fd, F7


def carry(syms: list[str], shuffle: bool = False, mult: float = 1.0, parts: bool = False):
    R, V, Fd, F7 = panel(syms); idx = R.index
    reb = ~pd.Series(idx.to_period("W-SUN"), index=idx).duplicated()
    W = pd.DataFrame(np.nan, index=idx, columns=syms)
    pos = np.flatnonzero(reb.values)
    for i in pos:
        if i < 1:
            continue
        t = idx[i - 1]; sc = F7.loc[t].dropna()
        if len(sc) < 9:
            W.iloc[i] = 0.0; continue
        order = list(sc.sort_values().index)
        if shuffle:
            order = list(rng.permutation(order))
        k = len(order) // 3; row = pd.Series(0.0, index=syms)
        for s in order[:k]:
            row[s] = TV / V.loc[t, s] / (2 * k)
        for s in order[-k:]:
            row[s] = -TV / V.loc[t, s] / (2 * k)
        W.iloc[i] = row.values
    W = W.ffill().fillna(0.0).clip(-3, 3); H = W.shift(1).fillna(0.0)  # held from the close after the decision
    price = (H * R.fillna(0.0)).sum(axis=1); fund = -(H * Fd.fillna(0.0)).sum(axis=1)
    cost = W.diff().abs().sum(axis=1).fillna(0.0) * SIDE * mult
    live = (W.abs().sum(axis=1) > 0); start = live.idxmax() if live.any() else idx[-1]
    net = (price + fund - cost)[idx >= start]
    return (net, price[idx >= start], fund[idx >= start]) if parts else net


def sr(x: pd.Series) -> str:
    x = x.dropna()
    return f"SR {x.mean() / x.std() * np.sqrt(365):+.2f} t {B.tstat(x):+.2f} ({len(x) / 365.25:.1f}y)" if len(x) > 60 else "n/a"


def sw_funding(sym: str, mask_hot: bool = False, actual: bool = False, mult: float = 1.0) -> pd.Series:
    """SW10 on the coin's spot daily bars (as live), from 2020; funding filter and actual funding optional."""
    d = B.crypto(sym); d = d[d.index >= "2019-10-01"]; c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; f = funding(sym)
    f7 = f.rolling("7D").mean(); thr = f7.rolling("365D", min_periods=360).quantile(0.8)
    f7d = f7.groupby(f7.index.normalize()).last().reindex(d.index).ffill(); thd = thr.groupby(thr.index.normalize()).last().reindex(d.index).ffill()
    fday = f.groupby(f.index.normalize()).sum().reindex(d.index).fillna(0.0).values
    first = f.index[0].normalize() + pd.Timedelta(days=120)
    out = {}; k = 20; n = len(d)
    while k < n - 1:
        if not sig[k] or not np.isfinite(a[k]) or d.index[k] < first:
            k += 1; continue
        if mask_hot and (not np.isfinite(thd.iat[k]) or f7d.iat[k] > thd.iat[k]):
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
        fcost = fday[k + 1:j + 1].sum() if actual else B.F_CR * (j - k)
        out[d.index[k]] = (ex - e) / a[k] - (2 * B.SIDE_CR + fcost) * e / a[k] * mult
        k = j + 1
    return pd.Series(out, dtype=float)


def main() -> None:
    syms = list(perps()); print(f"perps in the feed: {len(syms)}: {', '.join(syms)}")
    A, Bs = syms[0::2], syms[1::2]
    print("\nF1 cross-sectional funding carry, weekly")
    x, p, f = carry(A, parts=True); des = x[x.index <= DESIGN_END]; held = x[x.index > DESIGN_END]
    mid = des.index[0] + (des.index[-1] - des.index[0]) / 2
    print(f"  universe A ({len(A)}): design {sr(des)} [halves {sr(des[des.index < mid])} / {sr(des[des.index >= mid])}]; held-out {sr(held)}")
    print(f"    components, full: price {sr(p)}; funding {sr(f)}")
    xb = carry(Bs); print(f"  fresh universe B ({len(Bs)}): {sr(xb)}")
    pl = [carry(A, shuffle=True).mean() for _ in range(N_PLACEBO)]; p95 = np.percentile(pl, 95)
    print(f"  placebo: real mean {x.mean() * 1e4:+.2f}bp a day vs 95th percentile of random ranks {p95 * 1e4:+.2f}bp")
    dbl = carry(A, mult=2.0); dd = dbl[dbl.index <= DESIGN_END]; print(f"  double costs (design): {sr(dd)}")
    r = dict(des=B.tstat(des), h1=B.tstat(des[des.index < mid]), h2=B.tstat(des[des.index >= mid]), held=held.mean())
    ok1 = B.verdict("F1", r, xb.mean(), x.mean() > p95, B.tstat(dd))

    print("\nF2 SW10 skipping crowded-funding signals")
    def pool(fn, coins):
        return pd.concat([fn(s) for s in coins if s in syms]).sort_index()
    base_c, filt_c = pool(sw_funding, CORE), pool(lambda s: sw_funding(s, True), CORE)
    base_f, filt_f = pool(sw_funding, FRESH), pool(lambda s: sw_funding(s, True), FRESH)
    rf = B.report("F2 filtered, core", filt_c, DESIGN_END, B.fmtR); B.report("SW10 unfiltered, core", base_c, DESIGN_END, B.fmtR)
    print(f"    fresh: filtered {B.fmtR(filt_f)}; unfiltered {B.fmtR(base_f)}")
    bd, bh = base_c[base_c.index <= DESIGN_END], base_c[base_c.index > DESIGN_END]; fd, fh = filt_c[filt_c.index <= DESIGN_END], filt_c[filt_c.index > DESIGN_END]
    dbl = pool(lambda s: sw_funding(s, True, mult=2.0), CORE); dbl = dbl[dbl.index <= DESIGN_END]
    ok2 = B.verdict("F2", rf, filt_f.mean(), fd.mean() > bd.mean() and fh.mean() > bh.mean() and filt_f.mean() > base_f.mean(), B.tstat(dbl))

    print("\nF3 SW10 charged actual funding (2020 on)")
    for name, coins in (("core", CORE), ("fresh", FRESH)):
        print(f"  {name}: flat 0.03% a day {B.fmtR(pool(sw_funding, coins))}; actual funding {B.fmtR(pool(lambda s: sw_funding(s, actual=True), coins))}")

    print("\nF4 seven-day mean funding against the next seven days' return (pooled, weekly, rank correlation)")
    R, V, Fd, F7 = panel(syms); C = (1 + R.fillna(0)).cumprod()
    fwd = C.shift(-7) / C - 1; sun = F7.index[F7.index.dayofweek == 6]
    ic = [F7.loc[t].corr(fwd.loc[t], method="spearman") for t in sun if F7.loc[t].notna().sum() >= 9]
    ic = pd.Series(ic, dtype=float).dropna(); print(f"  mean rank correlation {ic.mean():+.3f} (t {B.tstat(ic):+.2f}, {len(ic)} weeks)")
    print("\nPASSED: " + (", ".join(k for k, v in (("F1", ok1), ("F2", ok2)) if v) or "none"))


if __name__ == "__main__":
    main()
