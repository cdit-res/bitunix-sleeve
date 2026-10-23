"""Swing and position edges, pre-registered 23 Oct 2026 before running. Owner: "not good edge, find better swing edges".

Every hypothesis names its counterparty. Settings below are fixed; anything changed after the first run is a new
counted variant in LEDGER.md.

H1 Diversified time-series momentum, long and short (Moskowitz, Ooi and Pedersen, 2012; Hurst, Ooi and Pedersen, 2017).
   Counterparty: hedgers paying to lay off risk, and slow diffusion of information across markets.
   Markets: gold, silver, platinum, palladium, WTI, Brent, copper, natgas (Yahoo continuous futures), SPY, TLT, BTC.
   Signal: mean of sign(1, 3 and 12-month return) (21, 63, 252 trading days; 30, 91, 365 calendar days for crypto).
   Weight: signal x 10% / 60-day vol / markets live, capped at 3x per market. Weekly rebalance.
   Fresh markets: QQQ, IWM, EWJ, EWT, EWY, EWZ, ETH, XRP. Diagnostics by asset class are not judged.
H2 Commodity cross-sectional momentum (Miffre and Rallis, 2007; Asness, Moskowitz and Pedersen, 2013).
   Counterparty: as H1, across commodities. Rank the eight on 12-1 month return; long top two, short bottom two,
   equal risk; monthly rebalance; needs six ranked. No fresh set exists.
H3 Crypto cross-sectional momentum (Liu, Tsyvinski and Wu, 2022). Counterparty: attention-driven underreaction.
   Core: the ten static-klines majors. Rank on 28-day return; long top third, short bottom third (at least two each,
   six eligible, 120 days of history), equal risk; weekly rebalance. Fresh: NEAR, INJ, LTC, TAO, FET, RUNE, SUI, HYPE.
H4 Cross-asset relative momentum, long only (Faber, 2010; Antonacci, 2014). Counterparty: slow reallocation by
   benchmarked investors. Universe (Bitunix-tradable or proxied): SPY, QQQ, IWM, SMH, XLE, XBI, GDX, EWJ, EWT, EWY,
   EWZ, GLD, SLV, USO, TLT. Hold the top three by 6-month return among those with a positive 6-month return,
   equal risk; monthly rebalance. No fresh set.
H5 Volatility-managed index exposure (Moreira and Muir, 2017). Counterparty: investors who do not cut exposure when
   volatility rises. w = expanding median of 21-day variance / current 21-day variance, capped at 3; monthly
   rebalance. Judged on alpha against constant exposure (Newey-West t, monthly returns). Design SPY and QQQ;
   fresh IWM, SMH, EWJ, EWT, EWY, EWZ and BTC.

Common: signals use closes through t-1 and trade at close t (one-day lag); costs 0.08% per unit of turnover
(taker 0.06% plus 0.02% slippage); funding 0.0105% a calendar day on gross tradfi notional, both sides; crypto longs
pay 0.03% a day, shorts receive nothing. Futures continuous series: daily return on the absolute prior price,
clipped to 30% (roll and negative-price guard). Design to 2016 (crypto to 2023), held out after, run once.
Gate: full-sample net t >= 3; the same sign in both design halves; held-out above zero; fresh above zero where a
fresh set exists; above the 95th percentile of 200 placebo draws (random signs, ranks or picks; H5: alpha over
constant exposure); t >= 2 at double costs.

Usage: python swing_search.py
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
TV, CAP = 0.10, 3.0
C_TRADE, F_TRADFI, F_CRYPTO = 0.0008, 0.000105, 0.0003
DESIGN_END, CRYPTO_DESIGN_END = "2016-12-31", "2023-12-31"
N_PLACEBO = 200
COMMOD = ("GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "NG=F")
H1_DESIGN = COMMOD + ("SPY", "TLT", "BTC")
H1_FRESH = ("QQQ", "IWM", "EWJ", "EWT", "EWY", "EWZ", "ETH", "XRP")
CORE = ("BTC", "ETH", "BNB", "XRP", "ADA", "LINK", "DOGE", "SOL", "DOT", "AVAX")
FRESH = ("NEAR", "INJ", "LTC", "TAO", "FET", "RUNE", "SUI", "HYPE")
H4_U = ("SPY", "QQQ", "IWM", "SMH", "XLE", "XBI", "GDX", "EWJ", "EWT", "EWY", "EWZ", "GLD", "SLV", "USO", "TLT")
H5_DESIGN, H5_FRESH = ("SPY", "QQQ"), ("IWM", "SMH", "EWJ", "EWT", "EWY", "EWZ")
CRYPTO = set(CORE) | set(FRESH)
rng = np.random.default_rng(20261023)


# ---------- data ----------
@lru_cache(maxsize=None)
def closes(sym: str) -> pd.Series:
    """Completed daily closes on the market's own trading days."""
    if sym in CRYPTO:
        d = E.static_klines(sym, "1d", "2016-01-01") if sym in CORE else E.bars(sym, "1d")
        c = E.completed(d, "1d").c
    else:
        d = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/daily/{E.safe(sym)}.csv.gz")), compression="gzip",
                        index_col=0, parse_dates=True)
        c = d["c"]
    c = c.astype(float).dropna(); c = c[~c.index.duplicated()].sort_index()
    c.index = c.index.normalize()
    return c[c.index <= END]


def rets(sym: str, c: pd.Series) -> pd.Series:
    r = c.diff() / c.shift().abs()
    return r.clip(-0.30, 0.30) if sym.endswith("=F") else r


def vol(sym: str, r: pd.Series) -> pd.Series:
    return r.rolling(60, min_periods=40).std() * np.sqrt(365 if sym in CRYPTO else 252)


class Panel:
    """Returns, vols and closes on a common calendar (zeros and carried values where a market is shut)."""

    def __init__(self, syms: tuple[str, ...], calendar: str):
        self.syms = syms; self.c, self.r, self.v = {}, {}, {}
        for s in syms:
            c = closes(s); self.c[s] = c; self.r[s] = rets(s, c); self.v[s] = vol(s, self.r[s])
        idx = sorted(set().union(*[set(c.index) for c in self.c.values()]))
        idx = pd.DatetimeIndex(idx)
        if calendar == "calendar":
            idx = pd.date_range(idx[0], idx[-1], freq="D")
        self.idx = idx
        self.R = pd.DataFrame({s: self.r[s].reindex(idx).fillna(0.0) for s in syms})
        self.V = pd.DataFrame({s: self.v[s].reindex(idx).ffill() for s in syms})
        self.days = pd.Series(np.r_[1, np.diff(idx.values).astype("timedelta64[D]").astype(int)], index=idx)

    def on(self, s: str, x: pd.Series) -> pd.Series:
        """A series computed on a market's own trading days, carried onto the panel calendar."""
        return x.reindex(self.idx).ffill()


def rebal_days(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    key = idx.to_period("W-SUN") if freq == "W" else idx.to_period("M")
    first = ~pd.Series(key, index=idx).duplicated()
    return first


def run(P: Panel, T: pd.DataFrame, freq: str, mult: float = 1.0) -> pd.Series:
    """Net daily return. T: target weights from closes through t; held from close t+1 (one-day lag)."""
    T = T.clip(-CAP, CAP).fillna(0.0)
    reb = rebal_days(P.idx, freq)
    W = T.shift(1)[reb.values].reindex(P.idx).ffill().fillna(0.0)
    H = W.shift(1).fillna(0.0)
    gross = (H * P.R).sum(axis=1)
    cost = W.diff().abs().sum(axis=1).fillna(0.0) * C_TRADE * mult
    fund = pd.Series(0.0, index=P.idx)
    for s in P.syms:
        if s in CRYPTO:
            fund += H[s].clip(lower=0) * F_CRYPTO
        else:
            fund += H[s].abs() * F_TRADFI
    fund = fund * P.days * mult
    out = gross - cost - fund
    live = (T.abs().sum(axis=1) > 0).idxmax() if (T.abs().sum(axis=1) > 0).any() else P.idx[-1]
    return out[out.index >= live]


# ---------- statistics ----------
def ann(idx: pd.DatetimeIndex) -> float:
    per_year = len(idx) / max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    return per_year


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(n=len(x), sr=np.nan, t=np.nan)
    sr = x.mean() / x.std() * np.sqrt(ann(x.index))
    return dict(n=len(x), sr=sr, t=x.mean() / x.std() * np.sqrt(len(x)), yrs=(x.index[-1] - x.index[0]).days / 365.25)


def mdd(x: pd.Series) -> float:
    """Deepest fall of cumulative return with the series scaled to 10% annual volatility."""
    x = x.dropna(); s = x * (TV / (x.std() * np.sqrt(ann(x.index))))
    eq = (1 + s).cumprod(); return float((eq / eq.cummax() - 1).min())


def fmt(d: dict) -> str:
    return "n/a" if not np.isfinite(d.get("sr", np.nan)) else f"SR {d['sr']:+.2f} t {d['t']:+.2f} ({d['yrs']:.1f}y)"


def split(x: pd.Series, design_end: str) -> dict:
    des = x[x.index <= design_end]; held = x[x.index > design_end]
    mid = des.index[0] + (des.index[-1] - des.index[0]) / 2 if len(des) else None
    return dict(full=stats(x), design=stats(des), h1=stats(des[des.index < mid]) if mid is not None else {},
                h2=stats(des[des.index >= mid]) if mid is not None else {}, held=stats(held))


def nw_alpha(y: pd.Series, x: pd.Series, lags: int = 3) -> tuple[float, float]:
    """OLS alpha of y on x with a Newey-West t."""
    d = pd.concat([y, x], axis=1).dropna(); Y = d.iloc[:, 0].values; X = np.c_[np.ones(len(d)), d.iloc[:, 1].values]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; n = len(Y)
    XtX = np.linalg.inv(X.T @ X); S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1); G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]); S += w * (G + G.T)
    V = XtX @ S @ XtX
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0]))


# ---------- signals ----------
def tsmom(P: Panel, sign_fixed: float | None = None) -> pd.DataFrame:
    T = pd.DataFrame(0.0, index=P.idx, columns=P.syms)
    for s in P.syms:
        c = P.c[s]; lbs = (30, 91, 365) if s in CRYPTO else (21, 63, 252)
        if s in CRYPTO:
            cc = c.asfreq("D").ffill(); sig = sum(np.sign(cc / cc.shift(L) - 1) for L in lbs) / 3
        else:
            sig = sum(np.sign(c / c.shift(L) - 1) for L in lbs) / 3
        sig = sig.where(c.expanding().count() > max(lbs))
        if sign_fixed is not None:
            sig = sig.notna().astype(float).where(sig.notna()) * sign_fixed
        T[s] = P.on(s, sig) * TV / P.V[s]
    live = T.notna().sum(axis=1).replace(0, np.nan)
    return T.div(live, axis=0)


def random_signs(P: Panel, T: pd.DataFrame, freq: str) -> pd.DataFrame:
    reb = rebal_days(P.idx, freq)
    S = pd.DataFrame(rng.choice([-1.0, 1.0], size=(int(reb.sum()), T.shape[1])), index=T.index[reb.values],
                     columns=T.columns).reindex(T.index).ffill()
    S = S.shift(-1).ffill()  # run() reads the row before each rebalance day
    return T.abs() * S


def ranked(P: Panel, score: pd.DataFrame, k_rule, min_n: int, freq: str, long_only: bool = False,
           shuffle: bool = False) -> pd.DataFrame:
    """Long the top k, short the bottom k (or long-only top k), equal risk. The decision for each rebalance day is
    made on the previous day's close and stored on that row, which is the row run() reads."""
    reb = rebal_days(P.idx, freq); T = pd.DataFrame(np.nan, index=P.idx, columns=P.syms)
    pos = np.flatnonzero(reb.values)
    for t in [P.idx[i - 1] for i in pos if i > 0]:
        sc = score.loc[t].dropna(); sc = sc[P.V.loc[t, sc.index].notna()]
        if long_only:
            sc = sc[sc > 0]
        n = len(sc)
        if n < min_n:
            T.loc[t] = 0.0; continue
        k = k_rule(n)
        order = list(sc.sort_values(ascending=False).index)
        if shuffle:
            order = list(rng.permutation(order))
        row = pd.Series(0.0, index=P.syms)
        legs = [(order[:k], 1.0)] + ([] if long_only else [(order[-k:], -1.0)])
        m = k * len(legs)
        for names, sgn in legs:
            for s in names:
                row[s] = sgn * TV / P.V.loc[t, s] / m
        T.loc[t] = row
    return T


def score_on(P: Panel, f) -> pd.DataFrame:
    return pd.DataFrame({s: P.on(s, f(s, P.c[s])) for s in P.syms})


# ---------- MAX-10 daily series for correlation ----------
def max10_daily() -> pd.Series:
    out = []
    for s in ("BTC", "ETH", "SOL", "XRP"):
        d = E.completed(E.static_klines(s, "1d", "2016-01-01"), "1d"); c, lo = d.c, d.l; a = E.atr(d)
        sig = c >= c.rolling(10).max().shift(1)
        for k in range(20, len(d) - 1):
            if sig.iat[k]:
                e, st = c.iat[k], c.iat[k] - a.iat[k]; ex = st if lo.iat[k + 1] <= st else c.iat[k + 1]
                out.append((d.index[k + 1].normalize(), (ex - e) / a.iat[k] - 0.0008 * e / a.iat[k]))
    return pd.DataFrame(out, columns=["t", "R"]).groupby("t").R.sum()


# ---------- report ----------
def line(name: str, x: pd.Series, design_end: str) -> dict:
    s = split(x, design_end)
    print(f"  {name}: full {fmt(s['full'])}; design {fmt(s['design'])} [halves {fmt(s['h1'])} / {fmt(s['h2'])}]; "
          f"held-out {fmt(s['held'])}; max DD at 10% vol {mdd(x) * 100:.0f}%")
    return s


def placebo_pct(real: pd.Series, draws: list[pd.Series]) -> tuple[float, float]:
    r = stats(real)["sr"]; sr = np.array([stats(d)["sr"] for d in draws])
    return float((sr < r).mean() * 100), float(np.nanpercentile(sr, 95))


def gate(name: str, s: dict, fresh: dict | None, pct: float, dbl: dict, extra: bool = True) -> bool:
    checks = {
        "full t >= 3": s["full"]["t"] >= 3,
        "both design halves positive": s["h1"].get("t", -1) > 0 and s["h2"].get("t", -1) > 0,
        "held-out positive": s["held"].get("t", -1) > 0,
        "fresh positive": True if fresh is None else fresh.get("t", -1) > 0,
        "above placebo 95th percentile": pct >= 95,
        "double costs t >= 2": dbl["t"] >= 2,
    }
    ok = all(checks.values()) and extra
    print(f"  GATE {name}: {'PASS' if ok else 'FAIL'} ({'; '.join(k for k, v in checks.items() if not v) or 'all checks met'})")
    return ok


def main() -> None:
    m10 = max10_daily(); results = {}
    print("H1 diversified time-series momentum, weekly")
    P = Panel(H1_DESIGN, "calendar"); T = tsmom(P); x = run(P, T, "W"); s = line("design markets", x, DESIGN_END)
    lo = run(P, tsmom(P, 1.0), "W"); line("long-only same markets (baseline)", lo, DESIGN_END)
    draws = [run(P, random_signs(P, T, "W"), "W") for _ in range(N_PLACEBO)]; pct, p95 = placebo_pct(x, draws)
    print(f"  placebo: above {pct:.0f}% of random-sign draws (95th percentile SR {p95:+.2f})")
    dbl = stats(run(P, T, "W", 2.0)); print(f"  double costs: {fmt(dbl)}")
    for grp, names in (("commodities", COMMOD), ("SPY", ("SPY",)), ("TLT", ("TLT",)), ("BTC", ("BTC",))):
        Tg = T.copy(); Tg[[c for c in T.columns if c not in names]] = 0.0
        print(f"    diagnostic, {grp} sleeve: {fmt(stats(run(P, Tg, 'W')))}")
    Pf = Panel(H1_FRESH, "calendar"); xf = run(Pf, tsmom(Pf), "W"); fr = stats(xf); print(f"  fresh markets: {fmt(fr)}")
    a, at = nw_alpha(x.resample("ME").sum(), lo.resample("ME").sum()); print(f"  alpha over long-only baseline: {a * 100:+.2f}% a month, t {at:+.2f}")
    results["H1"] = gate("H1", s, fr, pct, dbl, extra=at > 0); results["H1_x"] = x

    print("\nH2 commodity cross-sectional momentum, monthly")
    P = Panel(COMMOD, "business"); sc = score_on(P, lambda s, c: c.shift(21) / c.shift(252) - 1)
    T = ranked(P, sc, lambda n: 2, 6, "M"); x = run(P, T, "M"); s = line("eight commodities", x, DESIGN_END)
    draws = [run(P, ranked(P, sc, lambda n: 2, 6, "M", shuffle=True), "M") for _ in range(N_PLACEBO)]
    pct, p95 = placebo_pct(x, draws); print(f"  placebo: above {pct:.0f}% of random-rank draws (95th percentile SR {p95:+.2f})")
    dbl = stats(run(P, T, "M", 2.0)); print(f"  double costs: {fmt(dbl)}")
    results["H2"] = gate("H2", s, None, pct, dbl); results["H2_x"] = x

    print("\nH3 crypto cross-sectional momentum, weekly")
    k3 = lambda n: max(2, n // 3)
    P = Panel(CORE, "calendar")
    sc = score_on(P, lambda s, c: (c / c.shift(28) - 1).where(c.expanding().count() >= 120))
    T = ranked(P, sc, k3, 6, "W"); x = run(P, T, "W"); s = line("ten core coins", x, CRYPTO_DESIGN_END)
    draws = [run(P, ranked(P, sc, k3, 6, "W", shuffle=True), "W") for _ in range(N_PLACEBO)]
    pct, p95 = placebo_pct(x, draws); print(f"  placebo: above {pct:.0f}% of random-rank draws (95th percentile SR {p95:+.2f})")
    dbl = stats(run(P, T, "W", 2.0)); print(f"  double costs: {fmt(dbl)}")
    Pf = Panel(FRESH, "calendar")
    scf = score_on(Pf, lambda s, c: (c / c.shift(28) - 1).where(c.expanding().count() >= 120))
    fr = stats(run(Pf, ranked(Pf, scf, k3, 6, "W"), "W")); print(f"  fresh coins: {fmt(fr)}")
    results["H3"] = gate("H3", s, fr, pct, dbl); results["H3_x"] = x

    print("\nH4 cross-asset relative momentum, long only, monthly")
    P = Panel(H4_U, "business"); sc = score_on(P, lambda s, c: c / c.shift(126) - 1)
    T = ranked(P, sc, lambda n: min(3, n), 1, "M", long_only=True); x = run(P, T, "M"); s = line("fifteen ETFs", x, DESIGN_END)
    draws = [run(P, ranked(P, sc, lambda n: min(3, n), 1, "M", long_only=True, shuffle=True), "M") for _ in range(N_PLACEBO)]
    pct, p95 = placebo_pct(x, draws); print(f"  placebo: above {pct:.0f}% of random picks among positive momentum (95th percentile SR {p95:+.2f})")
    allpos = run(P, ranked(P, sc, lambda n: n, 1, "M", long_only=True), "M"); line("all positive-momentum assets (baseline)", allpos, DESIGN_END)
    dbl = stats(run(P, T, "M", 2.0)); print(f"  double costs: {fmt(dbl)}")
    results["H4"] = gate("H4", s, None, pct, dbl); results["H4_x"] = x

    print("\nH5 volatility-managed index exposure, monthly")

    def managed(syms: tuple[str, ...]) -> tuple[pd.Series, pd.Series, pd.Series]:
        P = Panel(syms, "business"); Tm = pd.DataFrame(index=P.idx, columns=P.syms, dtype=float); Tu = Tm.copy()
        for s in syms:
            r = P.r[s]; v2 = r.rolling(21).var(); med = v2.expanding(252).median()
            Tm[s] = P.on(s, (med / v2).clip(upper=CAP)) / len(syms); Tu[s] = P.on(s, v2.notna().astype(float).where(med.notna())) / len(syms)
        return run(P, Tm, "M"), run(P, Tu, "M"), run(P, Tm, "M", 2.0)

    xm, xu, xd = managed(H5_DESIGN); s = line("SPY and QQQ managed", xm, DESIGN_END); line("SPY and QQQ constant", xu, DESIGN_END)
    mm, mu = xm.resample("ME").sum(), xu.resample("ME").sum()
    a, at = nw_alpha(mm, mu); ad, adt = nw_alpha(mm[mm.index <= DESIGN_END], mu[mu.index <= DESIGN_END])
    mid = mm[mm.index <= DESIGN_END].index; mid = mid[0] + (mid[-1] - mid[0]) / 2
    a1 = nw_alpha(mm[mm.index < mid], mu[mu.index < mid])[1]; a2 = nw_alpha(mm[(mm.index >= mid) & (mm.index <= DESIGN_END)], mu[(mu.index >= mid) & (mu.index <= DESIGN_END)])[1]
    ah, aht = nw_alpha(mm[mm.index > DESIGN_END], mu[mu.index > DESIGN_END])
    print(f"  alpha over constant exposure: full {a * 100:+.2f}%/month t {at:+.2f}; design halves t {a1:+.2f} / {a2:+.2f}; held-out {ah * 100:+.2f}% t {aht:+.2f}")
    xmd = xd.resample("ME").sum(); dt = nw_alpha(xmd, mu)[1]; print(f"  double costs alpha t {dt:+.2f}")
    fm, fu, _ = managed(H5_FRESH); fa, fat = nw_alpha(fm.resample("ME").sum(), fu.resample("ME").sum())
    bm, bu, _ = managed(("BTC",)); ba, bat = nw_alpha(bm.resample("ME").sum(), bu.resample("ME").sum())
    print(f"  fresh ETFs alpha {fa * 100:+.2f}%/month t {fat:+.2f}; BTC alpha {ba * 100:+.2f}% t {bat:+.2f}")
    ok = at >= 3 and a1 > 0 and a2 > 0 and aht > 0 and fat > 0 and dt >= 2
    print(f"  GATE H5: {'PASS' if ok else 'FAIL'}"); results["H5"] = ok; results["H5_x"] = xm

    print("\nCorrelation of daily returns with MAX-10 (sum of R by exit day), 2017 on:")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        x = results[f"{h}_x"]; j = pd.concat([x, m10], axis=1).dropna(); j = j[j.index >= "2017-09-01"]
        print(f"  {h}: {j.corr().iat[0, 1]:+.2f} (n {len(j)})")
    print("\nPASSED: " + (", ".join(h for h in ("H1", "H2", "H3", "H4", "H5") if results[h]) or "none"))


if __name__ == "__main__":
    main()
