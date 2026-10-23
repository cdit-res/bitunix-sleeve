"""Disproof checks on B3 (MAX-10 entry, swing exit), run 23 Oct 2026 after B3 passed the batch 2 gate.

1. By year and by coin.
2. Regime at entry: BTC daily trend (close against a rising or falling EMA50) and the coin's volatility state.
3. Execution as the 08:30 run trades it (2022 on, hourly data): entry at the 08:00 UTC open after the signal close,
   exits at the 08:00 UTC open after an exit close, stops intraday on hourly lows. Placebo with the same execution.
4. Exit sensitivity: close below the prior 2, 3 or 5 closes; ten-day cap 5, 10 or 20. Reported, not selected.
5. Book: at most three rows at once (4% risk each, core coins first), marked to market daily, against MAX-10's
   one-day hold under the same cap.
Usage: python swing_b3_checks.py
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
SIDE, FUND, RISK = B.SIDE_CR, B.F_CR, 0.04


def trades(sym: str, look: int = 3, cap: int = 10, placebo: bool = False, mult: float = 1.0) -> pd.DataFrame:
    d = B.crypto(sym); c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; rows = []; k = 20; n = len(d)
    while k < n - 1:
        if not (placebo or sig[k]) or not np.isfinite(a[k]):
            k += 1; continue
        e, st = c[k], c[k] - a[k]; ex = None; j = k + 1; why = ""
        while j < n:
            if o[j] <= st:
                ex, why = o[j], "stop"; break
            if lo[j] <= st:
                ex, why = st, "stop"; break
            if c[j] < c[j - look:j].min():
                ex, why = c[j], "low"; break
            if j >= k + cap:
                ex, why = c[j], "time"; break
            j += 1
        if ex is None:
            break
        R = (ex - e) / a[k] - (2 * SIDE + FUND * (j - k)) * e / a[k] * mult
        rows.append(dict(sym=sym, t=d.index[k], t_exit=d.index[j], k=k, j=j, days=j - k, R=R, why=why, atr_pct=a[k] / e))
        k = k + 1 if placebo else j + 1
    return pd.DataFrame(rows)


def setR(df: pd.DataFrame) -> str:
    return B.fmtR(df.R.values) if len(df) else "n 0"


def regimes(T: pd.DataFrame) -> pd.DataFrame:
    btc = B.crypto("BTC").c; e = btc.ewm(span=50, adjust=False).mean()
    lab = pd.Series(np.where((btc > e) & (e > e.shift(5)), "up", np.where((btc < e) & (e < e.shift(5)), "down", "flat")), index=btc.index)
    T = T.copy(); T["btc"] = lab.reindex(T.t).values
    vs = {}
    for s in T.sym.unique():
        d = B.crypto(s); ap = E.atr(d) / d.c; med = ap.rolling(180, min_periods=60).median()
        vs[s] = pd.Series(np.where(ap > med, "high vol", "low vol"), index=d.index)
    T["vol"] = [vs[s].get(t, np.nan) for s, t in zip(T.sym, T.t)]
    return T


def hourly(sym: str) -> pd.DataFrame:
    import swing_exec_compare as X  # bounded loader: only the monthly files that hold data
    return X.hourly(sym)


def exec_0830(sym: str, placebo: bool = False) -> pd.DataFrame:
    """Signals and exits from daily closes; fills at the 08:00 UTC hourly open the next morning; stops on hourly lows."""
    d = B.crypto(sym); h = hourly(sym); a = E.atr(d).values; c = d.c.values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; rows = []; n = len(d)
    hidx = h.index; ho, hl = h.o.values, h.l.values
    k = int(np.searchsorted(d.index, hidx[0])) + 20
    while k < n - 2:
        if not (placebo or sig[k]) or not np.isfinite(a[k]):
            k += 1; continue
        t_in = d.index[k] + pd.Timedelta(days=1, hours=8); p = hidx.searchsorted(t_in)
        if p >= len(hidx):
            break
        e, st = ho[p], c[k] - a[k]
        if e <= st:
            k += 1; continue
        ex = None; j = k + 1; q = p
        while j < n:
            day_end = d.index[j] + pd.Timedelta(days=1); q_end = hidx.searchsorted(day_end)
            # stop on hourly bars from the fill to the end of day j
            hit = next((x for x in range(q, q_end) if hl[x] <= st), None)
            if hit is not None:
                ex = min(st, ho[hit]); break
            q = q_end
            if c[j] < c[j - 3:j].min() or j >= k + 10:
                t_out = d.index[j] + pd.Timedelta(days=1, hours=8); r = hidx.searchsorted(t_out)
                if r >= len(hidx):
                    break
                hit = next((x for x in range(q, r) if hl[x] <= st), None)  # stop can still hit before the exit hour
                ex = min(st, ho[hit]) if hit is not None else ho[r]; j += 1; break
            j += 1
        if ex is None:
            break
        days = j - k
        R = (ex - e) / a[k] - (2 * SIDE + FUND * days) * e / a[k]
        rows.append(dict(sym=sym, t=d.index[k], R=R))
        k = k + 1 if placebo else j
    return pd.DataFrame(rows)


def book(syms: tuple[str, ...], swing: bool, start: str = "2020-10-01", slots: int = 3) -> pd.Series:
    """Daily R of a capped book, marked to market, core coins first. Returns daily R summed across open rows."""
    data = {s: B.crypto(s) for s in syms}
    cands = []
    for s in syms:
        d = data[s]; a = E.atr(d)
        if swing:
            T = trades(s)
            for _, r in T.iterrows():
                cands.append((r.t, s in CORE, s, int(r.k), int(r.j), float(a.iat[int(r.k)])))
        else:
            sig = d.c >= d.c.rolling(10).max().shift(1)
            for k in np.flatnonzero(sig.values):
                if 20 <= k < len(d) - 1:
                    cands.append((d.index[k], s in CORE, s, int(k), int(k + 1), float(a.iat[k])))
    cands.sort(key=lambda x: (x[0], not x[1]))
    days = pd.date_range(start, B.END); pnl = pd.Series(0.0, index=days); open_until: list[pd.Timestamp] = []
    for t, _, s, k, j, a in cands:
        if t < pd.Timestamp(start):
            continue
        open_until = [u for u in open_until if u > t]
        if len(open_until) >= slots:
            continue
        d = data[s]; c, lo, o = d.c.values, d.l.values, d.o.values; e, st = c[k], c[k] - a; prev = e
        for x in range(k + 1, j + 1):
            if o[x] <= st:
                px = o[x]
            elif lo[x] <= st:
                px = st
            else:
                px = c[x]
            day = d.index[x]
            if day in pnl.index:
                pnl[day] += (px - prev) / a - FUND * e / a
            prev = px
            if px != c[x]:
                break
        if d.index[k] in pnl.index:
            pnl[d.index[k]] -= SIDE * e / a
        if d.index[j] in pnl.index:
            pnl[d.index[j]] -= SIDE * e / a
        open_until.append(d.index[j])
    return pnl


def book_stats(name: str, pnl: pd.Series) -> None:
    yrs = len(pnl) / 365.25; eq = (1 + RISK * pnl).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    m = (1 + RISK * pnl).groupby(pnl.index.to_period("M")).prod() - 1
    print(f"  {name}: {pnl.sum() / yrs:+.1f}R a year, daily SR {pnl.mean() / pnl.std() * np.sqrt(365):.2f}, "
          f"t {pnl.mean() / pnl.std() * np.sqrt(len(pnl)):+.2f}, worst drawdown at 4% a row {dd * 100:.0f}%, "
          f"worst month {m.min() * 100:+.0f}%, best month {m.max() * 100:+.0f}%, equity x{eq.iat[-1]:.1f} over {yrs:.1f} years")
    for y, g in pnl.groupby(pnl.index.year):
        print(f"     {y}: {g.sum():+.1f}R")


def main() -> None:
    T = pd.concat([trades(s) for s in CORE + FRESH]); core = T[T.sym.isin(CORE)]
    print("1. B3 by year (core and fresh pooled) and by coin")
    for y, g in T.groupby(T.t.dt.year):
        print(f"   {y}: {setR(g)}")
    for s, g in T.groupby("sym"):
        print(f"   {s}: {setR(g)}; exits: {g.why.value_counts().to_dict()}; median hold {g.days.median():.0f} days")
    print("\n2. Regime at entry (core and fresh pooled), signal against placebo")
    P = pd.concat([trades(s, placebo=True) for s in CORE + FRESH]); RT, RP = regimes(T), regimes(P)
    for col in ("btc", "vol"):
        for v in sorted(RT[col].dropna().unique()):
            a, b = RT[RT[col] == v], RP[RP[col] == v]
            print(f"   {col} {v}: signal {setR(a)}; placebo {setR(b)}; difference {a.R.mean() - b.R.mean():+.3f}R")
    print("\n3. Execution at the 08:30 run (2022 on, hourly fills and stops)")
    X = pd.concat([exec_0830(s) for s in CORE + FRESH]); XP = pd.concat([exec_0830(s, placebo=True) for s in CORE + FRESH])
    base = T[T.t >= "2022-01-01"]; basep = P[P.t >= "2022-01-01"]
    print(f"   at the daily close: signal {setR(base)}; placebo {setR(basep)}")
    print(f"   at 08:00 UTC fills: signal {setR(X)}; placebo {setR(XP)}; difference {X.R.mean() - XP.R.mean():+.3f}R")
    for s, g in X.groupby("sym"):
        print(f"      {s}: {setR(g)}")
    print("\n4. Exit sensitivity (core design to 2023 | core held-out 2024 on | fresh), signal minus placebo")
    for look in (2, 3, 5):
        for cap in (5, 10, 20):
            S = pd.concat([trades(s, look, cap) for s in CORE + FRESH]); Q = pd.concat([trades(s, look, cap, True) for s in CORE + FRESH])
            parts = []
            for name, f in (("design", lambda x: x.sym.isin(CORE) & (x.t <= "2023-12-31")), ("held", lambda x: x.sym.isin(CORE) & (x.t > "2023-12-31")),
                            ("fresh", lambda x: x.sym.isin(FRESH))):
                a, b = S[f(S)], Q[f(Q)]; parts.append(f"{a.R.mean():+.3f}R (t {B.tstat(a.R):+.1f}) vs {b.R.mean():+.3f}R")
            print(f"   exit below prior {look} closes, cap {cap} days: " + " | ".join(parts))
    print("\n5. Book, three rows at most (4% risk each), marked to market, Oct 2020 on")
    book_stats("B3 swing, core four", book(CORE, True)); book_stats("MAX-10 one-day, core four", book(CORE, False))
    book_stats("B3 swing, core four plus fresh six", book(CORE + FRESH, True))


if __name__ == "__main__":
    main()
