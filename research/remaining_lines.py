"""Remaining lines, pre-registered 25 Sep 2026 before running. Each is one hypothesis family; every variant counted.

TS: trend ensemble short side (owner request: shorts with Donchian worked for him). Mirror of trend_ensemble.py T1 with the
    0.5% stop rail: short on a daily close at or below the lowest of the prior N closes (N = 5 ... 360), stop at the
    midpoint of the N-day closing range trailed down daily, never raised; exit when a 4h high reaches the stop.
    Funding counted as zero for shorts (they usually receive it). Sets as T1.
EB: macro-event bracket, BTC on 5m bars (Bitstamp), CPI and FOMC 2017-2026 (macro_events.csv, official dates).
    Reference = close of the 5m bar ending at the release time. Buy stop at +x, sell stop at -x; first to trigger is
    the trade, the other cancels. Stop back at the reference price (risk = x). Target 3R. Exit at market 2h after the
    release. Variants: x = 0.25% and x = 0.5%. Design 2017-2022, held out 2023-2026.
SF: liquidation-spike fade, BTC 5m 2016-2026. A spike is a 5m bar whose return is at least 6 standard deviations of
    the prior 24h of 5m returns and whose range is at least 4 times the prior 24h median range. Fade it at the next
    bar's open (taker), stop 0.1 x the spike range beyond the spike's extreme, target half the spike range back from
    the extreme toward the spike's open, exit at market after 2h. Design to 2023-12-31, held out after.
LL: BTC to alt lead-lag, 15m Binance bars, 2025-01 to 2026-09. When BTC's 15m return is at least 2 standard
    deviations (prior 24h) and an alt's same-bar return is under half its 30-day beta times BTC's, buy (or sell) the
    alt at the next open, taker both sides. Variants: exit after 1 bar and after 4 bars. Net % per trade reported.
    Design 2025, held out 2026.
Gate as elsewhere: design t >= 3, the same sign in both halves, held-out above zero.

Usage: SLEEVE_DATA=/path/to/data python remaining_lines.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import trend_ensemble as TE  # noqa: E402

DATA = TE.DATA
MAKER, TAKER, SLIP = TE.MAKER, TE.TAKER, TE.SLIP
ALTS = ("ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")


def summ(x) -> str:
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


def halves(d: pd.DataFrame, col: str = "R") -> str:
    mid = d.t.min() + (d.t.max() - d.t.min()) / 2
    return f"{summ(d[d.t < mid][col])} / {summ(d[d.t >= mid][col])}"


# ---------- TS ----------
def trend_short(sym: str, h4: pd.DataFrame, d: pd.DataFrame) -> list[dict]:
    O, H = h4.o.values, h4.h.values; t4 = h4.index; dc = d.c.values; days = d.index
    at8 = t4.searchsorted(days + pd.Timedelta(hours=8)); out = []
    for N in TE.LOOKBACKS:
        lo = pd.Series(dc).rolling(N).min().shift(1).values
        mx = pd.Series(dc).rolling(N).max().values; mn = pd.Series(dc).rolling(N).min().values; mid = (mx + mn) / 2
        i = N + 1
        while i < len(days) - 2:
            if not dc[i] <= lo[i]:
                i += 1; continue
            k = at8[i + 1]
            if k >= len(O) - 1:
                break
            entry = O[k] * (1 - SLIP); stop = mid[i]
            if stop <= entry or (stop - entry) / entry < float(os.environ["MIN_STOP"]):
                i += 1; continue
            risk = stop - entry; stop_now = stop; j = i + 1; exit_px = None; kk = k
            while exit_px is None:
                k_next = at8[j + 1] if j + 1 < len(days) else len(O)
                for kk in range(max(k, at8[j]) if j > i + 1 else k, min(k_next, len(O))):
                    if H[kk] >= stop_now:
                        exit_px = max(stop_now, O[kk]) * (1 + SLIP); break
                if exit_px is None:
                    if j >= len(days) - 2:
                        exit_px = dc[j] * (1 + SLIP); kk = min(k_next, len(O)) - 1; break
                    stop_now = min(stop_now, mid[j]); j += 1
            R = (entry - exit_px) / risk - 2 * TAKER * entry / risk
            out.append(dict(sym=sym, N=N, t=t4[k], R=R))
            i = max(int(days.searchsorted(t4[kk].normalize())), i + 1)
    return out


def run_ts() -> None:
    rows = []
    for s in TE.CORE + TE.FRESH:
        h4, d = TE.bars(s); rows += trend_short(s, h4, d)
    T = pd.DataFrame(rows); core = T[T.sym.isin(TE.CORE)]; des = core[core.t < TE.HELD_FROM]
    print(f"TS trend shorts: design {summ(des.R)} | halves {halves(des)} | held-out {summ(core[core.t >= TE.HELD_FROM].R)} | fresh {summ(T[T.sym.isin(TE.FRESH)].R)}")
    T["yr"] = T.t.dt.year
    print("   by year: " + " | ".join(f"{y}: {g.R.mean():+.2f} ({len(g)})" for y, g in T.groupby("yr")))


# ---------- EB ----------
def run_eb(m5: pd.DataFrame) -> None:
    ev = pd.read_csv(DATA / "macro_events.csv")
    ev["ts"] = [pd.Timestamp(f"{d} {t}", tz="America/New_York").tz_convert("UTC").tz_localize(None) for d, t in zip(ev.date, ev.time_et)]
    O, H, L, C = (m5[k].values for k in ("o", "h", "l", "c")); idx = m5.index
    for x in (0.0025, 0.005):
        rows = []
        for e in ev.itertuples():
            k0 = idx.searchsorted(e.ts)  # first bar starting at or after the release
            if k0 <= 0 or k0 >= len(idx) - 30 or idx[k0] != e.ts:
                continue
            ref = C[k0 - 1]; up, dn = ref * (1 + x), ref * (1 - x); end = min(k0 + 24, len(C) - 1); side = 0
            for k in range(k0, end):
                hit_up, hit_dn = H[k] >= up, L[k] <= dn
                if hit_up and hit_dn:
                    side = 2; break
                if hit_up or hit_dn:
                    side = 1 if hit_up else -1; break
            if side == 0:
                rows.append(dict(t=e.ts, ev=e.event, R=0.0, how="none")); continue
            if side == 2:  # both levels in one bar: count the worse case, a loss on the first leg
                rows.append(dict(t=e.ts, ev=e.event, R=-1 - (TAKER + TAKER + 2 * SLIP) * ref / (x * ref), how="both")); continue
            entry = (up if side == 1 else dn) * (1 + side * SLIP); stop = ref; risk = side * (entry - stop)
            tgt = entry + side * 3 * risk; R = None
            for q in range(k, end + 1):
                if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                    R = -1 - (TAKER + TAKER + SLIP) * entry / risk; break
                if q > k and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)):
                    R = 3 - (TAKER + MAKER) * entry / risk; break
            if R is None:
                R = side * (C[end] - entry) / risk - (TAKER + TAKER + SLIP) * entry / risk
            rows.append(dict(t=e.ts, ev=e.event, R=R, how="trade"))
        T = pd.DataFrame(rows); des = T[T.t < "2023-01-01"]; ho = T[T.t >= "2023-01-01"]
        print(f"EB bracket x={x * 100:.2f}%: design {summ(des.R)} | halves {halves(des)} | held-out {summ(ho.R)} | "
              f"CPI {summ(T[T.ev == 'CPI'].R)} | FOMC {summ(T[T.ev.str.startswith('FOMC')].R)}")


# ---------- SF ----------
def run_sf(m5: pd.DataFrame) -> None:
    O, H, L, C = (m5[k].values for k in ("o", "h", "l", "c")); idx = m5.index
    r = pd.Series(np.log(C)).diff(); sd = r.rolling(288).std().shift(1).values
    rng = pd.Series(H - L); medr = rng.rolling(288).median().shift(1).values; rows = []; k = 300
    while k < len(C) - 30:
        if not (np.isfinite(sd[k]) and abs(r.iat[k]) >= 6 * sd[k] and (H[k] - L[k]) >= 4 * medr[k]):
            k += 1; continue
        side = -1 if r.iat[k] > 0 else 1  # fade
        ext = H[k] if side == -1 else L[k]; span = H[k] - L[k]
        j = k + 1; entry = O[j] * (1 + side * SLIP); stop = ext - side * 0.1 * span  # beyond the extreme
        tgt = ext + side * 0.5 * span
        risk = side * (entry - stop)
        if risk <= 0 or side * (tgt - entry) <= 0:
            k += 1; continue
        R = None; end = min(j + 24, len(C) - 1)
        for q in range(j, end + 1):
            if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                R = -1 - (TAKER + TAKER + SLIP) * entry / risk; break
            if q > j and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)):
                R = side * (tgt - entry) / risk - (TAKER + MAKER) * entry / risk; break
        if R is None:
            R = side * (C[end] - entry) / risk - (TAKER + TAKER + SLIP) * entry / risk
        rows.append(dict(t=idx[j], R=R, stop_pct=risk / entry * 100)); k = end + 1
    T = pd.DataFrame(rows); des = T[T.t < "2024-01-01"]; ho = T[T.t >= "2024-01-01"]
    print(f"SF spike fade: design {summ(des.R)} | halves {halves(des)} | held-out {summ(ho.R)} | median stop {T.stop_pct.median():.2f}%")


# ---------- LL ----------
def run_ll() -> None:
    b = pd.read_parquet(DATA / "BTCUSDT_15m.parquet"); rb = np.log(b.c).diff()
    sdb = rb.rolling(96).std().shift(1)
    for hold in (1, 4):
        rows = []
        for s in ALTS:
            a = pd.read_parquet(DATA / f"{s}_15m.parquet").reindex(b.index)
            ra = np.log(a.c).diff()
            beta = (ra.rolling(2880).cov(rb) / rb.rolling(2880).var()).shift(1)
            trig = (rb.abs() >= 2 * sdb) & ((ra * np.sign(rb)) < 0.5 * beta * rb.abs())
            ks = np.nonzero(trig.fillna(False).values)[0]
            O, C = a.o.values, a.c.values
            last = -10
            for k in ks:
                if k + hold >= len(C) or k <= last + hold:
                    continue
                side = int(np.sign(rb.iat[k])); e = O[k + 1]; x = C[k + hold]
                if not (np.isfinite(e) and np.isfinite(x)):
                    continue
                net = side * (x / e - 1) * 100 - (2 * TAKER + 2 * SLIP) * 100
                rows.append(dict(sym=s, t=b.index[k], ret=net)); last = k
        T = pd.DataFrame(rows); des = T[T.t < "2026-01-01"]; ho = T[T.t >= "2026-01-01"]
        print(f"LL lead-lag hold {hold} bar(s), net % per trade: design {summ(des.ret)} | halves {halves(des, 'ret')} | held-out {summ(ho.ret)}")


if __name__ == "__main__":
    run_ts()
    m5 = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
    run_eb(m5)
    run_sf(m5)
    run_ll()
