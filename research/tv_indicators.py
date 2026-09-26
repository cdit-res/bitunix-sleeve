"""The owner's TradingView tools rebuilt in Python, pre-registered 25 Sep 2026 before running.

H: Harmonic XABCD patterns (the LonesomeTheBlue script is closed source, so this uses the standard ratio table in
   indicators_spec.md). ZigZag from confirmed pivots, period P = 8 bars each side, so a pivot is known P bars late.
   After C is confirmed, if AB/XA and BC/AB fit a pattern within 10% tolerance, a limit order rests at the
   pattern's D level (D = A - AD x (A - X) for a bullish pattern), provided CD/BC at that level also fits.
   Patterns: Gartley (AB .618, BC .382-.886, CD 1.13-1.618, AD .786), Bat (.382-.5, .382-.886, 1.618-2.618, .886),
   Alt Bat (.382, .382-.886, 2.0-3.618, 1.13), Butterfly (.786, .382-.886, 1.618-2.618, 1.27-1.618),
   Crab (.382-.618, .382-.886, 2.24-3.618, 1.618), Deep Crab (.886, .382-.886, 2.0-3.618, 1.618).
   Cancel if price breaks beyond C first or after 3 x the X-to-C duration. Stop 1.0 ATR(14) beyond D.
   Split ticket: half at 0.382 and half at 0.618 of AD back toward A. Exit at market after 5 days.
L: LuxAlgo "Trendlines with Breaks", exact published logic: pivots length 14, slope = ATR(14) / 14 x 1.0,
   upper line from the last pivot high falling by the slope each bar, lower line from the last pivot low rising.
   Signal when the close crosses the live line (upper - slope x 14 / lower + slope x 14). Entry at the next bar's
   open (taker), stop 1.5 ATR(14), target 3R, exit at market after 5 days.
Both on 1h and 4h bars (two variants each), long and short. Costs as the other scripts. Stop first.
Sets: BTC, ETH, SOL, XRP design to 2023-12-31 (1h alts to 2024-12-31), held out after; LINK, BNB, ADA, DOGE, AVAX, DOT fresh.
Gate: design t >= 3, the same sign in both halves, held-out and fresh above zero.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("SLEEVE_DATA", Path(__file__).resolve().parent.parent / "data"))
MAKER, TAKER, SLIP = 0.0002, 0.0006, 0.0002
CORE = ("BTC", "ETHUSDT", "SOLUSDT", "XRPUSDT"); FRESH = ("LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")
PATTERNS = {  # AB/XA, BC/AB, CD/BC, AD/XA as (lo, hi)
    "gartley": ((.618, .618), (.382, .886), (1.13, 1.618), (.786, .786)),
    "bat": ((.382, .5), (.382, .886), (1.618, 2.618), (.886, .886)),
    "altbat": ((.382, .382), (.382, .886), (2.0, 3.618), (1.13, 1.13)),
    "butterfly": ((.786, .786), (.382, .886), (1.618, 2.618), (1.27, 1.618)),
    "crab": ((.382, .618), (.382, .886), (2.24, 3.618), (1.618, 1.618)),
    "deepcrab": ((.886, .886), (.382, .886), (2.0, 3.618), (1.618, 1.618)),
}
TOL, P = 0.10, 8


def load(sym: str, tf: str) -> pd.DataFrame:
    if sym == "BTC":
        b = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
        return b.resample(tf).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    f = DATA / f"{sym}_{tf}.csv.gz"
    return pd.read_csv(f, index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)


def atr(d: pd.DataFrame, n: int = 14) -> np.ndarray:
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().values


def pivots(d: pd.DataFrame, p: int) -> list[tuple[int, int, float]]:
    """Alternating zigzag of confirmed pivots: (bar index, +1 high / -1 low, price); known at index + p."""
    H, L = d.h.values, d.l.values; raw = []
    for i in range(p, len(d) - p):
        if H[i] == H[i - p:i + p + 1].max():
            raw.append((i, 1, H[i]))
        if L[i] == L[i - p:i + p + 1].min():
            raw.append((i, -1, L[i]))
    zz: list[tuple[int, int, float]] = []
    for pv in raw:
        if zz and zz[-1][1] == pv[1]:
            if (pv[1] == 1 and pv[2] >= zz[-1][2]) or (pv[1] == -1 and pv[2] <= zz[-1][2]):
                zz[-1] = pv
        else:
            zz.append(pv)
    return zz


def fits(r: float, lo: float, hi: float) -> bool:
    return lo * (1 - TOL) <= r <= hi * (1 + TOL)


def resolve(H, L, C, k, side, entry, stop, t1, t2, hold, fee_in):
    risk = side * (entry - stop); legs = []
    for tgt in (t1, t2):
        r = None; end = min(k + hold, len(C) - 1)
        for q in range(k, end + 1):
            if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                r = -1 - (fee_in + TAKER + SLIP) * entry / risk; break
            if q > k and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)):
                r = side * (tgt - entry) / risk - (fee_in + MAKER) * entry / risk; break
        if r is None:
            r = side * (C[end] - entry) / risk - (fee_in + TAKER) * entry / risk
        legs.append(r)
    return legs


def harmonic(sym: str, tf: str) -> list[dict]:
    d = load(sym, tf); H, L, C = d.h.values, d.l.values, d.c.values; A_ = atr(d)
    zz = pivots(d, P); hold = int(pd.Timedelta(days=5) / pd.Timedelta(tf)); out = []; busy = -1
    for j in range(3, len(zz) - 0):
        X, A, B, Cc = zz[j - 3], zz[j - 2], zz[j - 1], zz[j]
        side = 1 if X[1] == -1 else -1  # bullish when X is a low (D will be a low)
        xa, ab, bc = abs(A[2] - X[2]), abs(A[2] - B[2]), abs(Cc[2] - B[2])
        if xa == 0 or ab == 0 or bc == 0:
            continue
        placed = Cc[0] + P  # C known P bars after it prints
        if placed >= len(d) - 2 or placed <= busy:
            continue
        for name, (r_ab, r_bc, r_cd, r_ad) in PATTERNS.items():
            if not (fits(ab / xa, *r_ab) and fits(bc / ab, *r_bc)):
                continue
            ad = (r_ad[0] + r_ad[1]) / 2
            D = A[2] - side * ad * xa
            cd = abs(Cc[2] - D)
            if not fits(cd / bc, *r_cd) or side * (Cc[2] - D) <= 0:
                continue
            expiry = placed + 3 * max(Cc[0] - X[0], 1); k = None
            for q in range(placed, min(expiry, len(d) - 1)):
                if (side == 1 and H[q] > Cc[2]) or (side == -1 and L[q] < Cc[2]):
                    break  # C broken first: pattern void
                if (side == 1 and L[q] <= D) or (side == -1 and H[q] >= D):
                    k = q; break
            if k is None:
                continue
            stop = D - side * A_[k]; t1 = D + side * 0.382 * abs(A[2] - D); t2 = D + side * 0.618 * abs(A[2] - D)
            legs = resolve(H, L, C, k, side, D, stop, t1, t2, hold, MAKER)
            out.append(dict(sym=sym, tf=tf, kind="harmonic", pattern=name, side=side, t=d.index[k], R=float(np.mean(legs))))
            busy = k; break
    return out


def luxalgo(sym: str, tf: str) -> list[dict]:
    d = load(sym, tf); O, H, L, C = (d[k].values for k in ("o", "h", "l", "c")); n = 14; A_ = atr(d, n)
    hold = int(pd.Timedelta(days=5) / pd.Timedelta(tf)); out = []
    upper = lower = np.nan; sph = spl = 0.0; prev_up = prev_lo = np.nan; t = 2 * n; busy = -1
    while t < len(d) - 2:
        i = t - n  # candidate pivot bar, confirmed at t
        if H[i] == H[i - n:t + 1].max():
            upper, sph = H[i], A_[t] / n
        else:
            upper = upper - sph if np.isfinite(upper) else upper
        if L[i] == L[i - n:t + 1].min():
            lower, spl = L[i], A_[t] / n
        else:
            lower = lower + spl if np.isfinite(lower) else lower
        live_up, live_lo = upper - sph * n, lower + spl * n
        sig = 0
        if np.isfinite(prev_up) and C[t - 1] <= prev_up and C[t] > live_up:
            sig = 1
        elif np.isfinite(prev_lo) and C[t - 1] >= prev_lo and C[t] < live_lo:
            sig = -1
        prev_up, prev_lo = live_up, live_lo
        if sig and t + 1 > busy:
            k = t + 1; entry = O[k] * (1 + sig * SLIP); stop = entry - sig * 1.5 * A_[t]
            tgt = entry + sig * 3 * abs(entry - stop)
            legs = resolve(H, L, C, k, sig, entry, stop, tgt, tgt, hold, TAKER)
            out.append(dict(sym=sym, tf=tf, kind="luxalgo", pattern="break", side=sig, t=d.index[k], R=legs[0]))
            busy = k + hold
        t += 1
    return out


def summ(x) -> str:
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"


def main() -> None:
    rows = []
    for tf in ("1h", "4h"):
        for sym in CORE + FRESH:
            if tf == "1h" and sym != "BTC" and not (DATA / f"{sym}_1h.csv.gz").exists():
                continue
            rows += harmonic(sym, tf) + luxalgo(sym, tf)
        print(f"{tf} done", flush=True)
    T = pd.DataFrame(rows); T.to_csv(Path(__file__).with_name("tv_indicators_trades.csv"), index=False)
    for (kind, tf), g in T.groupby(["kind", "tf"]):
        cut = {"BTC": "2024-01-01"}; core = g[g.sym.isin(CORE)]
        held = np.array([t >= pd.Timestamp("2024-01-01" if (s == "BTC" or tf == "4h") else "2025-01-01") for s, t in zip(core.sym, core.t)])
        des, ho = core[~held], core[held]; mid = des.t.min() + (des.t.max() - des.t.min()) / 2
        print(f"\n{kind} {tf}: design {summ(des.R)} | halves {summ(des[des.t < mid].R)} / {summ(des[des.t >= mid].R)} | "
              f"held-out {summ(ho.R)} | fresh {summ(g[g.sym.isin(FRESH)].R)}")
        print(f"   longs {summ(g[g.side == 1].R)} | shorts {summ(g[g.side == -1].R)}")
        if kind == "harmonic":
            print("   by pattern: " + " | ".join(f"{p} {summ(x.R)}" for p, x in g.groupby("pattern")))


if __name__ == "__main__":
    main()
