"""Execution comparison, 23 Oct 2026: MAX-10's one-day hold filled the way the 08:30 run fills it (08:00 UTC open after
the signal close, out at the 08:00 UTC open a day later, stops on hourly lows), against its placebo with the same
fills. Read with section 3 of swing_b3_checks.py, which does the same for the swing hold. 2022 on (hourly data).

Usage: python swing_exec_compare.py [cache_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import swing_batch2 as B  # noqa: E402

COINS = ("BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def hourly(sym: str, start: str = "2022-01", end: str = B.END[:7]) -> pd.DataFrame:
    """Hourly bars from static-klines, fetching only the monthly files that hold data in range; cached if asked."""
    if CACHE is not None and (CACHE / f"{sym}_1h.pkl").exists():
        return pd.read_pickle(CACHE / f"{sym}_1h.pkl")
    pre = f".klines-cache/{sym}USDT/1h/"
    names = sorted(f for f in E._skl_files() if f.startswith(pre) and start <= f.rsplit("/", 1)[1][:7] <= end)
    rows = [r for n in names for r in json.loads(E._get(E.SKL_RAW + n))]
    d = pd.DataFrame([r[:5] for r in rows], columns=["t", "o", "h", "l", "c"]).astype(float)
    d.index = pd.to_datetime(d.t.astype("int64"), unit="ms"); d = d[~d.index.duplicated()].sort_index().drop(columns="t")
    d = E.completed(d, "1h")
    if CACHE is not None:
        CACHE.mkdir(parents=True, exist_ok=True); d.to_pickle(CACHE / f"{sym}_1h.pkl")
    return d


def one_day(sym: str, placebo: bool = False) -> pd.DataFrame:
    d = B.crypto(sym); h = hourly(sym); a = E.atr(d).values; c = d.c.values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; hidx = h.index; ho, hl = h.o.values, h.l.values; rows = []
    for k in range(int(np.searchsorted(d.index, hidx[0])) + 20, len(d) - 2):
        if not (placebo or sig[k]) or not np.isfinite(a[k]):
            continue
        p = hidx.searchsorted(d.index[k] + pd.Timedelta(days=1, hours=8)); r = hidx.searchsorted(d.index[k] + pd.Timedelta(days=2, hours=8))
        if r >= len(hidx):
            break
        e, st = ho[p], c[k] - a[k]
        if e <= st:
            continue
        hit = next((x for x in range(p, r) if hl[x] <= st), None)
        ex = min(st, ho[hit]) if hit is not None else ho[r]
        rows.append(dict(sym=sym, t=d.index[k], R=(ex - e) / a[k] - (2 * B.SIDE_CR + B.F_CR) * e / a[k]))
    return pd.DataFrame(rows)


def close_to_close(sym: str, placebo: bool = False) -> pd.DataFrame:
    d = B.crypto(sym); c, lo = d.c.values, d.l.values; a = E.atr(d).values
    sig = (d.c >= d.c.rolling(10).max().shift(1)).values; rows = []
    for k in range(20, len(d) - 1):
        if (placebo or sig[k]) and np.isfinite(a[k]) and d.index[k] >= pd.Timestamp("2022-01-21"):
            e, st = c[k], c[k] - a[k]; ex = st if lo[k + 1] <= st else c[k + 1]
            rows.append(dict(sym=sym, t=d.index[k], R=(ex - e) / a[k] - (2 * B.SIDE_CR + B.F_CR) * e / a[k]))
    return pd.DataFrame(rows)


def main() -> None:
    for name, f in (("one-day hold, close to close", close_to_close), ("one-day hold, 08:00 UTC fills", one_day)):
        S = pd.concat([f(s) for s in COINS]); P = pd.concat([f(s, True) for s in COINS])
        print(f"{name}: signal {B.fmtR(S.R)}; placebo {B.fmtR(P.R)}; difference {S.R.mean() - P.R.mean():+.3f}R")
        core = S[S.sym.isin(("BTC", "ETH", "SOL", "XRP"))]; corep = P[P.sym.isin(("BTC", "ETH", "SOL", "XRP"))]
        print(f"   core four: signal {B.fmtR(core.R)}; placebo {B.fmtR(corep.R)}")


if __name__ == "__main__":
    main()
