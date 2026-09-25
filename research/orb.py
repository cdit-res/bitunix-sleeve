"""15-minute day trading, pre-registered 25 Sep 2026 before running (Cole: day trading on the 15m chart).

ORB: opening-range breakout at the US cash open (09:30 New York), after Zarattini and Aziz (2023), who found the
  first 5-minute candle's direction profitable for QQQ day trades. Crypto's busiest flows since the spot ETFs sit in
  US hours, so the open is the natural session anchor.
  ORB-5: first 5m candle 09:30-09:35 ET. Close above open: long at 09:35 (taker plus slippage), stop at that candle's
  low; close below open: short, stop at its high. Target 10R, else exit at 16:00 ET at market.
  ORB-15: the same on the first 15m candle (09:30-09:45 ET), entry at 09:45.
  Weekdays only. Skip when the stop is under 0.15% (fees alone would exceed 1R).
  BTC: Bitstamp 5m, 2016-06 to 2026-09; design to 2022-12-31, held out 2023 onward.
  Fresh: ETH, SOL, XRP, LINK, BNB, ADA, DOGE, AVAX, DOT on Binance 15m, 2025-01 to 2026-09 (ORB-15 only).
TS-D: trend-ensemble shorts only while the sleeve's market trend is down (BTC daily close below a falling EMA50,
  entry 41), after remaining_lines.py showed the short side paid in 2018 and 2022 but not overall.
Gate as elsewhere.

Usage: SLEEVE_DATA=/path/to/data python orb.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import remaining_lines as RL  # noqa: E402
import trend_ensemble as TE  # noqa: E402

DATA = TE.DATA; MAKER, TAKER, SLIP = TE.MAKER, TE.TAKER, TE.SLIP
ALTS = RL.ALTS; MIN_STOP = 0.0015


def orb(bars: pd.DataFrame, minutes: int, sym: str) -> list[dict]:
    """bars: 5m or 15m OHLC indexed by UTC open time."""
    step = int((bars.index[1] - bars.index[0]).total_seconds() // 60)
    O, H, L, C = (bars[k].values for k in ("o", "h", "l", "c")); idx = bars.index; out = []
    days = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="B")
    for day in days:
        open_et = pd.Timestamp(f"{day.date()} 09:30", tz="America/New_York").tz_convert("UTC").tz_localize(None)
        close_et = pd.Timestamp(f"{day.date()} 16:00", tz="America/New_York").tz_convert("UTC").tz_localize(None)
        k0 = idx.searchsorted(open_et); n = minutes // step
        if k0 + n >= len(idx) or idx[k0] != open_et:
            continue
        o, c = O[k0], C[k0 + n - 1]; hi, lo = H[k0:k0 + n].max(), L[k0:k0 + n].min()
        if c == o:
            continue
        side = 1 if c > o else -1; k = k0 + n
        entry = O[k] * (1 + side * SLIP); stop = lo if side == 1 else hi
        risk = side * (entry - stop)
        if risk <= 0 or risk / entry < MIN_STOP:
            continue
        tgt = entry + side * 10 * risk; kend = min(idx.searchsorted(close_et) - 1, len(C) - 1); R = None
        for q in range(k, kend + 1):
            if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                R = -1 - (TAKER + TAKER + SLIP) * entry / risk; break
            if q > k and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)):
                R = 10 - (TAKER + MAKER) * entry / risk; break
        if R is None:
            R = side * (C[kend] - entry) / risk - (TAKER + TAKER + SLIP) * entry / risk
        out.append(dict(sym=sym, t=idx[k], side=side, R=R, stop_pct=risk / entry * 100))
    return out


def run_orb() -> None:
    m5 = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
    m15 = pd.read_parquet(DATA / "BTC_bitstamp_15m.parquet")
    for lab, bars, minutes in (("ORB-5", m5, 5), ("ORB-15", m15, 15)):
        T = pd.DataFrame(orb(bars, minutes, "BTC")); des = T[T.t < "2023-01-01"]; ho = T[T.t >= "2023-01-01"]
        print(f"{lab} BTC: design {RL.summ(des.R)} | halves {RL.halves(des)} | held-out {RL.summ(ho.R)} | "
              f"longs {RL.summ(T[T.side == 1].R)} | shorts {RL.summ(T[T.side == -1].R)} | median stop {T.stop_pct.median():.2f}%")
    fr = []
    for s in ALTS:
        b = pd.read_parquet(DATA / f"{s}_15m.parquet"); fr += orb(b, 15, s)
    F = pd.DataFrame(fr)
    print(f"ORB-15 fresh alts 2025-26: {RL.summ(F.R)} | median stop {F.stop_pct.median():.2f}%")


def run_tsd() -> None:
    bh4, bd = TE.bars("BTC"); e = bd.c.ewm(span=50, adjust=False).mean()
    down = ((bd.c < e) & (e < e.shift(5))); down.index = down.index + pd.Timedelta(days=1)  # known next day
    rows = []
    for s in TE.CORE + TE.FRESH:
        h4, d = TE.bars(s); rows += RL.trend_short(s, h4, d)
    T = pd.DataFrame(rows); T["down"] = [bool(down.asof(t)) if t >= down.index[0] else False for t in T.t]
    T = T[T.down]; core = T[T.sym.isin(TE.CORE)]; des = core[core.t < TE.HELD_FROM]
    print(f"TS-D shorts in a down market: design {RL.summ(des.R)} | halves {RL.halves(des)} | "
          f"held-out {RL.summ(core[core.t >= TE.HELD_FROM].R)} | fresh {RL.summ(T[T.sym.isin(TE.FRESH)].R)}")


if __name__ == "__main__":
    run_orb()
    run_tsd()
