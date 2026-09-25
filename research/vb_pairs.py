"""Two more families, pre-registered 25 Sep 2026 before running.

VB: Larry Williams volatility breakout, the classic futures day-trading system (widely used on Korean crypto venues
    with k = 0.5). Day = 00:00 to 24:00 UTC. Long when price trades above the day's open + 0.5 x the prior day's
    range (stop-entry fill at that level or the bar's open if beyond, plus slippage, taker); stop at the day's open;
    exit at the next 00:00 UTC open. VB-L long only; VB-S the mirror short (both counted). Funding 0.03% a day on
    longs, zero on shorts. Execution: BTC on 5m (Bitstamp), the other nine majors on 1h (Binance, 2022-26; the
    1h fill-bar stop-first rule is conservative). Sets: BTC design to 2023, held out after; ETH, SOL, XRP design
    2022-2024, held out 2025-26; LINK, BNB, ADA, DOGE, AVAX, DOT fresh.
PM: pair MAX-10 on the ratio to BTC (ETH/BTC, SOL/BTC, XRP/BTC; fresh: LINK, BNB, ADA, DOGE, AVAX, DOT over BTC).
    When the ratio's daily close is at or above its prior 10 closes, long the coin and short BTC at the next 08:00 UTC
    open in equal notional; at a 10-day closing low, the reverse. Stop 1.0 x daily ATR(14) of the ratio; exit at the
    following 08:00 UTC open, rolling while the signal repeats. Costs: maker entry and taker exit on both legs.
    Market beta cancels, so this is the two-sided (long and short) test of the continuation effect.
Gate as elsewhere.

Usage: SLEEVE_DATA=/path/to/data python vb_pairs.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import remaining_lines as RL  # noqa: E402
import trend_ensemble as TE  # noqa: E402

DATA = TE.DATA; MAKER, TAKER, SLIP, FUND = TE.MAKER, TE.TAKER, TE.SLIP, TE.FUND


def vb(sym: str, ex: pd.DataFrame, side: int) -> list[dict]:
    O, H, L, C = (ex[k].values for k in ("o", "h", "l", "c")); idx = ex.index
    d = ex.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    starts = idx.searchsorted(d.index); out = []
    for i in range(1, len(d) - 1):
        k0 = starts[i]; k1 = starts[i + 1] if i + 1 < len(starts) else len(idx)
        if k1 >= len(idx):
            break
        day_open = O[k0]; rng = d.h.iat[i - 1] - d.l.iat[i - 1]; lvl = day_open + side * 0.5 * rng
        if rng <= 0 or 0.5 * rng / day_open < 0.005:  # the sleeve's 0.5% stop rail
            continue
        k = None
        for q in range(k0, k1):
            if (side == 1 and H[q] >= lvl) or (side == -1 and L[q] <= lvl):
                k = q; break
        if k is None:
            continue
        entry = (max(lvl, O[k]) if side == 1 else min(lvl, O[k])) * (1 + side * SLIP); stop = day_open
        risk = side * (entry - stop)
        if risk <= 0:
            continue
        R = None
        # the fill bar: stop counted if the bar also reaches it after the fill (conservative)
        for q in range(k, k1):
            if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop):
                R = -1 - (TAKER + TAKER + SLIP) * entry / risk; break
        if R is None:
            exit_px = O[k1] * (1 - side * SLIP)
            R = side * (exit_px - entry) / risk - 2 * TAKER * entry / risk - (FUND if side == 1 else 0.0) * entry / risk
        out.append(dict(sym=sym, t=idx[k], R=R, stop_pct=risk / entry * 100))
    return out


def run_vb() -> None:
    btc = pd.read_parquet(DATA / "BTC_bitstamp_5m.parquet")
    for side, lab in ((1, "VB-L"), (-1, "VB-S")):
        rows = vb("BTC", btc, side)
        for s in RL.ALTS:
            ex = pd.read_csv(DATA / f"{s}_1h.csv.gz", index_col=0, parse_dates=True)[["o", "h", "l", "c"]].astype(float)
            rows += vb(s, ex, side)
        T = pd.DataFrame(rows)
        core_alt = T.sym.isin(("ETHUSDT", "SOLUSDT", "XRPUSDT")); btcm = T.sym == "BTC"
        des = T[(btcm & (T.t < "2024-01-01")) | (core_alt & (T.t < "2025-01-01"))]
        ho = T[(btcm & (T.t >= "2024-01-01")) | (core_alt & (T.t >= "2025-01-01"))]
        fr = T[~btcm & ~core_alt]
        print(f"{lab}: design {RL.summ(des.R)} | halves {RL.halves(des)} | held-out {RL.summ(ho.R)} | fresh {RL.summ(fr.R)} | median stop {T.stop_pct.median():.2f}%")


def pair(coin: str, base_d: pd.DataFrame, base_h4: pd.DataFrame) -> list[dict]:
    h4, d = TE.bars(coin)
    j = d.index.intersection(base_d.index); d, b = d.loc[j], base_d.loc[j]
    ratio = d.c / b.c; rh = d.h / b.l; rl = d.l / b.h  # conservative ratio range for ATR
    tr = pd.concat([rh - rl, (rh - ratio.shift()).abs(), (rl - ratio.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values; r = ratio.values
    hi = pd.Series(r).rolling(10).max().shift(1).values; lo = pd.Series(r).rolling(10).min().shift(1).values
    j4 = h4.index.intersection(base_h4.index); ho4, bo4 = h4.o.loc[j4], base_h4.o.loc[j4]
    ro = (ho4 / bo4); days = d.index; out = []; i = 15
    at8 = ro.index.searchsorted(days + pd.Timedelta(hours=8))
    while i < len(days) - 3:
        side = 1 if r[i] >= hi[i] else (-1 if r[i] <= lo[i] else 0)
        if side == 0:
            i += 1; continue
        k = at8[i + 1]
        if k >= len(ro) - 7:
            break
        entry = ro.iat[k]; first = True
        while True:
            k_end = min(at8[i + 2], len(ro) - 1); stop = entry - side * atr[i]
            path = ro.iloc[k:k_end].values; hit = np.nonzero(path <= stop)[0] if side == 1 else np.nonzero(path >= stop)[0]
            ex = stop if len(hit) else None
            nxt = 1 if r[i + 1] >= hi[i + 1] else (-1 if r[i + 1] <= lo[i + 1] else 0)
            roll = ex is None and nxt == side and i + 1 < len(days) - 3
            if ex is None:
                ex = ro.iat[k_end]
            cost = 2 * ((MAKER if first else 0.0) + (0.0 if roll else TAKER + SLIP))  # two legs
            R = side * (ex - entry) / atr[i] - cost * entry / atr[i]
            out.append(dict(sym=coin, t=ro.index[k], R=R)); i += 1
            if not roll:
                break
            entry, k, first = ex, k_end, False
        i += 1
    return out


def run_pm() -> None:
    bh4, bd = TE.bars("BTCUSDT") if (DATA / "BTCUSDT_4h.csv.gz").exists() else TE.bars("BTC")
    rows = []
    for c in ("ETHUSDT", "SOLUSDT", "XRPUSDT") + TE.FRESH:
        rows += pair(c, bd, bh4)
    T = pd.DataFrame(rows); core = T[T.sym.isin(("ETHUSDT", "SOLUSDT", "XRPUSDT"))]
    des = core[core.t < TE.HELD_FROM]
    print(f"PM pair MAX-10 vs BTC: design {RL.summ(des.R)} | halves {RL.halves(des)} | held-out {RL.summ(core[core.t >= TE.HELD_FROM].R)} | fresh {RL.summ(T[T.sym.isin(TE.FRESH)].R)}")


if __name__ == "__main__":
    run_vb()
    run_pm()
