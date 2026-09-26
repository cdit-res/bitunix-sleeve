"""Batch 3b, pre-registered 25 Sep 2026 before running.

C1: pre-FOMC drift (Lucca and Moench, 2015): long from 14:00 ET the day before an FOMC statement to 13:55 ET on
    the statement day, taker both ways. Stocks: SPXL and TQQQ 5m (2020-07 to 2026-02), costs on the underlying
    (move / leverage factor) at stock-perp rates; design to 2023, held out 2024 on. BTC 5m (Bitstamp), 2017-2026,
    design to 2022, held out 2023 on. Net % of the underlying per event.
C2: buying dips in uptrends (Connors RSI-2), daily bars: long at the next open (maker) when the close is above its
    200-day average and RSI(2) is below 10; exit at the next open after a close above the 5-day average, or after 10
    days; stop 1.5 x daily ATR(14) so R is defined. Stocks: TQQQ, SPXL, SOXL design to 2023, held out 2024 on;
    fresh = the 2x and sector 3x ETFs (FAS, LABU, DPST, NAIL, DRN, RETL, CURE, NVDX, TSLT, METU, AMZU, GGLL, AMUU,
    ROBN). Crypto: BTC, ETH, SOL, XRP design to 2023, held out 2024 on; six fresh coins. Crypto longs pay funding
    0.03% a day. Two variants counted (stocks, crypto).
C5: MAX-10 and trend long on stocks with the extra sector and single-stock ETFs as fresh sets (repeat of K05, K06
    with more power; not new variants).

Usage: STOCKS=/tmp/pmd/data SLEEVE_DATA=/path/to/data python batch3b.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MIN_STOP", "0.005")
import remaining_lines as RL  # noqa: E402
import stocks_intraday as SI  # noqa: E402
import trend_ensemble as TE  # noqa: E402

EXTRA = {"FAS": 3, "LABU": 3, "DPST": 3, "NAIL": 3, "DRN": 3, "RETL": 3, "CURE": 3, "AMUU": 2, "ROBN": 2}
SI.FACTOR.update(EXTRA)
STOCK_FRESH = SI.FRESH + tuple(EXTRA)


def prefomc_stocks(bars: dict) -> None:
    ev = pd.read_csv(TE.DATA / "macro_events.csv"); ev = ev[ev.event.str.startswith("FOMC")]
    rows = []
    for sym in ("SPXL", "TQQQ"):
        b = bars[sym]; L = SI.FACTOR[sym]
        for d in pd.to_datetime(ev.date):
            t0 = pd.Timestamp(f"{(d - pd.offsets.BDay(1)).date()} 14:00"); t1 = pd.Timestamp(f"{d.date()} 13:55")
            if t0 not in b.index or t1 not in b.index:
                continue
            r = (b.c.at[t1] / b.o.at[t0] - 1) / L
            rows.append(dict(sym=sym, t=d, net=(r - 2 * (SI.TAKER + SI.SLIP)) * 100))
    T = pd.DataFrame(rows); des = T[T.t < "2024-01-01"]; ho = T[T.t >= "2024-01-01"]
    print(f"C1 pre-FOMC drift, stocks (net % of underlying per event): design {RL.summ(des.net)} | held-out {RL.summ(ho.net)}")


def prefomc_btc() -> None:
    ev = pd.read_csv(TE.DATA / "macro_events.csv"); ev = ev[ev.event.str.startswith("FOMC")]
    m5 = pd.read_parquet(TE.DATA / "BTC_bitstamp_5m.parquet"); rows = []
    for d, tm in zip(pd.to_datetime(ev.date), ev.time_et):
        t1 = pd.Timestamp(f"{d.date()} {tm}", tz="America/New_York").tz_convert("UTC").tz_localize(None) - pd.Timedelta(minutes=5)
        t0 = t1 - pd.Timedelta(hours=24) + pd.Timedelta(minutes=5)
        if t0 not in m5.index or t1 not in m5.index:
            continue
        r = m5.c.at[t1] / m5.o.at[t0] - 1
        rows.append(dict(t=d, net=(r - 2 * (TE.TAKER + TE.SLIP) - TE.FUND) * 100))
    T = pd.DataFrame(rows); des = T[T.t < "2023-01-01"]; ho = T[T.t >= "2023-01-01"]
    print(f"C1 pre-FOMC drift, BTC (net % per event): design {RL.summ(des.net)} | held-out {RL.summ(ho.net)}")


def rsi2_trades(sym: str, d: pd.DataFrame, factor: float, fund: float) -> list[dict]:
    c, o, lo = d.c.values, d.o.values, d.l.values
    x = pd.Series(c).diff(); up = x.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).values
    sma200 = pd.Series(c).rolling(200).mean().values; sma5 = pd.Series(c).rolling(5).mean().values
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().values; out = []; i = 201
    while i < len(c) - 2:
        if not (c[i] > sma200[i] and rsi[i] < 10):
            i += 1; continue
        k = i + 1; entry = o[k]; stop = entry - 1.5 * atr[i]; risk = entry - stop; stop_u = risk / entry / factor
        ex = None; j = k
        for j in range(k, min(k + 10, len(c) - 1)):
            if lo[j] <= stop:
                ex = min(stop, o[j]) * (1 - SI.SLIP * factor); break
            if c[j] > sma5[j]:
                ex = o[j + 1] * (1 - SI.SLIP * factor); j += 1; break
        if ex is None:
            ex = o[j + 1] * (1 - SI.SLIP * factor); j += 1
        days = j - k + 1
        R = (ex - entry) / risk - (SI.MAKER + SI.TAKER) / stop_u - fund * days / stop_u
        out.append(dict(sym=sym, t=d.index[k], R=R)); i = j + 1
    return out


def run_rsi2(daily: dict) -> None:
    rows = [r for s, d in daily.items() for r in rsi2_trades(s, d, SI.FACTOR[s], SI.FUND_NIGHT)]
    T = pd.DataFrame(rows); core = T[T.sym.isin(SI.CORE)]; des = core[core.t < SI.HELD]
    print(f"C2 RSI-2 dips in uptrends, stocks: design {RL.summ(des.R)} | halves {RL.halves(des, 'R')} | held-out {RL.summ(core[core.t >= SI.HELD].R)} | fresh {RL.summ(T[T.sym.isin(STOCK_FRESH)].R)}")
    crows = []
    for s in TE.CORE + TE.FRESH:
        _, d = TE.bars(s); crows += rsi2_trades(s, d, 1.0, TE.FUND)
    C = pd.DataFrame(crows); core = C[C.sym.isin(TE.CORE)]; des = core[core.t < TE.HELD_FROM]
    print(f"C2 RSI-2 dips in uptrends, crypto: design {RL.summ(des.R)} | halves {RL.halves(des, 'R')} | held-out {RL.summ(core[core.t >= TE.HELD_FROM].R)} | fresh {RL.summ(C[C.sym.isin(TE.FRESH)].R)}")


def run_c5(daily: dict) -> None:
    for name, fn in (("MAX-10", lambda s, d: SI.max10(s, d)), ("trend long", lambda s, d: SI.trend(s, d, 1))):
        T = pd.DataFrame([r for s, d in daily.items() for r in fn(s, d)])
        core = T[T.sym.isin(SI.CORE)]; des = core[core.t < SI.HELD]
        print(f"C5 {name} on stocks, extended fresh set: design {RL.summ(des.R)} | held-out {RL.summ(core[core.t >= SI.HELD].R)} | fresh {RL.summ(T[T.sym.isin(STOCK_FRESH)].R)}")


if __name__ == "__main__":
    bars = {s: SI.load(s) for s in SI.FACTOR}
    daily = {s: SI.daily(b) for s, b in bars.items()}
    prefomc_stocks(bars); prefomc_btc(); run_rsi2(daily); run_c5(daily)
