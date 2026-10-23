"""Swing book for the 08:30 run, from 23 Oct 2026: $100 start, isolated margin, leverage read off the stop.

Books, in budget priority:
  MAX-10   crypto only (BTC, ETH, SOL, XRP tested; INJ untested). Long at the run after a 10-day closing high,
           stop 1 daily ATR, exit at the next run unless it repeats. 4% risk a row.
  TREND    20, 60 and 150-day closing highs, stop at the channel midpoint trailed daily (trend_subset.py rules),
           crypto, commodities, index and sector ETFs, 20 most traded Bitunix stocks. 2% risk a slice. New signals only.
  LEVELS   4h tested levels from engine.scan, crypto only, 2% a row, at most two open (10% of the 40% budget).
Caps: 40% open risk, 15% per correlated group. Evidence per class is printed with every row.

Usage: python swing.py [state.csv] [--equity 100]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E

EQUITY = 100.0
RISK = {"MAX10": 0.04, "TREND": 0.02, "LEVELS": 0.02}
CAP_TOTAL, CAP_GROUP, LEVELS_MAX = 0.40, 0.15, 2
LOOKBACKS = (20, 60, 150)
MAX10_TESTED = {"BTC", "ETH", "SOL", "XRP"}
CRYPTO = ("BTC", "ETH", "SOL", "XRP", "INJ")
COMMOD = {"XAU": "GC=F", "XAG": "SI=F", "XPT": "PL=F", "XPD": "PA=F", "CL": "CL=F", "BZ": "BZ=F", "COPPER": "HG=F",
          "NATGAS": "NG=F"}
ETFS = ("SPY", "QQQ", "IWM", "SMH", "XLE", "GDX", "XBI", "URNM", "EWJ", "EWT", "EWY", "EWZ")
STOCK_POOL = ("AAPL ADBE AMD AMZN ANET APP ARM ASML AVGO BABA CAT COIN CRM CSCO DELL GOOGL GS HD HOOD IBM INTC KLAC KO LLY "
              "LRCX META MRK MRVL MSFT MSTR MU NFLX NOW NVDA ORCL PLTR PYPL QCOM SHOP SMCI TSLA TSM TXN UBER V WMT").split()
GROUPS = {
    "crypto": set(CRYPTO) | {"MSTR", "COIN", "HOOD"},
    "precious metals": {"XAU", "XAG", "XPT", "XPD", "GDX"},
    "energy": {"CL", "BZ", "NATGAS", "XLE", "URNM"},
    "semis": {"SMH", "NVDA", "AMD", "AVGO", "MU", "TSM", "ASML", "KLAC", "LRCX", "MRVL", "QCOM", "TXN", "INTC", "ARM", "SMCI"},
    "US index and megacap": {"SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "ORCL", "PLTR"},
    "Asia ETFs": {"EWJ", "EWT", "EWY"},
}
EVIDENCE = {
    "MAX10": "beats placebo every set (+0.078R net, n 4740)",
    "TREND crypto": "passes gate, fails placebo: beta plus exit",
    "TREND etf": "ETF trend passes at Bitunix funding, fails placebo: beta",
    "TREND commodity": "untested on the futures; GLD and SLV tested inside the ETF set",
    "TREND stock": "single-stock trend fails placebo (survivorship): weakest class",
    "LEVELS": "levels alone tested negative; capped",
}


def group(sym: str) -> str:
    return next((g for g, s in GROUPS.items() if sym in s), "US stocks other")


def daily(sym: str) -> pd.DataFrame:
    """Completed daily bars: crypto from the live feed, everything else from the research feed (Yahoo)."""
    if sym in CRYPTO:
        return E.daily_done(sym)
    t = COMMOD.get(sym, sym)
    d = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/daily/{E.safe(t)}.csv.gz")), compression="gzip",
                    index_col=0, parse_dates=True)
    d = d[["o", "h", "l", "c", "v"]].astype(float).dropna(subset=["c"])
    if f"{E.safe(t)}_1d" in E.manifest().get("stocks", []):  # fresher cash-session bars from the live feed
        lv = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}live/stocks/{E.safe(t)}_1d.csv")), index_col=0, parse_dates=True)
        lv = lv[["o", "h", "l", "c", "v"]].astype(float); lv.index = lv.index.normalize()
        d = pd.concat([d[d.index < lv.index[0]], lv])
        now = pd.Timestamp.now("UTC").tz_localize(None)
        if d.index[-1].date() == now.date() and now.hour < 21:
            d = d.iloc[:-1]  # today's US session not closed yet
    return d


def top_stocks(n: int = 20) -> list[str]:
    """The n Bitunix-listed stocks with the highest 60-day median traded value in the feed."""
    val = {}
    for s in STOCK_POOL:
        try:
            d = daily(s); val[s] = float((d.c * d.v).iloc[-60:].median())
        except Exception:
            continue
    return [s for s, _ in sorted(val.items(), key=lambda x: -x[1])[:n]]


def trend(d: pd.DataFrame, n: int) -> dict:
    """trend_subset.py rule: long on a close at the prior n-day closing high at least 0.5% above the channel
    midpoint; stop at the midpoint, trailed up daily; exit when a low trades through it."""
    c, lo = d.c.values, d.l.values
    mx, mn = pd.Series(c).rolling(n).max(), pd.Series(c).rolling(n).min(); prev = mx.shift(1).values
    mid = ((mx + mn) / 2).values; long, stop, since, entry = False, np.nan, None, np.nan
    for t in range(n, len(c)):
        if long:
            if lo[t] <= stop:
                long = False
            else:
                stop = max(stop, mid[t])
        if not long and c[t] >= prev[t] and (c[t] - mid[t]) / c[t] >= 0.005:
            long, stop, since, entry = True, mid[t], d.index[t], c[t]
    if len(c) < n + 2:
        return dict(state="insufficient history")
    state = "new" if long and since == d.index[-1] else "long" if long else "flat"
    return dict(state=state, since=str(since.date()) if long else "", entry=entry, stop=stop if long else np.nan)


def size(sym: str, entry: float, stop: float, risk: float, equity: float) -> dict:
    E.RISK = risk
    t = E.ticket(sym, 1, entry, stop, entry * 10, equity)
    return dict(entry=E._fmt(entry), stop=E._fmt(stop), stop_pct=t["stop_pct"], lev=t["lev"],
                notional=t["notional"], margin=t["margin"])


def open_risk(path: str | None) -> tuple[float, dict[str, float], int]:
    """Risk already open on live rows in the state file, by group, and the number of open LEVELS rows."""
    if not path or not Path(path).exists():
        return 0.0, {}, 0
    S = pd.read_csv(path, dtype=str)
    S = S[S.status.isin(["open", "tp1"]) & S.live.astype(str).str.lower().isin(["y", "yes"])]
    by: dict[str, float] = {}; total = 0.0
    for _, r in S.iterrows():
        book = str(r.setup).split()[0].upper()
        rk = RISK.get(book, RISK["LEVELS"]); by[group(r.sym)] = by.get(group(r.sym), 0) + rk; total += rk
    lv = int(sum(not str(x).upper().startswith(("MAX10", "TREND")) for x in S.setup))
    return total, by, lv


def main(argv: list[str]) -> None:
    eq = float(argv[argv.index("--equity") + 1]) if "--equity" in argv else EQUITY
    state = next((a for a in argv[1:] if a.endswith(".csv")), None)
    print(f"DATA: crypto feed updated {E.manifest().get('updated_utc', 'unknown')} UTC ({E.feed_age_hours():.1f}h old); "
          f"tradfi daily from the research feed, updated {E.manifest().get('research_updated_utc', 'unknown')} UTC")
    print(f"REGIME: crypto {E.market_trend('crypto')} (BTC daily vs EMA50)")
    stocks = top_stocks()
    universe = [(s, "crypto") for s in CRYPTO] + [(s, "commodity") for s in COMMOD] + [(s, "etf") for s in ETFS] \
        + [(s, "stock") for s in stocks]
    cands, active = [], []
    print("MAX-10 (at the 00:00 UTC daily close):")
    for s in CRYPTO:
        try:
            m = E.max10(s)
        except Exception as e:
            print(f"  {s}: MAX-10 unavailable ({type(e).__name__})"); continue
        print(f"  MAX-10 {s}: {m['action']} (close {m['close']} vs prior 10-day high {m['prior_10d_high']})")
        if m["action"] in ("buy", "roll"):
            row = dict(sym=s, book="MAX10", rule=f"MAX-10 {m['action']}", cls="crypto", prio=0,
                       note=EVIDENCE["MAX10"] + ("" if s in MAX10_TESTED else "; INJ untested for MAX-10"),
                       **size(s, m["price"], m["stop"], RISK["MAX10"], eq))
            (cands if m["action"] == "buy" else active).append(row)
    for s, cls in universe:
        try:
            d = daily(s)
        except Exception as e:
            print(f"  {s}: data unavailable ({type(e).__name__})"); continue
        live = cls == "crypto" or f"{E.safe(s)}_1h" in E.manifest().get("stocks", [])
        px = E.price(s)[0] if live else float(d.c.iat[-1])
        asof = "" if live else f"; last close {d.index[-1].date()}, quote the current price before placing"
        comps = {n: trend(d, n) for n in LOOKBACKS}; nlong = sum(c["state"] in ("new", "long") for c in comps.values())
        for n, c in comps.items():
            if c["state"] not in ("new", "long") or not np.isfinite(c["stop"]) or c["stop"] >= px:
                continue
            decayed = c["state"] == "new" and (px - c["stop"]) < 0.5 * (c["entry"] - c["stop"])
            row = dict(sym=s, book="TREND", rule=f"trend {n}d since {c['since']}", cls=cls, prio=4 - nlong + 9 * decayed,
                       note=EVIDENCE[f"TREND {cls}"] + ("" if s in E.GUARANTEED_STOP else "; no guaranteed stop") + asof
                       + ("; DECAYED: price has given back over half the signal's room, skip" if decayed else ""),
                       **size(s, px, c["stop"], RISK["TREND"], eq))
            (cands if c["state"] == "new" else active).append(row)
    total, by, lv = open_risk(state)
    print(f"OPEN RISK on live rows: {total * 100:.0f}% of {CAP_TOTAL * 100:.0f}%"
          + ("; " + ", ".join(f"{g} {v * 100:.0f}%" for g, v in by.items()) if by else ""))
    print("NEW SIGNALS, in budget order (MAX-10 first, then trend by lookbacks agreeing). Market entry at the run, isolated:")
    taken = 0
    for r in sorted(cands, key=lambda r: (r["prio"], r["stop_pct"])):
        rk, g = RISK[r["book"]], group(r["sym"])
        fits = "DECAYED" not in r["note"] and total + rk <= CAP_TOTAL + 1e-9 and by.get(g, 0) + rk <= CAP_GROUP + 1e-9
        if fits:
            total += rk; by[g] = by.get(g, 0) + rk; taken += 1
        print(f"  {'TICKET' if fits else 'OVER CAP'} {r['sym']} long, {r['rule']}: entry {r['entry']} stop {r['stop']} "
              f"({r['stop_pct']}%), {r['lev']}x, notional {r['notional']}, margin {r['margin']}, risk {rk * 100:.0f}%, "
              f"group {g}; {r['note']}")
    if not cands:
        print("  none today")
    print(f"LEVELS budget: {max(LEVELS_MAX - lv, 0)} of {LEVELS_MAX} rows free at {RISK['LEVELS'] * 100:.0f}% "
          "(take from python engine.py scan --crypto-only --equity, sized here)")
    print("ACTIVE COMPONENTS (signal earlier; trail the stop if held, do not join late):")
    for r in sorted(active, key=lambda r: (r["cls"], r["sym"])):
        print(f"  {r['sym']} {r['rule']}: today's stop {r['stop']} ({r['stop_pct']}% below {r['entry']})")
    print(f"TOP 20 STOCKS by 60-day median traded value: {', '.join(stocks)}")


if __name__ == "__main__":
    main(sys.argv)
