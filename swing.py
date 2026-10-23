"""Swing book for the 08:30 run, from 23 Oct 2026: $100 start, isolated margin, leverage read off the stop.

Books, in budget priority:
  MAX-10   crypto only (BTC, ETH, SOL, XRP tested; INJ untested). Long at the run after a daily close at or above
           the prior ten closes, stop 1 daily ATR below that close. Held as a swing (swing_batch2.py B3, from 23 Oct
           2026): exit at the run after a close below the lowest of the prior three closes, or after ten closes.
           4% risk a row.
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
    "MAX10": "swing hold (SW10): +0.40R a trade net (t 5.6, n 1601), beats its placebo in every set",
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


def held_max10(path: str | None) -> dict[str, dict]:
    """Live, open MAX-10 rows in the state file, by coin: signal day (note sig=, else the day before the run) and stop."""
    if not path or not Path(path).exists():
        return {}
    S = pd.read_csv(path, dtype=str); out = {}
    S = S[S.status.isin(["open", "tp1"]) & S.live.astype(str).str.lower().isin(["y", "yes"]) & S.setup.astype(str).str.upper().str.startswith("MAX10")]
    for _, r in S.iterrows():
        kv = dict(x.strip().split("=", 1) for x in str(r.note).split(";") if "=" in x)
        sig = kv.get("sig") or str((pd.Timestamp(r.run) - pd.Timedelta(days=1)).date())
        out[str(r.sym)] = dict(signal_day=sig, stop=r.stop, entry=r.entry, id=r.id)
    return out


def max10_swing(sym: str, held: dict | None) -> dict:
    """Today's MAX-10 swing action for one coin from completed daily closes."""
    d = E.daily_done(sym); c = d.c; a = E.atr(d); prior = float(c.iloc[-11:-1].max())
    out = dict(sym=sym, day=str(d.index[-1].date()), close=E._fmt(c.iat[-1]), prior_high=E._fmt(prior))
    if held:
        n_after = int((d.index > pd.Timestamp(held["signal_day"])).sum()); low3 = float(c.iloc[-4:-1].min())
        if c.iat[-1] < low3:
            out["action"] = f"EXIT at market: close {E._fmt(c.iat[-1])} below the prior three closes ({E._fmt(low3)})"
        elif n_after >= 10:
            out["action"] = "EXIT at market: ten closes since entry"
        else:
            out["action"] = f"HOLD ({n_after} of 10 closes), stop {held['stop']} unchanged"
    elif c.iat[-1] >= prior:
        out["action"] = "BUY"; out["stop"] = float(c.iat[-1] - a.iat[-1])
    else:
        out["action"] = "none"
    return out


def max10_swing_record(since: str = "2026-09-28", coins: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP")) -> pd.DataFrame:
    """Mechanical paper record of the swing rule on daily closes since `since` (entry and exit at the 00:00 UTC close)."""
    rows = []
    for s in coins:
        d = E.daily_done(s); c, lo, o = d.c.values, d.l.values, d.o.values; a = E.atr(d).values
        sig = (d.c >= d.c.rolling(10).max().shift(1)).values; k = max(20, int(d.index.searchsorted(pd.Timestamp(since))) - 1)
        while k < len(d) - 1:
            if not sig[k]:
                k += 1; continue
            e, st = c[k], c[k] - a[k]; j = k + 1; ex = None
            while j < len(d):
                if o[j] <= st or lo[j] <= st:
                    ex = min(st, o[j]); break
                if c[j] < c[j - 3:j].min() or j >= k + 10:
                    ex = c[j]; break
                j += 1
            if ex is None:
                rows.append(dict(sym=s, entry_day=str(d.index[k].date()), R=np.nan, status="open")); break
            rows.append(dict(sym=s, entry_day=str(d.index[k].date()), exit_day=str(d.index[j].date()),
                             R=round((ex - e) / a[k] - (2 * 0.0008 + 0.0003 * (j - k)) * e / a[k], 3), status="closed"))
            k = j + 1
    return pd.DataFrame(rows)


def open_risk(path: str | None) -> tuple[float, dict[str, float], int]:
    """Risk already open on live rows in the state file, by group, and the number of open LEVELS rows."""
    if not path or not Path(path).exists():
        return 0.0, {}, 0
    S = pd.read_csv(path, dtype=str)
    S = S[S.status.isin(["open", "tp1"]) & S.live.astype(str).str.lower().isin(["y", "yes"])]
    by: dict[str, float] = {}; total = 0.0
    for _, r in S.iterrows():
        book = str(r.setup).split()[0].upper()
        kv = dict(x.strip().split("=", 1) for x in str(r.note).split(";") if "=" in x)
        rk = float(kv["risk"]) if "risk" in kv else RISK.get(book, RISK["LEVELS"])  # own rows carry their real risk
        by[group(r.sym)] = by.get(group(r.sym), 0) + rk; total += rk
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
    held = held_max10(state)
    print("MAX-10 SWING (at the 00:00 UTC daily close; exits and holds apply to live rows in the state file):")
    for s in CRYPTO:
        try:
            m = max10_swing(s, held.get(s))
        except Exception as e:
            print(f"  {s}: MAX-10 unavailable ({type(e).__name__})"); continue
        print(f"  MAX-10 {s}: {m['action']} (close {m['close']} on {m['day']} vs prior ten-close high {m['prior_high']})")
        if m["action"] == "BUY":
            px = E.price(s)[0]
            if px <= m["stop"]:
                print(f"    skipped: price {E._fmt(px)} already at or below the stop"); continue
            row = dict(sym=s, book="MAX10", rule=f"MAX-10 swing, signal close {m['day']}", cls="crypto", prio=0,
                       note=EVIDENCE["MAX10"] + ("" if s in MAX10_TESTED else "; INJ untested for MAX-10")
                       + f"; log note sig={m['day']}", **size(s, px, m["stop"], RISK["MAX10"], eq))
            cands.append(row)
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
    rec = max10_swing_record()
    done = rec[rec.status == "closed"] if len(rec) else rec
    print("MAX-10 SWING PAPER since 28 Sep (daily closes, core four): "
          + (f"{len(done)} closed, mean {done.R.mean():+.2f}R, total {done.R.sum():+.2f}R" if len(done) else "none closed")
          + (f"; {int((rec.status == 'open').sum())} open" if len(rec) else ""))


if __name__ == "__main__":
    main(sys.argv)
