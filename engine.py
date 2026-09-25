"""Bitunix sleeve engine: data, regime, candidates and scoring for the scheduled runs.

Data comes from the public static-klines repo (Binance spot, 10 majors, raw GitHub files),
which the sandbox can reach when WebFetch summaries are unreliable.

Usage:
  python engine.py regime [SYMS]          market trend and each symbol's volatility state
  python engine.py candidates [SYMS]      mechanical 4h candidates with regime tags and flags
  python engine.py levels [SYMS]          tested supports, resistances and trendlines; stop 1.0 ATR beyond
  python engine.py edges [SYMS]           tested edges (journal entry 47): MAX-10 action and trend components
  python engine.py gap                    retired: CME bitcoin futures trade 24/7 since 29 May 2026
  python engine.py score TICKETS.csv      score split tickets on 15m bars, stop first
Ticket CSV columns: id,sym,side,placed_utc,valid_until_utc,entry,stop,tp1,tp2  (side 1 or -1)
"""
from __future__ import annotations

import http.client
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://github.com/finom/static-klines"
RAW = "https://raw.githubusercontent.com/finom/static-klines/main/"
CLONE = Path("/tmp/static-klines")
COVERED = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT")
GUARANTEED_STOP = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}  # Bitunix help centre, updated 29 Jan 2026
MAKER, TAKER = 0.0002, 0.0006


# ---------- data ----------
def _ctx() -> ssl.SSLContext:
    bundle = Path("/root/.ccr/ca-bundle.crt")
    return ssl.create_default_context(cafile=str(bundle)) if bundle.exists() else ssl.create_default_context()


def _get(url: str) -> bytes:
    for _ in range(6):
        try:
            return urllib.request.urlopen(url, context=_ctx(), timeout=120).read()
        except (http.client.IncompleteRead, ConnectionError, TimeoutError, OSError):
            continue
    raise RuntimeError(f"download failed: {url}")


@lru_cache(maxsize=1)
def _files() -> tuple[str, ...]:
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, str(CLONE)],
                       check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    out = subprocess.run(["git", "-C", str(CLONE), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True, check=True)
    return tuple(f for f in out.stdout.split() if f.startswith(".klines-cache/"))


def klines(sym: str, interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    """OHLC bars indexed by UTC open time. Files are windows named by their start date."""
    end = end or (pd.Timestamp.now('UTC') + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pre = f".klines-cache/{sym}/{interval}/"
    names = sorted(f for f in _files() if f.startswith(pre))
    starts = [f.rsplit("/", 1)[1][:10] for f in names]
    keep = [n for n, s, nxt in zip(names, starts, starts[1:] + ["9999"]) if nxt > start and s < end]
    rows: list = []
    for n in keep:
        rows += json.loads(_get(RAW + n))
    if not rows:
        return pd.DataFrame(columns=["o", "h", "l", "c"])
    d = pd.DataFrame([r[:5] for r in rows], columns=["t", "o", "h", "l", "c"]).astype(float)
    d.index = pd.to_datetime(d.t.astype("int64"), unit="ms")
    d = d[~d.index.duplicated()].sort_index()[["o", "h", "l", "c"]]
    return d[(d.index >= start) & (d.index < end)]


# ---------- indicators ----------
def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    x = c.diff()
    up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(c: pd.Series) -> pd.DataFrame:
    line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


# ---------- regime ----------
@lru_cache(maxsize=32)
def daily(sym: str) -> pd.DataFrame:
    start = (pd.Timestamp.now('UTC') - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    d = klines(sym, "1d", start)
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return d[d.index < today]  # completed days only


@lru_cache(maxsize=1)
def market_trend() -> str:
    d = daily("BTCUSDT"); e = d.c.ewm(span=50, adjust=False).mean()
    if d.c.iat[-1] > e.iat[-1] and e.iat[-1] > e.iat[-6]:
        return "up"
    if d.c.iat[-1] < e.iat[-1] and e.iat[-1] < e.iat[-6]:
        return "down"
    return "flat"


def vol_state(sym: str) -> tuple[str, float]:
    d = daily(sym); ap = atr(d) / d.c
    med = ap.rolling(180, min_periods=60).median()
    return ("high vol" if ap.iat[-1] > med.iat[-1] else "low vol"), float(ap.iat[-1])


def alignment(side: int, trend: str) -> str:
    if trend == "flat":
        return "market flat"
    return "with market" if (trend == "up") == (side == 1) else "against market"


# ---------- candidates ----------
def candidates(sym: str, trend: str) -> list[dict]:
    """Mechanical 4h setups at the last completed bar. The run applies judgement and the 3:1 floor."""
    start = (pd.Timestamp.now('UTC') - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    d = klines(sym, "4h", start)
    now = pd.Timestamp.now('UTC').tz_localize(None)
    d = d[d.index + pd.Timedelta(hours=4) <= now]
    a = atr(d); e20 = d.c.ewm(span=20, adjust=False).mean(); e50 = d.c.ewm(span=50, adjust=False).mean()
    r = rsi(d.c); m = macd(d.c)
    hh = d.h.rolling(20).max().shift(); ll = d.l.rolling(20).min().shift()
    lo10 = d.l.rolling(10).min(); hi10 = d.h.rolling(10).max()
    width = (hh - ll) / a; coil = width < width.rolling(200, min_periods=60).quantile(0.2)
    i = len(d) - 1; px = d.c.iat[i]; vol, atrp = vol_state(sym); out = []

    def row(setup: str, side: int, entry: float, stop: float, objective: float, order: str) -> dict:
        risk = abs(entry - stop); stop_pct = risk / entry * 100
        rr = abs(objective - entry) / risk if risk else np.nan
        al = alignment(side, trend); flags = []
        if stop_pct < 0.5: flags.append("stop under 0.5%: does not print")
        if sym not in GUARANTEED_STOP: flags.append("no guaranteed stop: gap risk beyond 1R")
        if al == "against market" and vol == "high vol": flags.append("against market in high vol: conviction down one grade")
        if setup == "failed move" and al == "market flat": flags.append("flat-market failed move: under test")
        if setup == "pullback" and vol == "high vol": flags.append("high-vol pullback: needs divergence or pattern confluence")
        return dict(sym=sym, setup=setup, side="long" if side == 1 else "short", order=order, entry=round(entry, 6),
                    stop=round(stop, 6), stop_pct=round(stop_pct, 2), objective=round(objective, 6), rr_gross=round(rr, 2),
                    price=round(px, 6), trend=trend, vol=vol, align=al, rsi=round(r.iat[i], 1),
                    macd="above signal" if m.macd.iat[i] > m.signal.iat[i] else "below signal", flags="; ".join(flags))

    for side in (1, -1):
        trending = (e20.iat[i] > e50.iat[i] and d.c.iat[i] > e50.iat[i]) if side == 1 else (e20.iat[i] < e50.iat[i] and d.c.iat[i] < e50.iat[i])
        if trending:  # resting limit at the 4h EMA20
            lvl = e20.iat[i]; stop = (lo10.iat[i] - 0.25 * a.iat[i]) if side == 1 else (hi10.iat[i] + 0.25 * a.iat[i])
            if (stop < lvl < px) if side == 1 else (stop > lvl > px):
                out.append(row("pullback", side, lvl, stop, d.h.iloc[-20:].max() if side == 1 else d.l.iloc[-20:].min(), "maker limit"))
        swept = (d.l.iat[i] < ll.iat[i] and d.c.iat[i] > ll.iat[i]) if side == 1 else (d.h.iat[i] > hh.iat[i] and d.c.iat[i] < hh.iat[i])
        if swept:  # reclaim of the 20-bar extreme, limit at the close
            stop = (d.l.iat[i] - 0.25 * a.iat[i]) if side == 1 else (d.h.iat[i] + 0.25 * a.iat[i])
            out.append(row("failed move", side, px, stop, hh.iat[i] if side == 1 else ll.iat[i], "maker limit"))
    if coil.iat[i]:
        out.append(row("bracket", 1, hh.iat[i], ll.iat[i] - 0.25 * a.iat[i], hh.iat[i] + 3 * (hh.iat[i] - ll.iat[i]), "stop entry"))
        out.append(row("bracket", -1, ll.iat[i], hh.iat[i] + 0.25 * a.iat[i], ll.iat[i] - 3 * (hh.iat[i] - ll.iat[i]), "stop entry"))
    return out


# ---------- tested levels and trendlines (tested-level method) ----------
STOP_BEYOND_ATR = 1.0  # journal entry 44: 0.25 ATR beyond tested levels lost -0.26R to -0.45R; 1.0 ATR cut that by about 0.2R


def _pivots(v: np.ndarray, side: int, k: int = 3) -> list[int]:
    return [i for i in range(k, len(v) - k) if v[i] == (v[i - k:i + k + 1].min() if side == 1 else v[i - k:i + k + 1].max())]


def levels(sym: str, trend: str) -> list[dict]:
    """Tested support/resistance zones (2+ pivot touches within 0.5 ATR in 120 4h bars, unbroken on closes)
    and the latest trendline. Entry at the level only; stop 1.0 ATR beyond the zone edge or the line."""
    start = (pd.Timestamp.now('UTC') - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    d = klines(sym, "4h", start)
    now = pd.Timestamp.now('UTC').tz_localize(None)
    d = d[d.index + pd.Timedelta(hours=4) <= now]
    a = atr(d).values; H, L, C = d.h.values, d.l.values, d.c.values; i = len(d) - 1; px = C[i]
    vol, _ = vol_state(sym); out = []
    zones: dict[int, list[tuple[float, float, int]]] = {}
    for side in (1, -1):
        vals = L if side == 1 else H
        pts = sorted((vals[p], p) for p in _pivots(vals, side) if i - 120 <= p <= i - 3)
        cl: list[list[tuple[float, int]]] = []
        for v, p in pts:
            if cl and abs(v - cl[-1][-1][0]) <= 0.5 * a[i]:
                cl[-1].append((v, p))
            else:
                cl.append([(v, p)])
        zs = []
        for c in cl:
            if len(c) < 2:
                continue
            lvl = float(np.mean([v for v, _ in c])); last = max(p for _, p in c)
            edge = min(v for v, _ in c) if side == 1 else max(v for v, _ in c)
            if (side == 1 and (C[last:i + 1] < edge).any()) or (side == -1 and (C[last:i + 1] > edge).any()):
                continue
            if (side == 1 and lvl < px) or (side == -1 and lvl > px):
                zs.append((lvl, edge, len(c)))
        zs.sort(key=lambda z: -z[0] * side)
        zones[side] = zs

    def row(kind: str, side: int, entry: float, stop: float, touches: int, nxt: float | None) -> dict:
        opp = zones[-side][0][0] if zones[-side] else np.nan
        risk = abs(entry - stop); al = alignment(side, trend); flags = []
        if sym not in GUARANTEED_STOP: flags.append("no guaranteed stop: gap risk beyond 1R")
        if al == "against market" and vol == "high vol": flags.append("against market in high vol: conviction down one grade")
        if "trendline" in kind: flags.append("trendline stop distance untested")
        return dict(sym=sym, kind=kind, side="long" if side == 1 else "short", entry=round(entry, 6), touches=touches,
                    stop=round(stop, 6), stop_pct=round(risk / entry * 100, 2), next_rung=None if nxt is None else round(nxt, 6),
                    objective_opposite_zone=round(opp, 6), rr_gross=round(abs(opp - entry) / risk, 2) if risk else np.nan,
                    price=round(px, 6), trend=trend, vol=vol, align=al, flags="; ".join(flags))

    for side in (1, -1):
        zs = zones[side]
        for n, (lvl, edge, touches) in enumerate(zs[:3]):
            nxt = zs[n + 1][0] if n + 1 < len(zs) else None
            out.append(row("support" if side == 1 else "resistance", side, lvl, edge - side * STOP_BEYOND_ATR * a[i], touches, nxt))
        vals = L if side == 1 else H
        ps = [p for p in _pivots(vals, side) if i - 120 <= p <= i - 3]
        if len(ps) >= 2:
            p1, p2 = ps[-2], ps[-1]
            rising = vals[p2] > vals[p1] if side == 1 else vals[p2] < vals[p1]
            if p2 - p1 >= 5 and rising:
                s = (vals[p2] - vals[p1]) / (p2 - p1); js = np.arange(p1, i + 1); line = vals[p1] + s * (js - p1)
                intact = not ((C[js] < line).any() if side == 1 else (C[js] > line).any())
                nxt_line = vals[p1] + s * (i + 1 - p1)
                if intact and ((side == 1 and nxt_line < px) or (side == -1 and nxt_line > px)):
                    out.append(row("rising trendline" if side == 1 else "falling trendline", side, nxt_line,
                                   nxt_line - side * STOP_BEYOND_ATR * a[i], 2, zs[0][0] if zs else None))
    return out


# ---------- weekend gap (paper) ----------
def _ct(day: pd.Timestamp, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hm}", tz="America/Chicago").tz_convert("UTC").tz_localize(None)


def weekend_gap() -> dict:
    """BTC gap between Friday 16:00 CT and Sunday 17:00 CT. Paper only (journal entry 44)."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    fri = (now - pd.Timedelta(days=(now.weekday() - 4) % 7)).normalize()
    if _ct(fri + pd.Timedelta(days=2), "17:00") > now:
        fri -= pd.Timedelta(days=7)
    d = klines("BTCUSDT", "1h", (fri - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    t0, t1 = _ct(fri, "15:00"), _ct(fri + pd.Timedelta(days=2), "17:00")
    if t0 not in d.index or t1 not in d.index:
        return {"status": "data missing"}
    a, b = d.c.at[t0], d.o.at[t1]; g = b / a - 1; side = int(np.sign(g)); risk = abs(b - a)
    return {"friday_close": a, "sunday_open": b, "gap_pct": round(g * 100, 2), "qualifies": bool(abs(g) >= 0.005),
            "paper_ticket": {"side": "long" if side == 1 else "short", "entry": b, "stop": a,
                             "tp1_1R": b + side * risk, "tp2_3R": b + side * 3 * risk}}


# ---------- tested edges (journal entry 47; paper from 28 September 2026) ----------
TREND_N = (5, 10, 20, 30, 60, 90, 150, 250, 360)
MIN_STOP_PCT = 0.5  # sleeve rail: stops under 0.5% do not print


@lru_cache(maxsize=32)
def daily_long(sym: str) -> pd.DataFrame:
    """Completed daily bars (00:00 UTC) over about four years, enough for the 360-day component."""
    start = (pd.Timestamp.now('UTC') - pd.Timedelta(days=1500)).strftime("%Y-%m-%d")
    d = klines(sym, "1d", start)
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    return d[d.index < today]


def max10(sym: str) -> dict:
    """MAX-10: long the morning after a daily close at or above the prior 10 closes; stop 1 daily ATR below
    entry; exit at the next morning's run unless the signal repeats (then roll as a fresh one-day ticket)."""
    d = daily_long(sym); c = d.c; a = atr(d); hi = c.rolling(10).max().shift(1)
    sig, held = bool(c.iat[-1] >= hi.iat[-1]), bool(c.iat[-2] >= hi.iat[-2])
    yday = pd.Timestamp.now('UTC').tz_localize(None).normalize() - pd.Timedelta(days=1)
    action = "roll" if sig and held else "buy" if sig else "close at this run" if held else "none"
    return dict(sym=sym, day=str(d.index[-1].date()), data_current=bool(d.index[-1] == yday), close=round(c.iat[-1], 6),
                prior_10d_high=round(hi.iat[-1], 6), action=action, atr=round(a.iat[-1], 6),
                stop_pct=round(a.iat[-1] / c.iat[-1] * 100, 2), guaranteed_stop=sym in GUARANTEED_STOP)


def trend_components(sym: str) -> list[dict]:
    """Multi-horizon trend, long only: each lookback N goes long on a close at or above the prior N closes,
    stop at the midpoint of the N-day closing range, trailed up after each close and never lowered."""
    d = daily_long(sym); c, lo = d.c.values, d.l.values; out = []
    for n in TREND_N:
        mx = pd.Series(c).rolling(n).max(); mn = pd.Series(c).rolling(n).min()
        prev_hi = mx.shift(1).values; mid = ((mx + mn) / 2).values
        long, stop, since = False, np.nan, None
        for t in range(n, len(c)):
            if long:
                if lo[t] <= stop:
                    long = False
                else:
                    stop = max(stop, mid[t])
            if not long and c[t] >= prev_hi[t] and (c[t] - mid[t]) / c[t] * 100 >= MIN_STOP_PCT:
                long, stop, since = True, mid[t], d.index[t]
        if len(c) < n + 2:
            state = "insufficient history"
        else:
            state = "new entry" if long and since == d.index[-1] else "long" if long else "flat"
        out.append(dict(sym=sym, lookback=n, state=state, since=str(since.date()) if long else "",
                        stop=round(stop, 6) if long else None,
                        stop_pct=round((c[-1] - stop) / c[-1] * 100, 2) if long else None))
    return out


# ---------- scoring ----------
def score(tickets: pd.DataFrame) -> pd.DataFrame:
    """Split tickets: half to tp1, half to tp2, one stop. Maker entry; stop first when a bar spans both."""
    res = []
    for t in tickets.itertuples():
        placed, until = pd.Timestamp(t.placed_utc), pd.Timestamp(t.valid_until_utc)
        if t.sym not in COVERED:
            res.append(dict(id=t.id, status="not covered: score from venue or web prices")); continue
        d = klines(t.sym, "15m", (placed - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        d = d[d.index >= placed]; side = int(t.side)
        filled = (d.l <= t.entry) if side == 1 else (d.h >= t.entry)
        fill = filled[d.index < until]
        if not fill.any():
            res.append(dict(id=t.id, status="expired unfilled" if d.index[-1] >= until else "resting", R=0.0)); continue
        j0 = d.index.get_loc(fill.idxmax()); risk = abs(t.entry - t.stop); legs, done = [], True
        for tp in (t.tp1, t.tp2):
            r = None
            for j in range(j0, len(d)):
                if (d.l.iat[j] <= t.stop) if side == 1 else (d.h.iat[j] >= t.stop):
                    r = -1 - (MAKER + TAKER) * t.entry / risk; break
                if j > j0 and ((d.h.iat[j] >= tp) if side == 1 else (d.l.iat[j] <= tp)):
                    r = abs(tp - t.entry) / risk - 2 * MAKER * t.entry / risk; break
            if r is None:
                done = False; r = side * (d.c.iat[-1] - t.entry) / risk
            legs.append(r)
        res.append(dict(id=t.id, status="closed" if done else "open", filled_at=str(d.index[j0]), R=round(0.5 * legs[0] + 0.5 * legs[1], 3)))
    return pd.DataFrame(res)


def main(argv: list[str]) -> None:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cmd = argv[1] if len(argv) > 1 else "regime"
    syms = argv[2].split(",") if len(argv) > 2 and cmd != "score" else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
    if cmd == "regime":
        trend = market_trend(); print(f"market trend (BTC daily vs EMA50): {trend}")
        for s in syms:
            v, ap = vol_state(s); print(f"  {s}: {v}, daily ATR {ap * 100:.2f}%")
    elif cmd == "candidates":
        trend = market_trend(); rows = [c for s in syms for c in candidates(s, trend)]
        print(pd.DataFrame(rows).to_string(index=False) if rows else "no mechanical candidates at the last completed 4h bar")
    elif cmd == "levels":
        trend = market_trend(); rows = [r for s in syms for r in levels(s, trend)]
        print(pd.DataFrame(rows).to_string(index=False) if rows else "no tested levels near price")
    elif cmd == "edges":
        core = syms if len(argv) > 2 else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        print("MAX-10 (paper; 1 daily ATR stop below entry; one bet split across coins signalling the same day)")
        print(pd.DataFrame([max10(s) for s in core]).to_string(index=False))
        rows = [r for s in core for r in trend_components(s)]
        t = pd.DataFrame(rows)
        print("\nTrend components (paper; long only; exit on the stop; stop trails up to the range midpoint daily)")
        print(t.to_string(index=False))
        live = t[t.state.isin(["long", "new entry"])]
        print("\nActive components per coin: " + ", ".join(f"{s} {int((live.sym == s).sum())}/{len(TREND_N)}" for s in core))
    elif cmd == "gap":
        print("retired (journal entry 47): CME bitcoin futures trade 24/7 since 29 May 2026, so the weekly gap no longer forms")
    elif cmd == "score":
        print(score(pd.read_csv(argv[2])).to_string(index=False))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
