"""Bitunix sleeve engine: data, regime, candidates, tickets, scoring and the brief for the scheduled runs.

Data: the repo's own feed (data branch, refreshed hourly on GitHub Actions from Binance spot and Yahoo), with the
public static-klines repo as a fallback for crypto. Never the Bitunix API.

Usage:
  python engine.py refresh                          ask the feed for fresh data and wait up to two minutes
  python engine.py scan [--equity 170] [--crypto-only]   regime, prices, tested edges, every candidate with its ticket
  python engine.py score state.csv                  rescore every open row in place; print the book and the record
  python engine.py edges                            MAX-10 and trend components today, and the paper record since 28 Sep
  python engine.py verdict state.csv "K1=TAKE:reason" "K2=PASS:reason"   log judged candidates from the last scan
  python engine.py html brief.json                  the brief as HTML (brief.html) and plain text (brief.txt)
State CSV columns: id,run,sym,side,verdict,setup,entry,stop,tp1,tp2,valid_until,status,filled_at,closed_at,R,live,note
  R holds the open R for open rows and the realised R once a row resolves; only resolved rows enter the record.
  side long or short; verdict TAKE, PASS or OWN; status resting, open, tp1, closed, stopped, expired or cancelled.
"""
from __future__ import annotations

import http.client
import io
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

FEED = "https://raw.githubusercontent.com/cdit-res/bitunix-sleeve/data/"
SKL_REPO, SKL_RAW = "https://github.com/finom/static-klines", "https://raw.githubusercontent.com/finom/static-klines/main/"
CLONE = Path("/tmp/static-klines")
CRYPTO = ("BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT", "LTC", "NEAR", "INJ", "TAO", "FET",
          "RUNE", "SUI", "HYPE")
SCAN_CRYPTO = ("BTC", "ETH", "SOL", "XRP", "LINK", "NEAR", "INJ", "LTC")
SCAN_STOCKS = ("QQQ", "SPY", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "AVGO", "MSTR", "COIN", "PLTR",
               "MU", "TSM", "HOOD", "SMH")
EDGE_COINS = ("BTC", "ETH", "SOL", "XRP")
GUARANTEED_STOP = {"BTC", "ETH", "SOL", "XRP"}  # Bitunix help centre, updated 29 Jan 2026
GROUPS = {"alt beta": {"NEAR", "INJ", "RUNE", "LINK", "SOL", "XRP", "LTC"}, "AI": {"TAO", "FET"},
          "Nasdaq": {"QQQ", "TQQQ", "SQQQ"}, "semis": {"SMH", "SOXL", "SOXS", "NVDA", "AMD", "AVGO", "MU", "TSM"},
          "BTC": {"BTC", "MSTR", "COIN"}}
MAKER, TAKER, FEE_RT = 0.0002, 0.0006, 0.08  # FEE_RT in percent, maker entry and exit
RISK, LEV_CAP, MARGIN_HIT = 0.05, 80, 55.0
FLOOR = 1.6  # owner, 25 Sep 2026: 1.6:1 net or better to TP A; TP B reaches for the bigger dated move
EDGES_SINCE = "2026-09-28"
TREND_N = (5, 10, 20, 30, 60, 90, 150, 250, 360)
STOP_BEYOND_ATR = 1.0  # journal entry 44


# ---------- data ----------
def _ctx() -> ssl.SSLContext:
    bundle = Path("/root/.ccr/ca-bundle.crt")
    return ssl.create_default_context(cafile=str(bundle)) if bundle.exists() else ssl.create_default_context()


def _get(url: str) -> bytes:
    for _ in range(5):
        try:
            return urllib.request.urlopen(url, context=_ctx(), timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
        except (http.client.IncompleteRead, ConnectionError, TimeoutError, OSError):
            continue
    raise RuntimeError(f"download failed: {url}")


@lru_cache(maxsize=1)
def feed_root() -> str:
    """The data branch pinned to its current commit, so every file comes from one snapshot and no CDN copy is stale."""
    try:
        out = subprocess.run(["git", "ls-remote", "https://github.com/cdit-res/bitunix-sleeve", "data"], capture_output=True,
                             text=True, timeout=60).stdout.split()
        if out:
            return f"https://raw.githubusercontent.com/cdit-res/bitunix-sleeve/{out[0]}/"
    except Exception:
        pass
    return FEED


@lru_cache(maxsize=1)
def manifest() -> dict:
    try:
        return json.loads(_get(feed_root() + "manifest.json"))
    except Exception:
        return {}


def refresh(wait_s: int = 110) -> None:
    """Ask the feed for fresh data by pushing an empty commit to the `refresh` branch, then wait for it to publish."""
    import time
    before = feed_root()
    cmd = ("rm -rf /tmp/rf && git clone -q --depth 1 https://github.com/cdit-res/bitunix-sleeve /tmp/rf && cd /tmp/rf && "
           "git -c user.name=run -c user.email=run@users.noreply.github.com commit -q --allow-empty -m refresh && "
           "git push -qf origin HEAD:refresh")
    if subprocess.run(cmd, shell=True, capture_output=True).returncode != 0:
        print(f"refresh: could not trigger the feed from here; it was last updated {manifest().get('updated_utc', 'unknown')} UTC")
        return
    for _ in range(wait_s // 10):
        time.sleep(10)
        feed_root.cache_clear()
        if feed_root() != before:
            manifest.cache_clear(); bars.cache_clear()
            print(f"refresh: feed updated {manifest().get('updated_utc', 'unknown')} UTC"); return
    print(f"refresh: no update within {wait_s} s; the feed was last updated {manifest().get('updated_utc', 'unknown')} UTC")


def feed_age_hours() -> float:
    m = manifest()
    if not m.get("updated_utc"):
        return float("inf")
    return (pd.Timestamp.now("UTC") - pd.Timestamp(m["updated_utc"])).total_seconds() / 3600


def safe(t: str) -> str:
    return t.replace("^", "IDX_").replace("=", "_").replace(".", "_")


def base(sym: str) -> str:
    return sym.upper().replace("USDT", "").replace("-USD", "")


def is_crypto(sym: str) -> bool:
    return base(sym) in CRYPTO


@lru_cache(maxsize=1)
def _skl_files() -> tuple[str, ...]:
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", SKL_REPO, str(CLONE)],
                       check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    out = subprocess.run(["git", "-C", str(CLONE), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True, check=True)
    return tuple(f for f in out.stdout.split() if f.startswith(".klines-cache/"))


def static_klines(sym: str, interval: str, start: str) -> pd.DataFrame:
    pre = f".klines-cache/{base(sym)}USDT/{interval}/"
    names = sorted(f for f in _skl_files() if f.startswith(pre)); starts = [f.rsplit("/", 1)[1][:10] for f in names]
    keep = [n for n, nxt in zip(names, starts[1:] + ["9999"]) if nxt > start]
    rows = [r for n in keep for r in json.loads(_get(SKL_RAW + n))]
    d = pd.DataFrame([r[:6] + [r[9]] for r in rows], columns=["t", "o", "h", "l", "c", "v", "taker_buy_v"]).astype(float)
    d.index = pd.to_datetime(d.t.astype("int64"), unit="ms")
    return d[~d.index.duplicated()].sort_index().drop(columns="t")


@lru_cache(maxsize=256)
def bars(sym: str, interval: str) -> pd.DataFrame:
    """OHLC indexed by UTC bar open. Crypto 15m, 1h, 4h, 1d; stocks 1h and 1d (US cash session)."""
    if is_crypto(sym):
        try:
            d = pd.read_csv(io.BytesIO(_get(f"{feed_root()}live/crypto/{base(sym)}_{interval}.csv")), index_col=0, parse_dates=True)
        except Exception:
            days = {"15m": 20, "1h": 200, "4h": 400, "1d": 1600}[interval]
            d = static_klines(sym, interval, (pd.Timestamp.now("UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d"))
    else:
        d = pd.read_csv(io.BytesIO(_get(f"{feed_root()}live/stocks/{safe(sym)}_{interval}.csv")), index_col=0, parse_dates=True)
    return d.astype(float)


def completed(d: pd.DataFrame, interval: str) -> pd.DataFrame:
    step = pd.Timedelta({"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[interval])
    return d[d.index + step <= pd.Timestamp.now("UTC").tz_localize(None)]


def price(sym: str) -> tuple[float, pd.Timestamp]:
    d = bars(sym, "15m" if is_crypto(sym) else "1h")
    return float(d.c.iat[-1]), d.index[-1]


# ---------- indicators ----------
def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    x = c.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd_state(c: pd.Series) -> str:
    line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean(); sig = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return f"{'above' if line.iat[-1] > sig.iat[-1] else 'below'} signal, hist {'rising' if hist.iat[-1] > hist.iat[-2] else 'falling'}"


def _pivots(v: np.ndarray, side: int, k: int = 3) -> list[int]:
    return [i for i in range(k, len(v) - k) if v[i] == (v[i - k:i + k + 1].min() if side == 1 else v[i - k:i + k + 1].max())]


def divergence(d: pd.DataFrame, side: int) -> str:
    """Regular or hidden divergence on the last two pivots (lows for a long, highs for a short) in 60 bars."""
    r = rsi(d.c).values; v = (d.l if side == 1 else d.h).values; n = len(v)
    ps = [p for p in _pivots(v, side) if p >= n - 60]
    if len(ps) < 2:
        return "none"
    a, b = ps[-2], ps[-1]
    if side == 1:
        if v[b] < v[a] and r[b] > r[a]: return "regular bullish"
        if v[b] > v[a] and r[b] < r[a]: return "hidden bullish"
        if v[b] < v[a] and r[b] < r[a]: return "none"
    else:
        if v[b] > v[a] and r[b] < r[a]: return "regular bearish"
        if v[b] < v[a] and r[b] > r[a]: return "hidden bearish"
    return "none"


# ---------- regime ----------
def trend_of(d: pd.DataFrame) -> str:
    e = d.c.ewm(span=50, adjust=False).mean()
    if d.c.iat[-1] > e.iat[-1] and e.iat[-1] > e.iat[-6]: return "up"
    if d.c.iat[-1] < e.iat[-1] and e.iat[-1] < e.iat[-6]: return "down"
    return "flat"


@lru_cache(maxsize=4)
def market_trend(kind: str = "crypto") -> str:
    return trend_of(completed(bars("BTC", "1d"), "1d") if kind == "crypto" else bars("QQQ", "1d"))


def vol_state(sym: str) -> tuple[str, float]:
    d = completed(bars(sym, "1d"), "1d") if is_crypto(sym) else bars(sym, "1d"); ap = atr(d) / d.c
    med = ap.rolling(180, min_periods=60).median()
    return ("high vol" if ap.iat[-1] > med.iat[-1] else "low vol"), float(ap.iat[-1])


def alignment(side: int, trend: str) -> str:
    return "market flat" if trend == "flat" else ("with market" if (trend == "up") == (side == 1) else "against market")


def chasing(sym: str, side: int) -> bool:
    d = completed(bars(sym, "1d"), "1d") if is_crypto(sym) else bars(sym, "1d")
    e20 = d.c.ewm(span=20, adjust=False).mean(); a = atr(d)
    stretch = side * (d.c.iat[-1] - e20.iat[-1]) / a.iat[-1]; run5 = side * (d.c.iat[-1] / d.c.iat[-6] - 1)
    return bool(stretch > 2 or run5 > 0.10)


# ---------- tickets ----------
def ticket(sym: str, side: int, entry: float, stop: float, tp2: float, equity: float, tp1: float | None = None) -> dict:
    risk = abs(entry - stop); stop_pct = risk / entry * 100
    if tp1 is None:
        tp1 = entry + side * 1.5 * risk if side * (tp2 - entry) > 1.5 * risk else tp2
    lev = int(min(LEV_CAP, max(1, np.floor(MARGIN_HIT / (stop_pct + FEE_RT)))))
    notional = RISK * equity / ((stop_pct + FEE_RT) / 100); margin = notional / lev
    net = lambda x: (abs(x - entry) / entry * 100 - FEE_RT) / (stop_pct + FEE_RT)
    return dict(entry=entry, stop=stop, tp1=tp1, tp2=tp2, stop_pct=round(stop_pct, 2), net_rr=round(net(tp2), 2),
                net_rr_a=round(net(tp1), 2), lev=lev, notional=round(notional, 1), margin=round(margin, 2))


def _fmt(x: float) -> float:
    if not np.isfinite(x): return x
    mag = 10 ** (np.floor(np.log10(abs(x))) - 4) if x else 1
    return float(np.round(np.round(x / mag) * mag, 10))


def structures(sym: str, equity: float) -> list[dict]:
    """Tested levels, trendlines, pullbacks, failed moves and coil brackets. Crypto on 4h; stocks on daily."""
    tf = "4h" if is_crypto(sym) else "1d"
    d = completed(bars(sym, tf), tf) if is_crypto(sym) else bars(sym, tf)
    d = d.iloc[-300:]
    if len(d) < 80:
        return []
    kind = "crypto" if is_crypto(sym) else "stocks"; trend = market_trend(kind); vol, _ = vol_state(sym)
    a = atr(d).values; Hh, L, C = d.h.values, d.l.values, d.c.values; i = len(d) - 1; px = price(sym)[0]
    zones: dict[int, list[tuple[float, float, int]]] = {}
    for side in (1, -1):
        vals = L if side == 1 else Hh
        pts = sorted((vals[p], p) for p in _pivots(vals, side) if i - 120 <= p <= i - 3); cl: list = []
        for v, p in pts:
            if cl and abs(v - cl[-1][-1][0]) <= 0.5 * a[i]: cl[-1].append((v, p))
            else: cl.append([(v, p)])
        zs = []
        for c in cl:
            if len(c) < 2: continue
            lvl = float(np.mean([v for v, _ in c])); last = max(p for _, p in c)
            edge = min(v for v, _ in c) if side == 1 else max(v for v, _ in c)
            if (side == 1 and (C[last:i + 1] < edge).any()) or (side == -1 and (C[last:i + 1] > edge).any()): continue
            if (side == 1 and lvl < px) or (side == -1 and lvl > px): zs.append((lvl, edge, len(c)))
        zs.sort(key=lambda z: -z[0] * side); zones[side] = zs
    out = []

    dd = completed(bars(sym, "1d"), "1d") if is_crypto(sym) else bars(sym, "1d"); datr = float(atr(dd).iat[-1])

    def add(setup: str, side: int, entry: float, stop: float, hint: float | None, order: str, touches: int = 0) -> None:
        if abs(entry - px) > 1.5 * datr:
            return  # too far from price to fill inside the next window
        ext = Hh[-120:].max() if side == 1 else L[-120:].min()
        swings = [(Hh if side == 1 else L)[p] for p in _pivots(Hh if side == 1 else L, -side) if p >= i - 120]
        cands = {z[0] for z in zones[-side]} | {ext} | set(swings) | ({hint} if hint is not None and np.isfinite(hint) else set())
        # dated levels (tested zones, swing highs or lows, the 120-bar extreme) within 4 daily ATR of the entry
        cands = sorted((x for x in cands if 0 < side * (x - entry) <= 4 * datr), key=lambda x: side * (x - entry))
        if not cands:
            return
        okA = [x for x in cands if ticket(sym, side, entry, stop, x, equity)["net_rr"] >= FLOOR]
        tp1 = okA[0] if okA else cands[-1]  # TP A: the nearest dated level that clears the floor
        tp2 = cands[-1] if side * (cands[-1] - tp1) > 0 else tp1  # TP B: the farthest dated level, the bigger move
        t = ticket(sym, side, entry, stop, tp2, equity, tp1); al = alignment(side, trend); flags = []
        if t["stop_pct"] < 0.5: flags.append("stop under 0.5%")
        if t["net_rr_a"] < FLOOR: flags.append(f"net {t['net_rr_a']}:1 under {FLOOR}:1")
        if chasing(sym, side): flags.append("CHASING")
        if base(sym) not in GUARANTEED_STOP: flags.append("no guaranteed stop")
        if al == "against market" and vol == "high vol": flags.append("against market in high vol")
        if setup == "failed move" and al == "market flat": flags.append("flat-market failed move, under test")
        if setup == "pullback" and vol == "high vol": flags.append("high-vol pullback needs divergence or pattern")
        blocking = [f for f in flags if f.startswith(("stop under", "net ", "CHASING"))]
        up = completed(bars(sym, "1d"), "1d") if (tf == "4h" and is_crypto(sym)) else (bars(sym, "1d") if tf == "4h" else None)
        out.append(dict(sym=base(sym), setup=setup, side="long" if side == 1 else "short", order=order,
                        entry=_fmt(entry), stop=_fmt(stop), tp1=_fmt(t["tp1"]), tp2=_fmt(tp2), stop_pct=t["stop_pct"],
                        net_rr=t["net_rr"], net_rr_a=t["net_rr_a"], lev=t["lev"], margin=t["margin"], touches=touches, tf=tf,
                        rsi=round(float(rsi(d.c).iat[-1]), 1), macd=macd_state(d.c), div=divergence(d, side),
                        div_up=divergence(up, side) if up is not None else "n/a", trend=trend, vol=vol, align=al,
                        away=round(abs(entry - px) / datr, 2), eligible="no: " + ", ".join(blocking) if blocking else "yes",
                        flags="; ".join(flags)))

    for side in (1, -1):
        for lvl, edge, touches in zones[side][:2]:
            add("tested support" if side == 1 else "tested resistance", side, lvl, edge - side * STOP_BEYOND_ATR * a[i], None, "maker limit", touches)
        vals = L if side == 1 else Hh; ps = [p for p in _pivots(vals, side) if i - 120 <= p <= i - 3]
        if len(ps) >= 2:
            p1, p2 = ps[-2], ps[-1]
            if p2 - p1 >= 5 and ((vals[p2] > vals[p1]) if side == 1 else (vals[p2] < vals[p1])):
                s = (vals[p2] - vals[p1]) / (p2 - p1); js = np.arange(p1, i + 1); line = vals[p1] + s * (js - p1)
                nxt = vals[p1] + s * (i + 1 - p1)
                if not ((C[js] < line).any() if side == 1 else (C[js] > line).any()) and side * (px - nxt) > 0:
                    add("rising trendline" if side == 1 else "falling trendline", side, nxt, nxt - side * STOP_BEYOND_ATR * a[i], None, "maker limit", 2)
    e20 = pd.Series(C).ewm(span=20, adjust=False).mean().values; e50 = pd.Series(C).ewm(span=50, adjust=False).mean().values
    hh = pd.Series(Hh).rolling(20).max().shift().values; ll = pd.Series(L).rolling(20).min().shift().values
    for side in (1, -1):
        trending = (e20[i] > e50[i] and C[i] > e50[i]) if side == 1 else (e20[i] < e50[i] and C[i] < e50[i])
        if trending:
            lo10 = L[-10:].min() if side == 1 else Hh[-10:].max(); stop = lo10 - side * 0.25 * a[i]
            if (stop < e20[i] < px) if side == 1 else (stop > e20[i] > px):
                add("pullback", side, e20[i], stop, Hh[-20:].max() if side == 1 else L[-20:].min(), "maker limit")
        swept = (L[i] < ll[i] and C[i] > ll[i]) if side == 1 else (Hh[i] > hh[i] and C[i] < hh[i])
        if swept:
            add("failed move", side, C[i], L[i] - 0.25 * a[i] if side == 1 else Hh[i] + 0.25 * a[i], hh[i] if side == 1 else ll[i], "maker limit")
    width = (pd.Series(hh) - pd.Series(ll)) / pd.Series(a)
    if width.iat[i] < width.rolling(200, min_periods=60).quantile(0.2).iat[i]:
        add("bracket", 1, hh[i], ll[i] - 0.25 * a[i], None, "stop entry")
        add("bracket", -1, ll[i], hh[i] + 0.25 * a[i], None, "stop entry")
    return out


# ---------- tested edges ----------
def daily_done(sym: str) -> pd.DataFrame:
    return completed(bars(sym, "1d"), "1d")


def max10(sym: str) -> dict:
    d = daily_done(sym); c = d.c; a = atr(d); hi = c.rolling(10).max().shift(1)
    sig, held = bool(c.iat[-1] >= hi.iat[-1]), bool(c.iat[-2] >= hi.iat[-2])
    px = price(sym)[0]
    action = "roll" if sig and held else "buy" if sig else "close" if held else "none"
    return dict(sym=sym, last_close_day=str(d.index[-1].date()), close=_fmt(c.iat[-1]), prior_10d_high=_fmt(hi.iat[-1]), action=action,
                price=_fmt(px), stop=_fmt(px - a.iat[-1]) if sig else None, stop_pct=round(a.iat[-1] / px * 100, 2))


def trend_components(sym: str) -> list[dict]:
    d = daily_done(sym); c, lo = d.c.values, d.l.values; out = []
    for n in TREND_N:
        mx = pd.Series(c).rolling(n).max(); mn = pd.Series(c).rolling(n).min(); prev = mx.shift(1).values; mid = ((mx + mn) / 2).values
        long, stop, since, entry, init = False, np.nan, None, np.nan, np.nan; log = []; o = d.o.values
        for t in range(n, len(c)):
            if long:
                if lo[t] <= stop:
                    log.append((since, d.index[t], entry, init, min(stop, o[t]))); long = False
                else:
                    stop = max(stop, mid[t])
            if not long and c[t] >= prev[t] and (c[t] - mid[t]) / c[t] >= 0.005:
                long, stop, since, entry, init = True, mid[t], d.index[t], c[t], mid[t]
        state = "insufficient history" if len(c) < n + 2 else ("new entry" if long and since == d.index[-1] else "long" if long else "flat")
        out.append(dict(sym=sym, lookback=n, state=state, since=str(since.date()) if long else "", entry=_fmt(entry) if long else None,
                        stop=_fmt(stop) if long else None, log=log))
    return out


def edges_record(since: str = EDGES_SINCE) -> pd.DataFrame:
    """Paper record of both tested edges from `since`, scored mechanically on daily bars (entry and exit at the 00:00 UTC
    close stand in for the 08:30 run; stops on daily lows)."""
    rows = []; t0 = pd.Timestamp(since)
    for s in EDGE_COINS:
        d = daily_done(s); c, lo = d.c, d.l; a = atr(d); sig = c >= c.rolling(10).max().shift(1)
        for k in range(1, len(d) - 1):
            if d.index[k] < t0 - pd.Timedelta(days=1) or not sig.iat[k]:
                continue
            entry, stop = c.iat[k], c.iat[k] - a.iat[k]; ex = stop if lo.iat[k + 1] <= stop else c.iat[k + 1]
            rows.append(dict(edge="MAX10", sym=s, entry_day=str(d.index[k + 1].date()), R=round((ex - entry) / a.iat[k] - (MAKER + TAKER) * entry / a.iat[k], 3)))
        for comp in trend_components(s):
            for since_, exit_day, entry, init, exit_px in comp["log"]:
                if since_ >= t0 - pd.Timedelta(days=1):
                    risk = entry - init
                    rows.append(dict(edge=f"TREND-{comp['lookback']}", sym=s, entry_day=str(since_.date()), exit_day=str(exit_day.date()),
                                     R=round((exit_px - entry) / risk - 2 * TAKER * entry / risk, 3)))
    return pd.DataFrame(rows)


# ---------- scoring ----------
STATE_COLS = ["id", "run", "sym", "side", "verdict", "setup", "entry", "stop", "tp1", "tp2", "valid_until", "status",
              "filled_at", "closed_at", "R", "live", "note"]


def score_row(r: pd.Series) -> dict:
    """Split ticket: half to tp1, half to tp2, one stop, maker entry at the level (stop entry for brackets)."""
    out = r.to_dict()
    if r.status not in ("resting", "open", "tp1") or not isinstance(r.sym, str):
        return out
    side = 1 if str(r.side).lower().startswith("l") else -1
    try:
        d = bars(r.sym, "15m" if is_crypto(r.sym) else "1h")
    except Exception:
        out["note"] = f"{r.note}; no data for scoring" if isinstance(r.note, str) and "no data" not in r.note else r.note
        return out
    start = pd.Timestamp(r.run); d = d[d.index >= start.floor("15min")]
    if d.empty:
        return out
    try:
        entry, stop = float(r.entry), float(r.stop)
    except (TypeError, ValueError):
        return out
    if not (np.isfinite(entry) and np.isfinite(stop)) or entry == stop:
        return out  # no ticket levels (for example the owner's own position): reported, not scored
    tp1 = float(r.tp1) if not pd.isna(r.tp1) else np.nan; tp2 = float(r.tp2) if not pd.isna(r.tp2) else tp1
    tp1 = tp2 if np.isnan(tp1) else tp1; risk = abs(entry - stop)
    stop_entry = "bracket" in str(r.setup)
    if pd.isna(r.filled_at) or r.status == "resting":
        hit = (d.h >= entry) if (side == 1) == stop_entry else (d.l <= entry)
        until = pd.Timestamp(r.valid_until) if not pd.isna(r.valid_until) else pd.Timestamp.max
        hit = hit[d.index < until]
        if not hit.any():
            if d.index[-1] >= until:
                out.update(status="expired", closed_at=str(until), R=0.0)
            return out
        out["filled_at"] = str(hit.idxmax()); out["status"] = "open"
    j0 = int(d.index.searchsorted(pd.Timestamp(out["filled_at"]).floor("15min")))
    if j0 >= len(d):
        return out
    fee_in = (TAKER if stop_entry else MAKER) * entry / risk; legs = []; closed_at = None
    hit_stop = lambda j, lvl: (d.l.iat[j] <= lvl) if side == 1 else (d.h.iat[j] >= lvl)
    hit_tp = lambda j, lvl: np.isfinite(lvl) and ((d.h.iat[j] >= lvl) if side == 1 else (d.l.iat[j] <= lvl))
    legA, jA = None, None
    for j in range(j0, len(d)):
        if hit_stop(j, stop):
            legA = (-1 - TAKER * entry / risk, d.index[j]); break
        if j > j0 and hit_tp(j, tp1):
            legA = (abs(tp1 - entry) / risk - MAKER * entry / risk, d.index[j]); jA = j; break
    legB, s_run = None, stop
    for j in range(j0, len(d)):
        if jA is not None and j > jA:
            s_run = entry  # runner to breakeven once TP A has filled
        if hit_stop(j, s_run):
            legB = (side * (s_run - entry) / risk - TAKER * entry / risk, d.index[j]); break
        if j > j0 and hit_tp(j, tp2):
            legB = (abs(tp2 - entry) / risk - MAKER * entry / risk, d.index[j]); break
    legs = [legA, legB]
    mark = float(d.c.iat[-1]); openR = side * (mark - entry) / risk
    if all(legs):
        R = 0.5 * legs[0][0] + 0.5 * legs[1][0] - fee_in; closed_at = max(legs[0][1], legs[1][1])
        out.update(status="stopped" if R < 0 else "closed", closed_at=str(closed_at), R=round(R, 3))
    elif legs[0]:
        out.update(status="tp1" if legs[0][0] > 0 else "stopped", R=round(0.5 * legs[0][0] + 0.5 * openR - fee_in, 3))
    elif legs[1]:
        out.update(status="open", R=round(0.5 * legs[1][0] + 0.5 * openR - fee_in, 3))
    else:
        out.update(status="open", R=round(openR - fee_in, 3))
    out["mark"] = _fmt(mark)
    return out


def _agg(S: pd.DataFrame, verdict: str, live: bool | None = None) -> dict:
    a = S[(S.status == "agg") & (S.verdict == verdict)]
    if live is not None:
        a = a[a.live.astype(str).str.lower().isin(["y", "yes"]) == live]
    out = {"n": 0, "sumR": 0.0, "wins": 0, "gw": 0.0, "gl": 0.0, "eq": None}
    for _, r in a.iterrows():
        kv = dict(x.split("=") for x in str(r.note).split(";") if "=" in x)
        out["n"] += int(kv.get("n", 0)); out["wins"] += int(kv.get("wins", 0)); out["sumR"] += float(r.R or 0)
        out["gw"] += float(kv.get("gw", 0)); out["gl"] += float(kv.get("gl", 0))
        if "eq" in kv: out["eq"] = float(kv["eq"])
    return out


def resolved(S: pd.DataFrame) -> pd.DataFrame:
    d = S[S.status.isin(["closed", "stopped", "expired"])].copy(); d["R"] = pd.to_numeric(d.R, errors="coerce")
    return d[d.R.notna()]


def record(S: pd.DataFrame) -> list[str]:
    done = resolved(S); lines = []
    paper = done[done.verdict == "TAKE"].sort_values("closed_at"); A = _agg(S, "TAKE")
    eq = paper_eq0(S)
    for R in paper.R: eq *= 1 + RISK * R
    n = A["n"] + len(paper); wins = A["wins"] + int((paper.R > 0).sum())
    gw = A["gw"] + paper.R[paper.R > 0].sum(); gl = A["gl"] - paper.R[paper.R < 0].sum(); sumR = A["sumR"] + paper.R.sum()
    lines.append(f"Paper (TAKE rows): {n} resolved, win rate {wins / max(n, 1) * 100:.0f}%, mean {sumR / max(n, 1):+.2f}R, "
                 f"profit factor {gw / gl if gl else float('nan'):.2f}, simulated equity {eq:.2f} USDT from 200" + (" (indicative below 15)" if n < 15 else ""))
    P = _agg(S, "PASS"); pr = done[done.verdict == "PASS"].R; pn = P["n"] + len(pr); psum = P["sumR"] + pr.sum()
    lines.append(f"PASS rows (paper only): {pn} resolved" + (f", mean {psum / pn:+.2f}R" if pn else ""))
    if n and pn:
        lines.append(f"Evaluator edge, TAKE minus PASS: {sumR / n - psum / pn:+.2f}R" + (" (indicative below 30 each)" if min(n, pn) < 30 else ""))
    lv = done[done.live.astype(str).str.lower().isin(["y", "yes"])]; L = _agg(S, "TAKE", True); C = _agg(S, "OWN", True)
    ln = len(lv) + L["n"] + C["n"]; lsum = lv.R.sum() + L["sumR"] + C["sumR"]
    if ln:
        lines.append(f"Live (confirmed fills): {ln} resolved, net {lsum:+.2f}R, mean {lsum / ln:+.2f}R")
    return lines


def _live(x: pd.Series) -> pd.Series:
    return x.fillna("n").astype(str).str.lower().map(lambda v: "y" if v in ("y", "yes", "true") else "n")


def prune(S: pd.DataFrame, days: int = 3) -> pd.DataFrame:
    """Roll resolved rows older than `days` into one aggregate row per verdict and live flag."""
    done = resolved(S); cut = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days)
    old = done[pd.to_datetime(done.closed_at, errors="coerce") < cut]
    if old.empty:
        return S
    aggs = S[S.status == "agg"]; rest = S.drop(index=old.index); rest = rest[rest.status != "agg"]
    keys = set(zip(old.verdict, _live(old.live))) | set(zip(aggs.verdict, _live(aggs.live))); new = []
    for v, lv in sorted(keys):
        g = old[(old.verdict == v) & (_live(old.live) == lv)].sort_values("closed_at"); prev = _agg(aggs[(aggs.verdict == v) & (_live(aggs.live) == lv)], v)
        R = g.R.astype(float)
        note = (f"n={prev['n'] + len(g)};wins={prev['wins'] + int((R > 0).sum())};gw={prev['gw'] + R[R > 0].sum():.3f};"
                f"gl={prev['gl'] - R[R < 0].sum():.3f}")
        new.append(dict(id=f"AGG-{v}-{lv}", verdict=v, live=lv, status="agg", R=round(prev["sumR"] + R.sum(), 3), note=note))
    eq = paper_eq0(S)
    for x in old[old.verdict == "TAKE"].sort_values("closed_at").R.astype(float): eq *= 1 + RISK * x
    new.append(dict(id="AGG-EQ", verdict="EQ", live="n", status="agg", R=0.0, note=f"eq={eq:.4f}"))
    return pd.concat([rest, pd.DataFrame(new)], ignore_index=True)


def paper_eq0(S: pd.DataFrame) -> float:
    r = S[(S.status == "agg") & (S.verdict == "EQ")]
    return float(str(r.note.iat[0]).split("eq=")[1]) if len(r) else 200.0


def score_state(path: str) -> None:
    S = pd.read_csv(path, dtype=str)
    for c in STATE_COLS:
        if c not in S: S[c] = np.nan
    rows = [score_row(r) for _, r in S.iterrows()]; N = pd.DataFrame(rows)
    marks = N.pop("mark") if "mark" in N else pd.Series(np.nan, index=N.index)
    prune(N[STATE_COLS])[STATE_COLS].to_csv(path, index=False)
    live = N[N.status.isin(["resting", "open", "tp1"])]
    print(f"BOOK (crypto scored on 15m bars, stocks on 1h cash bars; feed updated {manifest().get('updated_utc', 'unknown')} UTC)")
    for (_, r), m in zip(live.iterrows(), marks[live.index]):
        print(f"  {r.id} {r.sym} {r.side} {r.verdict} {r.setup}: entry {r.entry} stop {r.stop} tp1 {r.tp1} tp2 {r.tp2} status {r.status}"
              f"{'' if pd.isna(m) else f', mark {m}'}{'' if pd.isna(r.R) else f', R {r.R}'}{', LIVE' if str(r.live).lower() in ('y', 'yes') else ''}")
    changed = N[(N.status != S.status)]
    for _, r in changed.iterrows():
        print(f"  CHANGED {r.id} {r.sym} {r.side}: {S.loc[_, 'status']} -> {r.status}{'' if pd.isna(r.R) else f' ({r.R}R)'}")
    open_risk = len(N[N.status.isin(["open", "tp1"]) & N.verdict.isin(["TAKE", "OWN"])]) * RISK * 100
    print(f"Open risk on filled TAKE and OWN rows: about {open_risk:.0f}% of equity (5% a row) against the 40% cap")
    print("\n".join(record(N)))


# ---------- brief ----------
TD = '<td style="border:1px solid #c5cdd6;white-space:nowrap{}">{}</td>'
TH = '<th bgcolor="#e4e9ef" style="background-color:#e4e9ef;border:1px solid #c5cdd6;text-align:left">{}</th>'  # the mail tool strips "background:"


def _table(head: list[str], rows: list[list]) -> str:
    def cell(x):
        s = str(x); num = s.replace(",", "").replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).rstrip("x%").isdigit()
        return TD.format(";text-align:right" if num else "", s)
    return ('<table cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px"><tr>' + "".join(TH.format(h) for h in head)
            + "</tr>" + "".join("<tr>" + "".join(cell(x) for x in r) + "</tr>" for r in rows) + "</table>")


def html(spec: dict) -> tuple[str, str]:
    P = '<p style="margin:14px 0 4px"><b>{}</b></p>'; L = '<p style="margin:0">{}</p>'
    h, t = [], []
    h += [P.format("Status"), L.format(spec["status"])]; t += ["Status", spec["status"], ""]
    if spec.get("do_now"):
        h.append(P.format("Do now")); t.append("Do now")
        for n, x in enumerate(spec["do_now"][:5], 1):
            h.append(L.format(f"{n}. {x}")); t.append(f"{n}. {x}")
        t.append("")
    h.append(P.format("New tickets")); t.append("New tickets")
    head = ["Ticker", "Side", "Entry", "TP A", "TP B", "Stop", "Lev", "Margin"]
    if spec.get("tickets"):
        h.append(_table(head, spec["tickets"])); t += [", ".join(head)] + [", ".join(map(str, r)) for r in spec["tickets"]]
    else:
        h.append(L.format("No new tickets.")); t.append("No new tickets.")
    for f in spec.get("flags", []):
        h.append(f'<p style="margin:6px 0 0">{f}</p>'); t.append(f)
    t.append("")
    if spec.get("book"):
        head = ["Position", "Entry", "Stop", "Mark", "R", "Action"]
        h += [P.format("Book"), _table(head, spec["book"])]; t += ["Book", ", ".join(head)] + [", ".join(map(str, r)) for r in spec["book"]] + [""]
    h.append(P.format("Score")); t.append("Score")
    for x in [spec.get("score", "")] + ([spec["edges"]] if spec.get("edges") else []):
        h.append(L.format(x)); t.append(x)
    body = '<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:14px;color:#1f2933">' + "".join(h) + "</div>"
    text = "\n".join(t)
    for bad in ("—", "–", "|"):
        assert bad not in body + text, f"forbidden character {bad!r} in the brief"
    return body, text


# ---------- commands ----------
def scan(equity: float, stocks: bool) -> None:
    age = feed_age_hours()
    print(f"DATA: feed updated {manifest().get('updated_utc', 'unknown')} UTC ({age:.1f}h old)"
          + ("; STALE: quote current prices from a Coinbase ticker before printing any row" if age > 1.5 else ""))
    ct = market_trend("crypto"); print(f"REGIME: crypto market trend {ct} (BTC daily close vs EMA50)", end="")
    if stocks:
        print(f"; US market trend {market_trend('stocks')} (QQQ)", end="")
    print()
    syms = list(SCAN_CRYPTO) + (list(SCAN_STOCKS) if stocks else [])
    print("PRICES AND VOLATILITY:")
    for s in syms:
        try:
            p, ts = price(s); v, ap = vol_state(s)
            print(f"  {s} {_fmt(p)} at {ts:%Y-%m-%d %H:%M} UTC bar, {v}, daily ATR {ap * 100:.2f}%")
        except Exception as e:
            print(f"  {s}: unavailable ({type(e).__name__})")
    print("TESTED EDGES (paper from 28 Sep; long only):")
    for s in EDGE_COINS:
        m = max10(s); comps = trend_components(s); act = [c for c in comps if c["state"] in ("long", "new entry")]
        new = [str(c["lookback"]) for c in comps if c["state"] == "new entry"]
        print(f"  {s}: MAX-10 {m['action']}" + (f" at {m['price']}, stop {m['stop']} ({m['stop_pct']}%)" if m['action'] in ("buy", "roll") else "")
              + f"; trend components long {len(act)}/9" + (f", new today: {', '.join(new)}" if new else "")
              + ("; stops " + ", ".join(f"{c['lookback']}d {c['stop']}" for c in act) if act else ""))
    rows = []
    for s in syms:
        try:
            rows += structures(s, equity)
        except Exception as e:
            print(f"  candidates for {s} unavailable ({type(e).__name__}: {e})")
    el = sorted((r for r in rows if r["eligible"] == "yes"), key=lambda r: r["away"]); nel = [r for r in rows if r["eligible"] != "yes"]
    more = len(el) - 10; el = el[:10]  # the ten nearest get verdicts; the rest are too far to matter this window
    for n, r in enumerate(el, 1):
        r["cid"] = f"K{n}"
    pd.DataFrame(el).to_csv("candidates.csv", index=False)
    print(f"CANDIDATES within 1.5 daily ATR of price, ELIGIBLE (the {len(el)} nearest), each needs a verdict (equity {equity} USDT, 5% risk; "
          f"TP A is the nearest dated level at {FLOOR}:1 net or better, TP B the farthest within 4 daily ATR; saved to candidates.csv):")
    for r in el:
        print(f"  {r['cid']} {r['sym']} {r['side']} {r['setup']} ({r['tf']}, {r['order']}, {r['away']} ATR away): entry {r['entry']} stop {r['stop']} ({r['stop_pct']}%) "
              f"TP A {r['tp1']} ({r['net_rr_a']}:1) TP B {r['tp2']} ({r['net_rr']}:1), {r['lev']}x, margin {r['margin']}; RSI {r['rsi']}, MACD {r['macd']}, "
              f"divergence {r['div']} (daily {r['div_up']}); {r['align']}, {r['vol']}" + (f"; {r['flags']}" if r['flags'] else ""))
    why = pd.Series([f"under {FLOOR}:1" if f"{FLOOR}:1" in x else x.strip() for r in nel for x in r["eligible"][4:].split(",")]).value_counts()
    print(f"NOT ELIGIBLE ({len(nel)} rows): " + ", ".join(f"{k} {v}" for k, v in why.items())
          + (f"; {more} further eligible rows not listed (farther from price)" if more > 0 else ""))
    if "--all" in sys.argv:
        for r in nel:
            print(f"  {r['sym']} {r['side']} {r['setup']} {r['entry']} stop {r['stop']} tp2 {r['tp2']} ({r['eligible'][4:]})")


def verdicts(path: str, args: list[str]) -> None:
    """Append judged candidates to the state as TAKE or PASS rows; print the ticket rows for the TAKEs."""
    S = pd.read_csv(path, dtype=str); C = pd.read_csv("candidates.csv", dtype=str).set_index("cid")
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    until = now + pd.Timedelta(days=3 if now.dayofweek == 4 else 1)  # the next run of the same kind
    new, tickets = [], []
    for a in args:
        cid, rest = a.split("=", 1); verdict, _, reason = rest.partition(":"); verdict = verdict.strip().upper(); c = C.loc[cid.strip()]
        assert verdict in ("TAKE", "PASS"), f"verdict must be TAKE or PASS: {a}"
        rid = f"{'T' if verdict == 'TAKE' else 'P'}{now:%m%d%H%M}-{cid.strip()}"
        new.append(dict(id=rid, run=now.isoformat(timespec="minutes"), sym=c.sym, side=c.side, verdict=verdict, setup=c.setup,
                        entry=c.entry, stop=c.stop, tp1=c.tp1, tp2=c.tp2, valid_until=until.isoformat(timespec="minutes"),
                        status="resting", live="n", note=reason.strip().replace(",", ";")))
        if verdict == "TAKE":
            tickets.append([c.sym, c.side, c.entry, c.tp1, c.tp2, c.stop, f"{c.lev}x", c.margin])
    pd.concat([S, pd.DataFrame(new)], ignore_index=True)[STATE_COLS].to_csv(path, index=False)
    print(f"added {len(new)} rows ({len(tickets)} TAKE) to {path}")
    if tickets:
        print("tickets for brief.json: " + json.dumps(tickets))


def main(argv: list[str]) -> None:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cmd = argv[1] if len(argv) > 1 else "scan"
    eq = float(argv[argv.index("--equity") + 1]) if "--equity" in argv else 170.0
    if cmd == "refresh":
        refresh()
    elif cmd == "scan":
        scan(eq, "--crypto-only" not in argv)
    elif cmd == "score":
        score_state(argv[2])
    elif cmd == "verdict":
        verdicts(argv[2], argv[3:])
    elif cmd == "edges":
        for s in EDGE_COINS:
            print(max10(s)); print(pd.DataFrame([{k: v for k, v in c.items() if k != "log"} for c in trend_components(s)]).to_string(index=False))
        rec = edges_record()
        print("\nPaper record since", EDGES_SINCE, ":", "none yet" if rec.empty else "")
        if not rec.empty:
            print(rec.groupby("edge").R.agg(["size", "mean", "sum"]).round(3).to_string())
    elif cmd == "html":
        spec = json.loads(Path(argv[2]).read_text()); body, text = html(spec)
        Path("brief.html").write_text(body); Path("brief.txt").write_text(text); print(text)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
