"""Bitunix Super Alert templates against the standing gate (owner, 25 Oct 2026), pre-registered before running.

Bitunix does not publish the conditions a template fills in, so each is built from its card (name, description, tags)
with textbook settings, the owner's choice: RSI 14 (Wilder), Bollinger 20 and 2 (population sd), MACD 12/26/9 (golden
cross: DIF crosses above DEA), EMA 50, DEMA 55, TEMA 9, VWAP reset each UTC day (crypto) or session (stocks).
Conditions are read on a closed bar; long unless stated; A or B is the exit family.
  BX1  Bollinger oversold mean reversion: close below the lower band, RSI under 30, close below EMA 50. A
  BX2  Bollinger upper band breakout: close crosses above the upper band, RSI over 50. B
  BX3  Mid-band MACD cross: golden cross, close above the middle band, RSI over 50. B
  BX4  TEMA cross above VWAP: TEMA 9 crosses above VWAP; 1h and 4h only. B
  BX5  DEMA55 bullish trend follow: golden cross, close above DEMA 55. B
  BX6  VWAP bullish confirmation: golden cross, close above VWAP; 1h and 4h only. B
  BX7  Three white soldiers trend confirm: the pattern, RSI over 50, close above EMA 50. B
  BX8  MACD golden cross, low RSI: golden cross, RSI under 30. A
  BX9  MACD death cross, high RSI, short: death cross, RSI over 70. A
  BX10 Hammer oversold bounce: a hammer, RSI under 30, close below EMA 50. A
  BX11 Morning star bullish divergence: a morning star whose low undercuts the lowest low of bars i-40 to i-5 while DIF
       at the pattern low is above DIF at that earlier low. A
Hammer: body at most 0.35 of the range, upper shadow at most 0.1 of it, lower shadow at least twice the body. Three white
soldiers: three bullish bars, each with a body of at least half its range and an upper shadow of at most a quarter of
it, each closing above the last and opening within the prior body. Morning star: a bearish bar with a body of at least
half its range; a bar whose body is at most 0.3 of the first body and sits at or below the first close; a bullish bar
closing above the midpoint of the first body.

Execution, an exit per template (owner's choice). Entry at the next bar's open as taker. Stop 1 ATR(14) beyond the
signal bar's low (high for the short; for BX11 the pattern's low); 1R is the entry-to-stop distance; trades left with
under half an ATR of room after the open are skipped; a gap through the stop fills at the open.
  A, mean reversion and reversal: out at the first close at or beyond the Bollinger middle band, or at the 20th close.
  B, breakout and trend: a chandelier trail, the highest high since entry less 3 signal-bar ATRs, raised and never
     lowered; out at the 200th close at the latest.
The first 200 bars of each series are warm-up. Trades still open when the data ends are marked at the last close.
One position per template and symbol at a time. Costs: taker 0.06% and slippage 0.02% a side; longs pay funding at
0.03% a day (crypto) or 0.0105% (stocks); shorts receive none.

Sets: crypto 1h from 2022: BTC, ETH, SOL, XRP design to 2024, held out from 2025; LINK, BNB, ADA, DOGE, AVAX, DOT fresh.
      crypto 4h and 1d from 2020: the same coins, design to 2023, held out from 2024.
      stocks 1d from 2016: the 18 scan stocks design to 2022, held out from 2023; 40 further names fresh.
      stocks 1h from 27 Nov 2023: the 18 scan stocks design to June 2025, held out after; IWM, NFLX, ORCL, SOXL, TQQQ
      fresh.
Gate, as standing: design net t of 3 or more, positive in both halves of the design period, positive held out and on
fresh symbols, and above its placebo in every set. Placebo: the same exit from every bar in the same state (A long: close
below the middle band; A short: above it; B: close above EMA 50), on the same symbol and dates; each trade is scored
against the placebo mean of its symbol and set. 51 variants, every one counted.

Data: crypto from static-klines (Binance spot), stocks from the feed's research folders (Yahoo), cached in ../data.
Usage: python bitunix_templates.py fetch | run
"""
from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
import swing as S  # noqa: E402

DATA = ROOT / "data" / "templates"
CRYPTO_DESIGN = ("BTC", "ETH", "SOL", "XRP")
CRYPTO_FRESH = ("LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT")
CRYPTO_START = {"1h": "2022-01-01", "4h": "2020-01-01", "1d": "2020-01-01"}
STOCK_DESIGN = tuple(E.SCAN_STOCKS)
STOCK_FRESH_1D = tuple(s for s in S.STOCK_POOL if s not in STOCK_DESIGN) + tuple(s for s in S.ETFS if s not in STOCK_DESIGN)
STOCK_FRESH_1H = ("IWM", "NFLX", "ORCL", "SOXL", "TQQQ")  # every hourly file outside the design set except UVXY (decay product)

TAKER, SLIP = 0.0006, 0.0002
FUND = {"crypto": 0.0003, "stock": 0.000105}
A_CAP, B_CAP, TRAIL, MIN_ROOM, WARMUP = 20, 200, 3.0, 0.5, 200
BAR = {"1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h"), "1d": pd.Timedelta("1D")}
UNIVERSES = {  # kind, interval, design symbols, fresh symbols, start, held-out start
    "crypto 1h": ("crypto", "1h", CRYPTO_DESIGN, CRYPTO_FRESH, "2022-01-01", "2025-01-01"),
    "crypto 4h": ("crypto", "4h", CRYPTO_DESIGN, CRYPTO_FRESH, "2020-01-01", "2024-01-01"),
    "crypto 1d": ("crypto", "1d", CRYPTO_DESIGN, CRYPTO_FRESH, "2020-01-01", "2024-01-01"),
    "stocks 1d": ("stock", "1d", STOCK_DESIGN, STOCK_FRESH_1D, "2016-01-01", "2023-01-01"),
    "stocks 1h": ("stock", "1h", STOCK_DESIGN, STOCK_FRESH_1H, "2023-11-27", "2025-07-01"),
}
TEMPLATES = {  # exit family, side, stop anchor, card name
    "BX1": ("A", 1, "bar", "Bollinger oversold mean reversion"),
    "BX2": ("B", 1, "bar", "Bollinger upper band breakout"),
    "BX3": ("B", 1, "bar", "Mid-band MACD cross"),
    "BX4": ("B", 1, "bar", "TEMA cross above VWAP"),
    "BX5": ("B", 1, "bar", "DEMA55 bullish trend follow"),
    "BX6": ("B", 1, "bar", "VWAP bullish confirmation"),
    "BX7": ("B", 1, "bar", "Three white soldiers trend confirm"),
    "BX8": ("A", 1, "bar", "MACD golden cross, low RSI"),
    "BX9": ("A", -1, "bar", "MACD death cross, high RSI (short)"),
    "BX10": ("A", 1, "bar", "Hammer oversold bounce"),
    "BX11": ("A", 1, "pattern", "Morning star bullish divergence"),
}
VWAP_TEMPLATES = ("BX4", "BX6")


# ---------- data ----------
def _skl(sym: str, interval: str, start: str) -> pd.DataFrame:
    """static-klines in parallel, skipping the repo's placeholder files for future periods."""
    pre = f".klines-cache/{sym}USDT/{interval}/"
    names = sorted(f for f in E._skl_files() if f.startswith(pre))
    starts = [f.rsplit("/", 1)[1][:10] for f in names]
    today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    keep = [n for n, s, nxt in zip(names, starts, starts[1:] + ["9999"]) if nxt > start and s <= today]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for chunk in ex.map(lambda n: json.loads(E._get(E.SKL_RAW + n)), keep) for r in chunk]
    d = pd.DataFrame([r[:6] for r in rows], columns=["t", "o", "h", "l", "c", "v"]).astype(float)
    d.index = pd.to_datetime(d.t.astype("int64"), unit="ms")
    d = d[~d.index.duplicated()].sort_index().drop(columns="t")
    return d[(d.index >= start) & (d.index + BAR[interval] <= pd.Timestamp.now("UTC").tz_localize(None))]  # closed bars


def _research(folder: str, sym: str) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(E._get(f"{E.FEED}research/{folder}/{E.safe(sym)}.csv.gz")), compression="gzip",
                    index_col=0, parse_dates=True)
    return d[["o", "h", "l", "c", "v"]].astype(float).dropna(subset=["o", "h", "l", "c"])


def load(kind: str, sym: str, interval: str) -> pd.DataFrame:
    """kind crypto or stock; cached as csv.gz under data/templates."""
    f = DATA / f"{kind}_{sym}_{interval}.csv.gz"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    d = _skl(sym, interval, CRYPTO_START[interval]) if kind == "crypto" else _research(
        "daily" if interval == "1d" else "hourly", sym)
    DATA.mkdir(parents=True, exist_ok=True)
    d.to_csv(f)
    return d


def fetch() -> None:
    jobs = [("crypto", s, iv) for s in CRYPTO_DESIGN + CRYPTO_FRESH for iv in ("1h", "4h", "1d")]
    jobs += [("stock", s, "1d") for s in STOCK_DESIGN + STOCK_FRESH_1D]
    jobs += [("stock", s, "1h") for s in STOCK_DESIGN + STOCK_FRESH_1H]
    for kind, s, iv in jobs:
        try:
            d = load(kind, s, iv)
            print(f"{kind} {s} {iv}: {len(d)} bars {d.index[0]} to {d.index[-1]}")
        except Exception as e:
            print(f"{kind} {s} {iv}: FAILED {type(e).__name__}: {e}")


# ---------- indicators and signals ----------
def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def indicators(d: pd.DataFrame, interval: str) -> pd.DataFrame:
    c = d.c
    z = pd.DataFrame(index=d.index)
    z["atr"], z["rsi"] = E.atr(d), E.rsi(c)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    z["mid"], z["up"], z["lo"] = mid, mid + 2 * sd, mid - 2 * sd
    z["dif"] = _ema(c, 12) - _ema(c, 26)
    z["dea"] = _ema(z.dif, 9)
    z["ema50"] = _ema(c, 50)
    e55 = _ema(c, 55)
    z["dema55"] = 2 * e55 - _ema(e55, 55)
    t1 = _ema(c, 9)
    t2 = _ema(t1, 9)
    z["tema9"] = 3 * t1 - 3 * t2 + _ema(t2, 9)
    if interval != "1d":
        tp, day = (d.h + d.l + d.c) / 3, d.index.normalize()
        z["vwap"] = (tp * d.v).groupby(day).cumsum() / d.v.groupby(day).cumsum().replace(0, np.nan)
    return z


def signals(d: pd.DataFrame, z: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = d.o, d.h, d.l, d.c
    prev = lambda x, k=1: x.shift(k)
    prevb = lambda x, k=1: x.shift(k, fill_value=False)
    gc = (z.dif > z.dea) & (prev(z.dif) <= prev(z.dea))
    dc = (z.dif < z.dea) & (prev(z.dif) >= prev(z.dea))
    rng, body = h - l, (c - o).abs()
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    hammer = (rng > 0) & (body <= 0.35 * rng) & (upper <= 0.1 * rng) & (lower >= 2 * body)
    soldier = (c > o) & (rng > 0) & (body >= 0.5 * rng) & (h - c <= 0.25 * rng)
    tws = (soldier & prevb(soldier) & prevb(soldier, 2) & (c > prev(c)) & (prev(c) > prev(c, 2))
           & (o >= prev(o)) & (o <= prev(c)) & (prev(o) >= prev(o, 2)) & (prev(o) <= prev(c, 2)))
    a_o, a_c, a_rng, a_body = prev(o, 2), prev(c, 2), prev(rng, 2), prev(body, 2)
    star = ((a_c < a_o) & (a_rng > 0) & (a_body >= 0.5 * a_rng) & (prev(body) <= 0.3 * a_body)
            & (np.maximum(prev(o), prev(c)) <= a_c) & (c > o) & (c > (a_o + a_c) / 2))
    lv, dif, div = l.values, z.dif.values, np.zeros(len(d), bool)
    for i in np.flatnonzero(star.values):
        if i < 40:
            continue
        p = i - 2 + int(np.argmin(lv[i - 2:i + 1]))
        j = i - 40 + int(np.argmin(lv[i - 40:i - 4]))  # bars i-40 to i-5
        div[i] = bool(lv[p] < lv[j] and dif[p] > dif[j])
    sig = {
        "BX1": (c < z.lo) & (z.rsi < 30) & (c < z.ema50),
        "BX2": (c > z.up) & (prev(c) <= prev(z.up)) & (z.rsi > 50),
        "BX3": gc & (c > z.mid) & (z.rsi > 50),
        "BX5": gc & (c > z.dema55),
        "BX7": tws & (z.rsi > 50) & (c > z.ema50),
        "BX8": gc & (z.rsi < 30),
        "BX9": dc & (z.rsi > 70),
        "BX10": hammer & (z.rsi < 30) & (c < z.ema50),
        "BX11": star & div,
    }
    if "vwap" in z:
        sig["BX4"] = (z.tema9 > z.vwap) & (prev(z.tema9) <= prev(z.vwap))
        sig["BX6"] = gc & (c > z.vwap)
    return {k: np.asarray(v, dtype=bool) for k, v in sig.items()}


# ---------- trades ----------
def outcomes(d: pd.DataFrame, z: pd.DataFrame, fam: str, side: int, anchor: np.ndarray, fund: float,
             bar: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """Net R and exit bar of a trade signalled at every bar (NaN and -1 where none is possible)."""
    o, h, l, c = (d[x].values for x in ("o", "h", "l", "c"))
    atr, mid = z.atr.values, z.mid.values
    n = len(d)
    k = np.arange(n - 1)
    e = o[k + 1]
    stop = anchor[k] - side * atr[k]
    risk = side * (e - stop)
    ok = np.isfinite(risk) & np.isfinite(atr[k]) & (risk >= MIN_ROOM * atr[k])
    if fam == "A":
        ok &= np.isfinite(mid[k])
    s, hi = stop.copy(), e.copy()
    px, ix = np.full(len(k), np.nan), np.full(len(k), -1)
    alive = ok.copy()
    for j in range(1, (A_CAP if fam == "A" else B_CAP) + 1):
        t = k + j
        live = alive & (t < n)
        if not live.any():
            break
        tt = np.minimum(t, n - 1)
        hit = live & ((l[tt] <= s) if side == 1 else (h[tt] >= s))
        fill = np.minimum(s, o[tt]) if side == 1 else np.maximum(s, o[tt])
        px[hit], ix[hit] = fill[hit], tt[hit]
        live &= ~hit
        alive &= ~hit
        if fam == "A":
            back = (c[tt] >= mid[tt]) if side == 1 else (c[tt] <= mid[tt])
            out = live & (back | (j == A_CAP))
        else:  # chandelier trail, long only
            hi = np.where(live, np.maximum(hi, h[tt]), hi)
            s = np.where(live, np.maximum(s, hi - TRAIL * atr[k]), s)
            out = live & (j == B_CAP)
        px[out], ix[out] = c[tt][out], tt[out]
        alive &= ~out
    px[alive], ix[alive] = c[n - 1], n - 1  # open at the end of the data: marked at the last close
    days = (d.index.values[np.maximum(ix, 0)] - d.index.values[k + 1]) / np.timedelta64(1, "D") + bar / pd.Timedelta("1D")
    with np.errstate(divide="ignore", invalid="ignore"):
        net = side * (px - e) / risk - 2 * (TAKER + SLIP) * e / risk - (fund * days * e / risk if side == 1 else 0.0)
    R, X = np.full(n, np.nan), np.full(n, -1)
    R[k[ok]], X[k[ok]] = net[ok], ix[ok]
    return R, X


def take(sig: np.ndarray, R: np.ndarray, X: np.ndarray) -> list[int]:
    """Signal bars actually traded: one position at a time."""
    out, free = [], -1
    for i in np.flatnonzero(sig):
        if i >= free and np.isfinite(R[i]):
            out.append(int(i))
            free = X[i]
    return out


def run() -> pd.DataFrame:
    rows: list[dict] = []
    for uni, (kind, iv, design, fresh, start, split) in UNIVERSES.items():
        mid_day = pd.Timestamp(start) + (pd.Timestamp(split) - pd.Timestamp(start)) / 2
        for sym in design + fresh:
            try:
                d = load(kind, sym, iv)
            except Exception as e:
                print(f"{uni} {sym}: no data ({type(e).__name__})")
                continue
            z, n = indicators(d, iv), len(d)
            sig = signals(d, z)
            lows, highs = d.l.values, d.h.values
            pattern = pd.Series(lows).rolling(3).min().values
            cfg = {("A", 1, "bar"): lows, ("A", -1, "bar"): highs, ("A", 1, "pattern"): pattern, ("B", 1, "bar"): lows}
            res = {key: outcomes(d, z, key[0], key[1], anc, FUND[kind], BAR[iv]) for key, anc in cfg.items()}
            c, mid, ema50 = d.c.values, z.mid.values, z.ema50.values
            state = {("A", 1): c < mid, ("A", -1): c > mid, ("B", 1): c > ema50}
            when = d.index
            usable = (when >= start) & (np.arange(n) >= WARMUP)
            sets = ({"design": usable & (when < split), "held": usable & (when >= split)} if sym in design
                    else {"fresh": usable})
            for name, (fam, side, anchor, _) in TEMPLATES.items():
                if name not in sig:
                    continue
                R, X = res[(fam, side, anchor)]
                for set_name, m in sets.items():
                    pool = R[m & state[(fam, side)] & np.isfinite(R)]
                    pm = float(pool.mean()) if len(pool) else np.nan
                    for i in take(sig[name] & m, R, X):
                        rows.append(dict(uni=uni, template=name, sym=sym, set=set_name, half=int(when[i] >= mid_day),
                                         t=when[i], R=R[i], placebo=pm, ex=R[i] - pm, bars=X[i] - i))
    return pd.DataFrame(rows)


def tstat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def report(T: pd.DataFrame) -> None:
    print(__doc__.split("\n\n")[0])
    print("\nNet R a trade. Design, its halves (h1, h2), held out and fresh; d = trade minus its placebo mean.")
    passes = 0
    for uni in UNIVERSES:
        print(f"\n{uni.upper()}")
        print(f"{'':5s} {'n':>5s} {'design':>7s} {'t':>6s} {'h1':>7s} {'h2':>7s} {'held':>7s} {'n':>5s} {'fresh':>7s} {'n':>5s}"
              f" {'d des':>7s} {'t':>6s} {'d held':>7s} {'d frsh':>7s}  verdict")
        for name in TEMPLATES:
            g = T[(T.uni == uni) & (T.template == name)]
            if name in VWAP_TEMPLATES and uni.endswith("1d"):
                continue
            D, H, F = g[g.set == "design"], g[g.set == "held"], g[g.set == "fresh"]
            h1, h2 = D[D.half == 0].R.mean(), D[D.half == 1].R.mean()
            fails = [k for k, bad in (("t", not tstat(D.R) >= 3), ("halves", not (h1 > 0 and h2 > 0)),
                                      ("held", not H.R.mean() > 0), ("fresh", not F.R.mean() > 0),
                                      ("placebo", not (D.ex.mean() > 0 and H.ex.mean() > 0 and F.ex.mean() > 0))) if bad]
            passes += not fails
            print(f"{name:5s} {len(D):5d} {D.R.mean():+7.3f} {tstat(D.R):+6.2f} {h1:+7.3f} {h2:+7.3f} {H.R.mean():+7.3f}"
                  f" {len(H):5d} {F.R.mean():+7.3f} {len(F):5d} {D.ex.mean():+7.3f} {tstat(D.ex):+6.2f} {H.ex.mean():+7.3f}"
                  f" {F.ex.mean():+7.3f}  {'PASS' if not fails else 'fail: ' + ', '.join(fails)}")
    print(f"\n{passes} of 51 variants pass the gate.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "run":
        T = run()
        T.to_csv(DATA / "trades.csv.gz", index=False)
        report(T)
    else:
        print(__doc__)
