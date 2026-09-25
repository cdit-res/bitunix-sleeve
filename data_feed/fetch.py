"""Public market data for the sleeve, fetched on GitHub Actions and published to the `data` branch.

The Claude sandbox reaches GitHub but not Yahoo, Binance or Coinbase, so this job fetches from those public
sources and the runs and research read plain CSVs from raw.githubusercontent.com. Never the Bitunix API.

live/      crypto 15m, 1h, 4h, 1d (Binance spot via data-api.binance.vision, Coinbase as fallback) and
           stock perp underlyings 1h and 1d (Yahoo), refreshed every run.
research/  daily history since listing for the Bitunix stock, ETF, index, volatility and commodity universe
           (Yahoo, split and dividend adjusted), plus 1h for two years on the liquid core. Refreshed daily.

Usage: python fetch.py OUT_DIR [--research]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "LINK", "BNB", "ADA", "DOGE", "AVAX", "DOT", "LTC", "NEAR", "INJ",
          "TAO", "FET", "RUNE", "SUI", "HYPE"]
CRYPTO_INTERVALS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
KEEP_DAYS = {"15m": 20, "1h": 200, "4h": 400, "1d": 1600}

US = """AAOI AAPL ADBE ALAB AMC AMD AMZN ANET APLD APP ARM ASML ASTS AVGO AXTI BABA BE BX CAT COHR COIN CRCL
CRM CRWV CSCO DELL DIS DJT DKNG FLEX FLNC GLW GOOGL GPRO GS GTLB HD HIMS HOOD IBM INTC IONQ IREN KLAC KO LITE
LLY LRCX MDB META MRK MRNA MRVL MSFT MSTR MU NBIS NFLX NKE NOK NOW NVDA NVO ONDS ORCL PATH PEP PLTR PYPL QBTS
QCOM RDDT RIVN RKLB SHOP SMCI SNDK SNOW SOFI SONY TEAM TEM TSLA TSM TTWO TXN UBER V WDC WEN WMT BRK-B""".split()
ETF = """SPY QQQ IWM DIA SMH XLE XBI GDX EWJ EWT EWY EWZ URNM BITO SOXL SOXS TQQQ SQQQ TMF TBT UVXY SVXY VXX
NVDL TSLL KORU TLT GLD SLV USO XLK XLF""".split()
INDEX = ["^GSPC", "^NDX", "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^TNX", "DX-Y.NYB"]
COMMOD = ["CL=F", "BZ=F", "GC=F", "SI=F", "HG=F", "NG=F", "PL=F", "PA=F"]
ASIA = ["005930.KS", "000660.KS", "0700.HK", "1810.HK", "1211.HK", "3690.HK", "9988.HK"]
CORE_1H = """SPY QQQ IWM SMH NVDA TSLA AAPL MSFT AMZN META GOOGL AMD AVGO MSTR COIN PLTR NFLX MU TSM ORCL HOOD
UVXY TQQQ SOXL""".split()

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (bitunix-sleeve data feed)"


def _get(url: str, params: dict | None = None, tries: int = 4):
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 451):
                return None  # region block: use the fallback source
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def binance(sym: str, interval: str, days: int) -> pd.DataFrame | None:
    url = "https://data-api.binance.vision/api/v3/klines"
    end = int(time.time() * 1000); start = end - days * 86_400_000; rows: list = []
    while start < end:
        page = _get(url, {"symbol": f"{sym}USDT", "interval": interval, "startTime": start, "limit": 1000})
        if not page:
            break
        rows += page
        nxt = page[-1][0] + 1
        if len(page) < 1000 or nxt <= start:
            break
        start = nxt
    if not rows:
        return None
    d = pd.DataFrame([r[:6] + [r[9]] for r in rows], columns=["t", "o", "h", "l", "c", "v", "taker_buy_v"])
    d["t"] = pd.to_datetime(d.t, unit="ms")
    return d.drop_duplicates("t").set_index("t").astype(float)


def coinbase(sym: str, interval: str, days: int) -> pd.DataFrame | None:
    gran = CRYPTO_INTERVALS[interval]
    if interval == "4h":  # Coinbase has no 4h candles: build them from 1h
        h = coinbase(sym, "1h", days)
        return None if h is None else h.resample("4h").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    url = f"https://api.exchange.coinbase.com/products/{sym}-USD/candles"
    end = datetime.now(timezone.utc); rows: list = []
    start_all = end - pd.Timedelta(days=days)
    while end > start_all:
        start = max(start_all, end - pd.Timedelta(seconds=gran * 300))
        page = _get(url, {"granularity": gran, "start": start.isoformat(), "end": end.isoformat()})
        if not page:
            break
        rows += page; end = start
        time.sleep(0.2)
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["t", "l", "h", "o", "c", "v"])
    d["t"] = pd.to_datetime(d.t, unit="s")
    return d.drop_duplicates("t").set_index("t").sort_index()[["o", "h", "l", "c", "v"]].astype(float)


def merge(path: Path, new: pd.DataFrame, keep_days: int) -> pd.DataFrame:
    """Append new bars to the stored file, newest values winning, trimmed to the retention window."""
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    cut = new.index.max() - pd.Timedelta(days=keep_days)
    return new[new.index >= cut]


def live_crypto(out: Path, manifest: dict) -> None:
    d0 = out / "live" / "crypto"; d0.mkdir(parents=True, exist_ok=True)
    for sym in CRYPTO:
        for iv in CRYPTO_INTERVALS:
            path = d0 / f"{sym}_{iv}.csv"
            days = 3 if path.exists() and iv != "1d" else KEEP_DAYS[iv]
            src, d = "binance", binance(sym, iv, days)
            if d is None or d.empty:
                src, d = "coinbase", coinbase(sym, iv, days)
            if d is None or d.empty:
                manifest["missing"].append(f"{sym} {iv}"); continue
            d = merge(path, d, KEEP_DAYS[iv]); d.to_csv(path, float_format="%.10g")
            manifest["crypto"][f"{sym}_{iv}"] = {"source": src, "last_bar": str(d.index[-1]), "rows": len(d)}


def yahoo(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for k in range(0, len(tickers), 25):
        chunk = tickers[k:k + 25]
        try:
            raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception:
            continue
        for t in chunk:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                d.columns = ["o", "h", "l", "c", "v"]
                if d.index.tz is not None:
                    d.index = d.index.tz_convert("UTC").tz_localize(None)
                if len(d):
                    out[t] = d
            except Exception:
                continue
        time.sleep(1.0)
    return out


def safe(t: str) -> str:
    return t.replace("^", "IDX_").replace("=", "_").replace(".", "_")


def live_stocks(out: Path, manifest: dict) -> None:
    d0 = out / "live" / "stocks"; d0.mkdir(parents=True, exist_ok=True)
    for iv, period in (("1h", "60d"), ("1d", "2y")):
        got = yahoo(CORE_1H + ["^VIX", "^VIX3M"], period, iv)
        for t, d in got.items():
            d.to_csv(d0 / f"{safe(t)}_{iv}.csv", float_format="%.10g")
            manifest["stocks"][f"{t}_{iv}"] = {"last_bar": str(d.index[-1]), "rows": len(d)}


def research(out: Path, manifest: dict) -> None:
    d0 = out / "research" / "daily"; d1 = out / "research" / "hourly"
    d0.mkdir(parents=True, exist_ok=True); d1.mkdir(parents=True, exist_ok=True)
    for t, d in yahoo(US + ETF + INDEX + COMMOD + ASIA, "max", "1d").items():
        d.to_csv(d0 / f"{safe(t)}.csv.gz", float_format="%.10g")
        manifest["research_daily"][t] = [str(d.index[0].date()), str(d.index[-1].date()), len(d)]
    for t, d in yahoo(CORE_1H, "730d", "1h").items():
        d.to_csv(d1 / f"{safe(t)}.csv.gz", float_format="%.10g")
        manifest["research_hourly"][t] = [str(d.index[0]), str(d.index[-1]), len(d)]


def main(argv: list[str]) -> None:
    out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({"updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "missing": [],
                     "crypto": {}, "stocks": {}})
    manifest.setdefault("research_daily", {}); manifest.setdefault("research_hourly", {})
    live_crypto(out, manifest)
    live_stocks(out, manifest)
    if "--research" in argv:
        manifest["research_daily"], manifest["research_hourly"] = {}, {}
        research(out, manifest)
        manifest["research_updated_utc"] = manifest["updated_utc"]
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in manifest.items()}))


if __name__ == "__main__":
    main(sys.argv)
