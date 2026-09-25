"""Rebuild the research data in ../data from public sources.

  BTC_bitstamp_1h.csv.gz   Bitstamp BTC/USD 1-minute bars (ff137/bitstamp-btcusd-minute-data), resampled to 1h
  {SYM}_1h.csv.gz          static-klines 1h bars from 2022 for the 10 majors
  {SYM}_4h.csv.gz          static-klines 4h bars from 2017 for the 10 majors

bt.py, edge.py and regime.py also need github.com/arkanoeth/Binance_Future_Prices and
github.com/cryptobigbro/binance-BTCUSDT cloned beside this repo.
Usage: python fetch_data.py [btc] [majors]
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402

DATA = ROOT / "data"
FF137 = "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data/"
BTC_FILES = ("historical/btcusd_bitstamp_1min_2012-2025.csv.gz", "updates/btcusd_bitstamp_1min_latest.csv")


def download(url: str, dest: Path) -> Path:
    """Resumable download: large raw files often arrive cut short through the proxy."""
    ctx = engine._ctx()
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), context=ctx).headers["Content-Length"])
    for _ in range(50):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= size:
            return dest
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=300) as r, dest.open("ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError:
            continue
    raise RuntimeError(f"incomplete download: {url}")


def btc_hourly() -> None:
    parts = []
    for name in BTC_FILES:
        f = download(FF137 + name, DATA / Path(name).name)
        d = pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close"])
        d.index = pd.to_datetime(d.pop("timestamp").astype("int64"), unit="s")
        parts.append(d.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"}))
    m = pd.concat(parts)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    h = m.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    h[h.index >= "2016-06-01"].to_csv(DATA / "BTC_bitstamp_1h.csv.gz")


def majors() -> None:
    for sym in engine.COVERED:
        for interval, start in (("1h", "2022-01-01"), ("4h", "2017-01-01")):
            d = engine.klines(sym, interval, start)
            d.index.name = "t"
            d.to_csv(DATA / f"{sym}_{interval}.csv.gz")
            print(sym, interval, len(d), d.index.min(), d.index.max())


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    jobs = sys.argv[1:] or ["btc", "majors"]
    if "btc" in jobs:
        btc_hourly()
    if "majors" in jobs:
        majors()
