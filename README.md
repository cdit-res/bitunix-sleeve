# bitunix-sleeve

Engine for the scheduled trade runs, and the research behind its rules.

## engine.py

Data comes from the public static-klines repo (Binance spot, 10 majors) through raw GitHub files. A run fetches it with:

```
curl -sSLO https://raw.githubusercontent.com/cdit-res/bitunix-sleeve/main/engine.py
python engine.py regime
```

| Command | Output |
|---|---|
| `python engine.py regime` | Market trend (BTC daily close against EMA50 and its five-day slope) and each coin's volatility state |
| `python engine.py levels` | Tested supports and resistances (2+ pivot touches within 0.5 ATR over 120 4h bars, unbroken on closes) and the latest trendline, with the stop 1.0 ATR beyond |
| `python engine.py candidates` | Pullback, failed-move and bracket candidates on 4h bars, with flags |
| `python engine.py edges` | The two tested edges: MAX-10 action per coin, and each trend component's state and trailing stop |
| `python engine.py gap` | Retired: CME bitcoin futures trade 24/7 since 29 May 2026 |
| `python engine.py score tickets.csv` | Scores split tickets on 15m bars, taking the stop first when one bar spans both stop and target |

Symbols default to BTC, ETH, SOL, XRP and LINK; pass others as a comma list, e.g. `python engine.py levels BTCUSDT,ETHUSDT`. Covered: BTC, ETH, SOL, XRP, LINK, BNB, ADA, DOGE, AVAX, DOT. Guaranteed stops exist only on BTC, ETH, SOL and XRP.

## research/

Each script's header records its settings, fixed before it was run.

| Script | Test | Result |
|---|---|---|
| bt.py, edge.py | Three structures on 4h bars, 20 pairs 2022-23 and BTC 2017-19, plus patterns and exits | About 0R after fees; stops under 0.5% lose about 0.66R per trade |
| regime.py | Regime cells | The low-vol pullback cell failed out of sample on BTC 2019-26 |
| gap.py | Weekend gap continuation | BTC weakly positive since 2022; alts negative on the held-out period |
| support.py | Tested support or trendline, stop just beyond, re-enter at the next support | Longs -0.26R (t -5.2), shorts -0.41R; held-out 2025-26 longs -0.45R; re-entries worse |
| support_variants.py | Same levels with the stop 1.0 ATR beyond, or entry at the sweep | Wide stops cut the loss by about 0.2R in design and held-out; still no edge on their own |
| loop1.py | Trend-aligned levels, capitulation and blow-off entries, NR7 daily bracket | No edge: aligned levels -0.10R; early capitulation longs -0.52R held-out; NR7 +0.09R held-out (t 1.25) |
| nr7_broad.py | NR7 bracket on all 10 majors | Design +0.08R (t 2.77) but second half -0.03R and held-out -0.03R: fails |
| range_grid.py | Fade the edges of a tested 1h range (grid), target mid or far edge | Fail: -0.43R (t -7.6) design, -0.38R held-out, -0.26R on six fresh coins |
| range_break.py | Trade the break of the same ranges | Fail: -1.1R with the stop inside, about 0R with the stop at mid |
| funding_extremes.py | Fade funding extremes (Binance BTC and ETH 2020-23; eight coins 2025-26) | Fail: -0.01R design, -0.12R held-out |
| trend_ensemble.py | T1 multi-horizon trend (5 to 360-day closing highs, trailing midpoint stop, long only); T3 MAX-10 next-day continuation | **Pass.** T1 daily Sharpe 1.53 design, 1.49 held-out, 1.18 fresh (with the 0.5% stop rail). MAX-10 +0.11R (t 4.7), +0.08R, +0.06R |
| max10_checks.py | MAX-10 robustness: per coin and year, taker entry, no stop, lookbacks 5 to 30, short side | Holds at every lookback and with taker entry; 12 of 15 years positive; shorts lose |
| trend_subset.py | Trend with three lookbacks (20, 60, 150) for manual use | Holds: daily Sharpe 1.03 design, 1.00 held-out, 0.90 fresh |
| fetch_data.py | Rebuilds `data/` from the public sources | |

Rules for any change: net of fees, t of 3 or more, the same sign in both halves, and a held-out period run once. Count every variant tried.

Held-out rules: BTC, ETH, SOL and XRP after 2023 (or 2024 for 1h alt tests) are the time held-out; LINK, BNB, ADA, DOGE, AVAX and DOT are fresh coins never used to design a rule.

Data sources:

- github.com/ff137/bitstamp-btcusd-minute-data (BTC 1-minute, 2012 to date)
- github.com/finom/static-klines (10 majors)
- github.com/arkanoeth/Binance_Future_Prices and github.com/cryptobigbro/binance-BTCUSDT (older hourly data)
- github.com/supervik/historical-funding-rates-fetcher and github.com/ZuShen168/funding_rate_data (funding rates)
