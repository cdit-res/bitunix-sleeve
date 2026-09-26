# bitunix-sleeve

Engine for the scheduled trade runs, and the research behind its rules.

## engine.py

Reads the repo's own data feed (the `data` branch, below) and falls back to static-klines for crypto if the feed is missing. A run:

```
curl -sSLO https://raw.githubusercontent.com/cdit-res/bitunix-sleeve/main/engine.py
python engine.py score state.csv
python engine.py scan
python engine.py verdict state.csv "K1=TAKE:reason" "K2=PASS:reason"
python engine.py html brief.json
```

| Command | Output |
|---|---|
| `scan [--equity 170] [--crypto-only]` | Regime, prices, the tested edges today, and the ten nearest eligible candidates (crypto on 4h, stocks on daily) with the full ticket: entry, stop, TP A (nearest dated level at 1.6:1 net or better), TP B (farthest dated level within 4 daily ATR), leverage and margin |
| `score state.csv` | Rescores every open row on 15m (crypto) or 1h (stocks) bars: fills, TP A, runner to breakeven, TP B, stop, expiry; prints the book, the paper record, the evaluator split (TAKE minus PASS) and live results; rolls rows resolved over three days ago into aggregates |
| `verdict state.csv ...` | Logs the run's TAKE and PASS verdicts on the scan's candidates and prints the ticket rows |
| `edges` | MAX-10 and trend components per coin, and their paper record since 28 September 2026 |
| `html brief.json` | The brief as phone-width HTML tables and plain text |

## data_feed/ and the `data` branch

A GitHub Actions job (`.github/workflows/data.yml`) fetches public data every ten minutes, when GitHub runs it, and publishes to the `data` branch: Binance spot 15m, 1h, 4h and daily bars with taker volume for 18 coins; Yahoo 1h and daily bars for 24 stock perp underlyings and VIX; and, daily, Yahoo history since listing for 147 stocks, ETFs, indices and commodities (research). Never the Bitunix API.

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
| tv_indicators.py | Harmonic XABCD patterns (standard ratio table, ZigZag P = 8) and LuxAlgo Trendlines with Breaks, 1h and 4h | Fail: harmonics -0.14R and -0.30R design; trendline breaks -0.13R (1h), +0.13R (4h, t 2.4) with a negative second half and held-out |
| range_deviation.py | Range deviation reclaim (sweep, then close back inside) and trend-aligned range fades | Fail: -0.29R to -0.46R in every set; BTC on 5m bars -0.58R |
| remaining_lines.py | Trend shorts, CPI/FOMC brackets, liquidation-spike fade, BTC-to-alt lead-lag | Fail. Trend shorts paid in 2018 (+1.35R) and 2022 (+0.30R) only; held-out -0.08R |
| orb.py | Opening-range breakout at the US open on BTC (5m and 15m) and alts (15m); trend shorts in a down market | Fail: ORB -0.21R to -0.32R; down-market shorts +0.45R design but -0.11R held-out |
| vb_pairs.py | Larry Williams volatility breakout (long, short); pair MAX-10 against BTC | Fail |
| max10_intraday.py | MAX-10 as a same-day trade, and with a pullback entry | About 0R: the edge accrues outside 08:00 to 21:00 UTC |
| stocks_intraday.py | US stocks on 5m bars (TQQQ, SPXL, SOXL and 2x single-stock ETFs as proxies, 2020-26): ORB-5, ORB-15, intraday momentum, overnight drift, MAX-10, trend long and short | Fail at Bitunix stock-perp costs: ORB-5 -0.55R, intraday momentum -0.20% a trade, trend shorts -0.29R |
| harness.py | The standard table: strategy, asset, period, trades, gross, costs, net, max DD, Sharpe, OOS, walk-forward | Written to results_table.csv |
| batch3a.py | The daily edges traded inside the ticket windows; disproof of MAX-10 and T1 | Windows fail (-0.01R to -0.09R); MAX-10 holds walk-forward and at double costs |
| batch3b.py | Pre-FOMC drift; RSI-2 dips; stock MAX-10 and trend with more ETFs | Fail |
| stocks_cfd.py | Stock CFDs on 33 years of daily data: turn of the month, short UVXY carry, trend and MAX-10 on single stocks, weekly reversal, RSI-2 and limit dips on indices, event gaps | RSI-2 index dips conditional; short UVXY positive but t 2.1; single-stock longs reflect survivorship |
| batch5_etf.py | Trend on index, sector, country and metal ETFs; overnight drift after down days | ETF trend passes at Bitunix funding but random entries match it (beta) |
| batch4_crypto.py | Taker-flow persistence, round-number fades, Asia-range bracket, turtle soup, funding filter, volatility scaling | Fail; improvements not adopted |
| placebo_crypto.py | The passing crypto edges against random or unconditional entries with the same exit | MAX-10 beats its placebo in every set; the trend ensemble does not |
| fetch_data.py | Rebuilds `data/` from the public sources | |

Rules for any change: net of fees, t of 3 or more, the same sign in both halves, a held-out period run once, fresh symbols, and a placebo the signal must beat. Count every variant tried. The full record is research/LEDGER.md and research/PROGRAMME.md.

Held-out rules: BTC, ETH, SOL and XRP after 2023 (or 2024 for 1h alt tests) are the time held-out; LINK, BNB, ADA, DOGE, AVAX and DOT are fresh coins never used to design a rule.

Data sources:

- github.com/ff137/bitstamp-btcusd-minute-data (BTC 1-minute, 2012 to date)
- github.com/finom/static-klines (10 majors)
- github.com/arkanoeth/Binance_Future_Prices and github.com/cryptobigbro/binance-BTCUSDT (older hourly data)
- github.com/supervik/historical-funding-rates-fetcher and github.com/ZuShen168/funding_rate_data (funding rates)
- github.com/piekstra/market-data (5-minute US leveraged ETF candles, 2020-26)
- macro_events.csv: CPI (bls.gov) and FOMC (federalreserve.gov) release dates, 2017-26
