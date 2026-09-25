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
| `python engine.py gap` | BTC weekend gap status (paper only) |
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
| fetch_data.py | Rebuilds `data/` from the public sources | |

Rules for any change: net of fees, t of 3 or more, the same sign in both halves, and a held-out period run once. Count every variant tried.

Data sources:

- github.com/ff137/bitstamp-btcusd-minute-data (BTC 1-minute, 2012 to date)
- github.com/finom/static-klines (10 majors)
- github.com/arkanoeth/Binance_Future_Prices and github.com/cryptobigbro/binance-BTCUSDT (older hourly data)
