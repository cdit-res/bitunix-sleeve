# Research programme: an intraday edge for the Bitunix sleeve

## Charter (owner, 25 September 2026)

- **Objective.** Find a repeatable intraday trading edge within these constraints:
  - leverage of 60x to 80x;
  - Bitunix USDT perps, crypto and stock;
  - maker-first tickets placed by hand at the 08:30 and 20:00 UK runs;
  - two ticket windows a day.
- **Constraints stay as given.** Do not replace them with investment advice.
- **Never declare success in sample.** Try to disprove every candidate with:
  - out-of-sample tests and walk-forward tests;
  - regime analysis;
  - realistic costs.
- **Keep this ledger** (LEDGER.md, ledger.csv, build_ledger.py) so results persist across sessions.
- **When a hypothesis fails,** generate the next one.
- **Stop only** with a robust edge shown under realistic conditions, or with the defensible avenues exhausted and the reason explained.

## What the constraints imply

| Item | Figure |
|---|---|
| Leverage and stop size | Under the sleeve's rule (a stop-out costs about 55% of margin), 60x to 80x means stops of about 0.7% to 0.9% |
| Round-trip cost | 0.04% maker/maker; 0.10% maker in, taker stop; 0.16% taker/taker |
| Cost in R | At 0.8% stops, about 0.05R on winners and 0.125R on losers |
| Gross edge needed | About 0.1R a trade before costs |
| Holding window | Ticket to ticket: about 11.5h in the morning window (07:30 to 19:00 UTC in summer) and 12.5h in the evening window (19:00 to 07:30 UTC) |

## Standing gate

- Net of fees and slippage; longs pay funding at 0.03% a day.
- Design t of 3 or more.
- The same sign in both halves.
- Positive on the time held-out, run once.
- Positive on coins or symbols never used in design.
- Every variant is counted.
- Candidates that pass then face:
  - walk-forward by year;
  - regime splits: trend, volatility, bull and bear years;
  - double costs;
  - a placebo: the same exit on random entry days, which the signal must beat in every set (added 26 September 2026);
  - four weeks of paper trading before live tickets.

## Status (26 September 2026)

See LEDGER.md (83 entries) and results_table.csv (the standard evaluation table) for the full record.

| Result | What it is | Verdict |
|---|---|---|
| T03 MAX-10 (crypto) | Long the morning after a 10-day closing high, 1 ATR stop, 24h hold | Genuine entry edge: beats a long-every-day placebo by +0.09R, +0.09R and +0.06R (design, held out, fresh); survives double costs. Paper from 28 Sep |
| SC6 RSI-2 dips (index ETFs) | Close above SMA200 and RSI(2) under 10; exit above SMA5 or at 10 days | Conditional: passes at Bitunix funding (0.0035% an interval), beats its placebo in every set, fails at double fees. Watch list |
| T01 trend ensemble (crypto) | Nine Donchian lookbacks, trailing midpoint stop | Positive, but random entry days with the same exit do as well: crypto drift plus the exit, not the entry. Paper as beta |
| SC9a trend ensemble (ETFs) | The same on index, sector, country and metal ETFs | Positive at Bitunix funding, same finding as T01: drift plus exit |
| Everything intraday | Every construction in the ledger with a hold of one window or less | None passes: at stops of 0.7% to 0.9% costs run at roughly 0.05R to 0.15R a trade and no signal has that much gross edge |

Why intraday fails, in one line: with a stop of b, a drift edge is worth about b times mu over sigma squared in R, so at b = 0.75% the conditional drift would need a daily Sharpe near 0.6 (an annualised 11) just to cover fees. Nothing on public OHLC, taker volume, funding or the macro calendar comes close.

## Avenues

| Avenue | State |
|---|---|
| Price-pattern intraday rules on liquid crypto (levels, ranges, breakouts, reversals, harmonics, trendlines, sessions, round numbers, Asia bracket) | Exhausted, 2016-26 |
| Order flow from taker volume (Binance 1h, 2022-26) | Exhausted at the window horizon |
| Intraday and overnight stock rules at Bitunix costs | Exhausted on 5m ETF proxies (2020-26) and 33 years of daily index data |
| Single-stock long rules | Blocked by survivorship: the Bitunix list is today's winners, so random entries look as good |
| Daily continuation and trend | MAX-10 genuine; trend is beta |
| Short volatility carry (UVXY) | Positive in 10 of 13 years, t 2.1: under the gate |
| Funding, open interest and liquidations over time | Blocked by data: collect forward (feed can add Hyperliquid) |
| The evaluator's judgement | Only measurable live: TAKE minus PASS, now logged on every eligible row |

## Swing search, 23 October 2026

The owner asked for better swing edges than MAX-10. Five constructions with the longest published records were pre-registered and run once (swing_search.py; ledger SW1 to SW6). None passed.

| Test | Held-out | Why it failed |
|---|---|---|
| Diversified time-series momentum, long and short | SR -0.13 | Long-only on the same markets beats it; the signs add nothing |
| Commodity cross-sectional momentum | SR -0.16 | No signal gross of costs (SR +0.15) |
| Crypto cross-sectional momentum | SR -1.48 | Reverses after 2023 |
| Cross-asset relative momentum | SR +0.16 | Random picks from the same positive-momentum set do as well |
| Volatility-managed SPY and QQQ | alpha t +0.88 | Alpha over constant exposure never reaches t 2; negative on fresh ETFs |

What this adds to the record: on perps, any hold of weeks pays funding on the whole notional, about 3.8% a year on tradfi at the assumed 0.0105% a day and about 11% a year on crypto longs at 0.03% a day. That takes roughly half the Sharpe of a slow book, so a position strategy has to beat simply holding the same markets by more than the funding. Nothing tested here does. MAX-10 remains the only entry edge: a one-day hold pays little funding.

## Swing batches 2 and 3, 23 October 2026

| Test | Result | Verdict |
|---|---|---|
| Month-end rebalancing front-run (SPY; SPY minus TLT) | +0.47% and +0.62% an event in design (t 1.6, 1.4); beats every mid-month window; negative since March 2023 | fail |
| Option-expiration week | flat in design, negative held out | fail |
| RSI-2 index dips entered at the next open | +0.06R (t 1.4); the close entry earns +0.14R | fail: SC6 needs a close entry |
| MAX-10 held as a swing (SW10) | +0.45R design (t 3.6), +0.30R held out, +0.40R fresh (t 4.1); beats its placebo by about +0.25R in every set | **PASS, promoted** |
| SW10 on commodities; on index ETFs | +0.06R, held out -0.06R; design -0.19R | fail |

SW10 keeps MAX-10's entry and replaces the one-day exit. Checks before promotion (swing_b3_checks.py): beats its placebo in every BTC regime and volatility state; every exit setting from two to five closes and five to twenty days beats its placebo in every set; filled at 08:00 UTC it keeps +0.14R over placebo (2022 on); in a book capped at three rows of 4% it made +25.8R a year with a daily Sharpe of 1.14, against +12.1R and 0.73 for the one-day hold, with a worst drawdown of -75% at 4% a row (-45% at 2%). Weak spot: 2025 (-0.07R a trade, -4.0R for the capped book) and the paper month since 28 September (three trades, -3.25R).

## Swing batch 4, 23 October 2026

Nothing adopted. The volume filter (SW16) and pyramid add-ons (SW18) improve the average trade but not the capped book. The short mirror of SW10 fails (SW17), so the book has no bear-market counterpart. Leveraged-ETF structures (SW14, SW15) fail at the assumed funding; the hedge against L x the underlying earns gross, but adjusted closes credit the long leg with dividends and coupons a perp does not pay, so it stays a lead until Bitunix funding on QQQ, TQQQ, TLT and TMF is known.

## Swing batch 5, 23 October 2026

SW10's rule on weekly bars (SW19 to SW21). Crypto beats its placebo in every set by +0.12R to +0.53R, with stops near 15% (about 3x), but at about three trades a coin a year it reaches only t 2.1 in design: a watch candidate, not a rule. Index ETFs and commodities fail.

Where the search stands after 21 swing and position tests: one adopted rule (SW10), two refinements that improve the trade but not the book, and one weekly candidate short of power. The public daily price data for this universe looks close to exhausted; what could add more is data not yet in hand: Bitunix funding on the leveraged-ETF and index perps (SW15), and a live record for SW19.

