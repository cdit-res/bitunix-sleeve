"""Standard evaluation table for every strategy (owner, 25 Sep 2026).

Columns: Strategy, Asset, Period, Trades, Gross expectancy, Costs, Net expectancy, Max DD, Sharpe,
OOS expectancy, Walk-forward result.

Input: one row per trade with sym, t (entry), t_exit, gross (R before costs), cost (R), net (= gross - cost),
and optionally variant (for walk-forward selection among pre-registered variants).
Definitions:
  Net expectancy     mean net R per trade over every trade (all symbols, all dates), with t.
  Max DD             deepest fall of cumulative net R, trades at 1R each, booked on their exit date.
  Sharpe             annualised from daily net R booked on exit dates (zero on days without exits);
                     sqrt(365) for crypto, sqrt(252) for stocks.
  OOS expectancy     mean net R on the held-out time for design symbols, and on fresh symbols (all dates).
  Walk-forward       with several variants: each year uses the variant with the best mean net R over all
                     earlier years; result = that chained series. With one variant: years positive of years
                     traded, and the worst year.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results_table.csv"
COLS = ["Strategy", "Asset", "Period", "Trades", "Gross expectancy", "Costs", "Net expectancy", "Max DD", "Sharpe",
        "OOS expectancy", "Walk-forward result"]


def tstat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def fmt(x: pd.Series) -> str:
    x = pd.Series(x).dropna()
    return f"{x.mean():+.3f}R (t {tstat(x):+.1f}, n {len(x)})" if len(x) > 2 else f"n {len(x)}"


def max_dd(T: pd.DataFrame) -> float:
    daily = T.groupby(T.t_exit.dt.normalize()).net.sum().sort_index().cumsum()
    return float((daily - daily.cummax()).min()) if len(daily) else float("nan")


def sharpe(T: pd.DataFrame, ann: int) -> float:
    s = T.groupby(T.t_exit.dt.normalize()).net.sum()
    if len(s) < 20:
        return float("nan")
    freq = "D" if ann == 365 else "B"
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=freq), fill_value=0.0)
    return float(s.mean() / s.std() * np.sqrt(ann)) if s.std() > 0 else float("nan")


def walk_forward(T: pd.DataFrame, burn_years: int = 3) -> str:
    T = T.assign(yr=T.t.dt.year)
    if "variant" not in T or T.variant.nunique() < 2:
        by = T.groupby("yr").net.mean()
        return f"{int((by > 0).sum())} of {len(by)} years positive; worst {by.min():+.2f}R ({by.idxmin()})" if len(by) else "n/a"
    years = sorted(T.yr.unique()); chain = []
    for y in years[burn_years:]:
        past = T[T.yr < y]
        best = past.groupby("variant").net.mean().idxmax()
        chain.append(T[(T.yr == y) & (T.variant == best)])
    W = pd.concat(chain) if chain else T.iloc[:0]
    by = W.groupby("yr").net.mean()
    return f"{fmt(W.net)}; {int((by > 0).sum())} of {len(by)} years positive" if len(W) else "n/a"


def evaluate(name: str, T: pd.DataFrame, *, asset: str, held_from: str, fresh: set[str] | tuple = (),
             ann: int = 365, primary: str | None = None, save: bool = True) -> dict:
    """T: trades. primary: the variant reported in the headline columns (others feed walk-forward only)."""
    T = T.copy()
    T["t"] = pd.to_datetime(T.t); T["t_exit"] = pd.to_datetime(T.t_exit)
    H = T[T.variant == primary] if primary is not None and "variant" in T else T
    fresh = set(fresh); design_syms = ~H.sym.isin(fresh)
    ho = H[design_syms & (H.t >= held_from)]; fr = H[H.sym.isin(fresh)]
    oos = f"held-out {fmt(ho.net)}" + (f"; fresh {fmt(fr.net)}" if len(fresh) else "")
    row = {"Strategy": name, "Asset": asset,
           "Period": f"{H.t.min():%Y-%m} to {H.t_exit.max():%Y-%m}" if len(H) else "n/a",
           "Trades": len(H), "Gross expectancy": f"{H.gross.mean():+.3f}R" if len(H) else "n/a",
           "Costs": f"{H.cost.mean():.3f}R" if len(H) else "n/a", "Net expectancy": fmt(H.net),
           "Max DD": f"{max_dd(H):.1f}R" if len(H) else "n/a",
           "Sharpe": f"{sharpe(H, ann):.2f}" if len(H) else "n/a", "OOS expectancy": oos,
           "Walk-forward result": walk_forward(T[~T.sym.isin(fresh)])}
    if save:
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
        old = old[old.Strategy != name]
        pd.concat([old, pd.DataFrame([row])]).to_csv(OUT, index=False)
    return row


def gate(T: pd.DataFrame, held_from: str, fresh: set[str] | tuple = ()) -> dict:
    """The standing gate on the design set: t >= 3, both halves the same sign, held-out and fresh above zero."""
    fresh = set(fresh); core = T[~T.sym.isin(fresh)]; des = core[core.t < held_from]
    mid = des.t.min() + (des.t.max() - des.t.min()) / 2 if len(des) else None
    h1, h2 = (des[des.t < mid].net, des[des.t >= mid].net) if mid is not None else (pd.Series(dtype=float),) * 2
    ho, fr = core[core.t >= held_from].net, T[T.sym.isin(fresh)].net
    ok = (tstat(des.net) >= 3 and np.sign(h1.mean()) == np.sign(h2.mean()) == np.sign(des.net.mean()) > 0
          and ho.mean() > 0 and (fr.mean() > 0 if len(fresh) else True))
    return {"design": fmt(des.net), "halves": f"{fmt(h1)} / {fmt(h2)}", "held_out": fmt(ho),
            "fresh": fmt(fr) if len(fresh) else "n/a", "pass": bool(ok)}


def show(rows: list[dict]) -> None:
    pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 60)
    print(pd.DataFrame(rows)[COLS].to_string(index=False))
