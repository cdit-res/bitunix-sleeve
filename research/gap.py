"""Hypothesis A: weekend gap continuation on hourly bars. Stop-first when a bar spans stop and target."""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
ENTRY_COST, TP_COST, STOP_COST = 0.0008, 0.0002, 0.0006  # taker+slippage in, maker TP, taker stop/time exit

def hourly(sym: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{sym}_1h.csv.gz", index_col=0, parse_dates=True)

def ct(day: pd.Timestamp, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hm}", tz="America/Chicago").tz_convert("UTC").tz_localize(None)

def leg(d, i0, i1, side, entry, stop, tgt) -> float:
    """R for one leg from bar i0 to i1 (exclusive): stop first within a bar."""
    risk = abs(entry - stop)
    for j in range(i0, i1):
        if (d.l.iat[j] <= stop) if side == 1 else (d.h.iat[j] >= stop):
            return -1 - (ENTRY_COST + STOP_COST) * entry / risk
        if (d.h.iat[j] >= tgt) if side == 1 else (d.l.iat[j] <= tgt):
            return abs(tgt - entry) / risk - (ENTRY_COST + TP_COST) * entry / risk
    return side * (d.c.iat[i1 - 1] - entry) / risk - (ENTRY_COST + STOP_COST) * entry / risk

def trades(sym: str, start: str, end: str, min_gap: float = 0.005) -> pd.DataFrame:
    d = hourly(sym); rows = []
    for f in pd.date_range(start, end, freq="W-FRI"):
        t_fri = ct(f, "15:00"); t_sun = ct(f + pd.Timedelta(days=2), "17:00"); t_end = ct(f + pd.Timedelta(days=7), "16:00")
        if t_fri not in d.index or t_sun not in d.index: continue
        a = d.c.at[t_fri]; b = d.o.at[t_sun]; g = b / a - 1
        if abs(g) < min_gap: continue
        side = int(np.sign(g)); risk = abs(b - a)
        i0 = d.index.get_loc(t_sun); i1 = min(d.index.searchsorted(t_end), len(d))
        r1 = leg(d, i0, i1, side, b, a, b + side * risk)
        r3 = leg(d, i0, i1, side, b, a, b + side * 3 * risk)
        rows.append(dict(sym=sym, t=f, gap=g, A0=r1, A1=0.5 * r1 + 0.5 * r3))
    return pd.DataFrame(rows)

def summ(x: pd.Series) -> str:
    x = x.dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"

if __name__ == "__main__":
    import sys
    start, end = sys.argv[1], sys.argv[2]
    T = pd.concat([trades(s, start, end) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")])
    for s, g in T.groupby("sym"):
        print(f"{s}: A0 {summ(g.A0)} | A1 {summ(g.A1)}")
    alts = T[T.sym != "BTCUSDT"]
    half = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    print(f"ETH+SOL+XRP pooled: A0 {summ(alts.A0)} | A1 {summ(alts.A1)}")
    print(f"  first half: A1 {summ(alts[alts.t < half].A1)} | second half: A1 {summ(alts[alts.t >= half].A1)}")
    print(f"all four pooled: A0 {summ(T.A0)} | A1 {summ(T.A1)}")
