"""N (pre-registered in loop1.py, unchanged): NR7 daily bracket. Broad check on 10 majors, daily bars built
from 4h, execution on 4h bars (stop first; a bar touching entry and stop counts as a loss). 3R, 3-day limit.
Design to 2023-12-31, held out 2024-01-01 onward."""
import numpy as np, pandas as pd
from pathlib import Path
DATA = Path(__file__).resolve().parent.parent / "data"
SYMS = ("BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT")
TAKER, MAKER, SLIP = 0.0006, 0.0002, 0.0002
def summ(x):
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean()/x.std()*np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"
rows = []
for s in SYMS:
    h4 = pd.read_csv(DATA / f"{s}_4h.csv.gz", index_col=0, parse_dates=True)
    d = h4.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    rng = d.h - d.l; nr7 = rng == rng.rolling(7).min(); pos = h4.index.searchsorted(d.index)
    H, L, O, C = h4.h.values, h4.l.values, h4.o.values, h4.c.values
    for i in range(10, len(d) - 5):
        if not nr7.iat[i]: continue
        hi, lo = d.h.iat[i], d.l.iat[i]; k0, k1 = pos[i + 1], pos[i + 2]
        for k in range(k0, k1):
            up, dn = H[k] >= hi, L[k] <= lo
            if not (up or dn): continue
            if up and dn:
                rows.append(dict(sym=s, t=d.index[i], R=-1 - (TAKER * 2 + SLIP) * hi / (hi - lo))); break
            side = 1 if up else -1
            entry = (max(hi, O[k]) if side == 1 else min(lo, O[k])) * (1 + side * SLIP)
            stop = lo if side == 1 else hi; risk = side * (entry - stop); tgt = entry + side * 3 * risk
            R = None; end = min(k + 18, len(C))
            for q in range(k, end):
                if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop): R = -1 - (TAKER * 2 + SLIP) * entry / risk; break
                if q > k and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)): R = 3 - (TAKER + MAKER + SLIP) * entry / risk; break
            if R is None: R = side * (C[end - 1] - entry) / risk - (TAKER * 2 + SLIP) * entry / risk
            rows.append(dict(sym=s, t=d.index[i], side=side, R=R, stop_pct=risk / entry * 100)); break
T = pd.DataFrame(rows); des = T[T.t < "2024-01-01"]; ho = T[T.t >= "2024-01-01"]
print(f"NR7 bracket, 10 majors: design {summ(des.R)} | held-out {summ(ho.R)} | median stop {T.stop_pct.median():.2f}%")
print("  design halves: " + summ(des[des.t < '2021-01-01'].R) + " | " + summ(des[des.t >= '2021-01-01'].R))
print("  by symbol (design / held-out): " + "; ".join(f"{s[:-4]} {des[des.sym==s].R.mean():+.2f}/{ho[ho.sym==s].R.mean():+.2f}" for s in SYMS))
print("  longs " + summ(T[T.side==1].R) + " | shorts " + summ(T[T.side==-1].R))
