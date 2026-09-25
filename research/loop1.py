"""Edge loop, round 1 (pre-registered 25 Sep 2026, before running):
G  trend-aligned level entries: F2 levels (stop 1.0 ATR beyond), first rung only, keep a long only when the
   asset's own 14/28-day trend blend is positive and a short only when negative.
C  capitulation, 'catch the bottom': daily move beyond 2.5 sigma of the prior 30 days.
   C-early: enter next open against the move. C-confirm: wait for a daily close beyond the prior day's
   extreme (within 5 days), enter next open. Stop 0.25 daily ATR beyond the extreme since the event, 3R, 10 days.
N  NR7 bracket: after the narrowest daily range of 7, stop entries at that day's high and low, stop at the
   opposite side, 3R, 3 days.
Design: BTC to 2023-12-31, alts to 2024-12-31. Held out: after that, run once."""
import sys
import numpy as np, pandas as pd
import support as S, support_variants as V

TAKER, MAKER, SLIP = 0.0006, 0.0002, 0.0002
HOLD_SPLIT = {"BTC": "2024-01-01", "ETHUSDT": "2025-01-01", "SOLUSDT": "2025-01-01", "XRPUSDT": "2025-01-01"}

def summ(x):
    x = pd.Series(x).dropna()
    return f"n={len(x)} mean={x.mean():+.3f} t={x.mean()/x.std()*np.sqrt(len(x)):+.2f}" if len(x) > 5 else f"n={len(x)}"

def period(sym, t): return "held-out" if t >= pd.Timestamp(HOLD_SPLIT[sym]) else "design"

def resolve(h1, k0, side, entry, stop, tgt, hold_h, fee_in):
    risk = side * (entry - stop)
    if risk <= 0: return None
    L, H, C = h1.l.values, h1.h.values, h1.c.values; end = min(k0 + hold_h, len(C))
    for q in range(k0, end):
        if (side == 1 and L[q] <= stop) or (side == -1 and H[q] >= stop): return -1 - (fee_in + TAKER + SLIP) * entry / risk
        if q > k0 and ((side == 1 and H[q] >= tgt) or (side == -1 and L[q] <= tgt)): return 3 - (fee_in + MAKER) * entry / risk
    return side * (C[end - 1] - entry) / risk - (fee_in + TAKER) * entry / risk

def daily(h1):
    return h1.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()

def run_G():
    S.Market.levels = V.patched_levels(None, 1.0)
    T = pd.DataFrame([t for s in S.SYMS for side in (1, -1) for t in S.run(s, side, "H")])
    T = T[T.rung == 1].copy()
    blends = {}
    for s in S.SYMS:
        h1, _ = S.load(s); d = daily(h1).c
        b = (np.sign(d / d.shift(14) - 1) + np.sign(d / d.shift(28) - 1)) / 2; b.index = b.index + pd.Timedelta(days=1)
        blends[s] = b
    T["tsm"] = [blends[s].asof(t) for s, t in zip(T.sym, T.t)]
    T["per"] = [period(s, t) for s, t in zip(T.sym, T.t)]
    T["aligned"] = ((T.side == 1) & (T.tsm > 0)) | ((T.side == -1) & (T.tsm < 0))
    T["strong"] = ((T.side == 1) & (T.tsm == 1)) | ((T.side == -1) & (T.tsm == -1))
    print("G trend-aligned level entries (F2 levels, first rung):")
    for p in ("design", "held-out"):
        g = T[T.per == p]
        print(f"  {p}: all {summ(g.R)} | aligned {summ(g[g.aligned].R)} | strong {summ(g[g.strong].R)} | against {summ(g[~g.aligned].R)}")
        print(f"     strong longs {summ(g[g.strong & (g.side == 1)].R)} | strong shorts {summ(g[g.strong & (g.side == -1)].R)}")

def run_C():
    rows = []
    for s in S.SYMS:
        h1, _ = S.load(s); d = daily(h1); r = d.c.pct_change(); sig = r.rolling(30).std().shift(1)
        tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        pos = h1.index.searchsorted(d.index)
        for i in range(40, len(d) - 12):
            for side in (1, -1):
                if not ((side == 1 and r.iat[i] <= -2.5 * sig.iat[i]) or (side == -1 and r.iat[i] >= 2.5 * sig.iat[i])): continue
                ext = d.l.iat[i] if side == 1 else d.h.iat[i]
                # early: next day open
                k = pos[i + 1]; entry = h1.o.iat[k] * (1 + side * SLIP); stop = ext - side * 0.25 * atr.iat[i]
                R = resolve(h1, k, side, entry, stop, entry + side * 3 * side * (entry - stop), 240, TAKER)
                if R is not None: rows.append(dict(sym=s, t=d.index[i], side=side, kind="early", R=R))
                # confirm: first close beyond prior day's extreme within 5 days, enter next open
                for j in range(i + 1, min(i + 6, len(d) - 2)):
                    if (side == 1 and d.c.iat[j] > d.h.iat[j - 1]) or (side == -1 and d.c.iat[j] < d.l.iat[j - 1]):
                        ext2 = d.l.iloc[i:j + 1].min() if side == 1 else d.h.iloc[i:j + 1].max()
                        k = pos[j + 1]; entry = h1.o.iat[k] * (1 + side * SLIP); stop = ext2 - side * 0.25 * atr.iat[j]
                        R = resolve(h1, k, side, entry, stop, entry + side * 3 * side * (entry - stop), 240, TAKER)
                        if R is not None: rows.append(dict(sym=s, t=d.index[i], side=side, kind="confirm", R=R))
                        break
    T = pd.DataFrame(rows); T["per"] = [period(s, t) for s, t in zip(T.sym, T.t)]
    print("C capitulation / blow-off (2.5 sigma daily):")
    for kind in ("early", "confirm"):
        for side, lab in ((1, "buy the capitulation"), (-1, "sell the blow-off")):
            g = T[(T.kind == kind) & (T.side == side)]
            print(f"  {kind} {lab}: design {summ(g[g.per == 'design'].R)} | held-out {summ(g[g.per == 'held-out'].R)}")

def run_N():
    rows = []
    for s in S.SYMS:
        h1, _ = S.load(s); d = daily(h1); rng = d.h - d.l; nr7 = rng == rng.rolling(7).min()
        pos = h1.index.searchsorted(d.index)
        for i in range(10, len(d) - 5):
            if not nr7.iat[i]: continue
            hi, lo = d.h.iat[i], d.l.iat[i]; k0, k1 = pos[i + 1], pos[i + 2]
            for k in range(k0, k1):
                up, dn = h1.h.iat[k] >= hi, h1.l.iat[k] <= lo
                if up or dn:
                    side = 1 if up and not dn else (-1 if dn and not up else 0)
                    if side == 0: rows.append(dict(sym=s, t=d.index[i], R=-1 - 0.0016 * hi / (hi - lo))); break
                    entry = (max(hi, h1.o.iat[k]) if side == 1 else min(lo, h1.o.iat[k])) * (1 + side * SLIP); stop = lo if side == 1 else hi
                    R = resolve(h1, k, side, entry, stop, entry + side * 3 * side * (entry - stop), 72, TAKER)
                    if R is not None: rows.append(dict(sym=s, t=d.index[i], R=R))
                    break
    T = pd.DataFrame(rows); T["per"] = [period(s, t) for s, t in zip(T.sym, T.t)]
    print(f"N NR7 daily bracket: design {summ(T[T.per == 'design'].R)} | held-out {summ(T[T.per == 'held-out'].R)}")

if __name__ == "__main__":
    for f in sys.argv[1:]: {"G": run_G, "C": run_C, "N": run_N}[f]()
