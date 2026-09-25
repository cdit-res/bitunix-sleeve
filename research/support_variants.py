"""Pre-registered rescue variants for hypothesis F (25 Sep 2026), longs and shorts, horizontal ladder only.
F2 wide stop: entry at the level, stop 1.0 ATR beyond the zone edge.
F3 sweep entry: entry 0.25 ATR beyond the zone edge (where tight stops sit), stop 1.25 ATR beyond the edge."""
import sys
import numpy as np, pandas as pd
import support as S

def patched_levels(entry_off: float, stop_off: float):
    base = S.Market.levels
    def levels(self, i, side):
        out = []
        for lvl, stop in base(self, i, side):
            edge = stop + side * S.BUF_H * self.atr[i]  # recover the zone edge
            a = self.atr[i]
            e = lvl if entry_off is None else edge - side * entry_off * a
            out.append((e, edge - side * stop_off * a))
        out.sort(key=lambda x: -x[0] * side)
        return out
    return levels

if __name__ == "__main__":
    tr = S.btc_trend()
    for name, eo, so in (("F2 wide stop (entry at level, stop 1.0 ATR beyond edge)", None, 1.0),
                         ("F3 sweep entry (entry 0.25 ATR beyond edge, stop 1.25 ATR beyond)", 0.25, 1.25)):
        S.Market.levels = patched_levels(eo, so)
        T = pd.DataFrame([t for s in S.SYMS for side in (1, -1) for t in S.run(s, side, "H")])
        T["btc"] = [tr.asof(x) for x in T.t]; T = T[T.t < "2025-01-01"]
        half = T.t.min() + (T.t.max() - T.t.min()) / 2
        print(f"\n{name}")
        for side in (1, -1):
            g = T[T.side == side]
            print(f"  {'longs' if side == 1 else 'shorts'}: all {S.summ(g.R)} | halves {S.summ(g[g.t < half].R)} ; {S.summ(g[g.t >= half].R)} | median stop {g.stop_pct.median():.2f}%")
            print("     by rung: " + " | ".join(f"{r}: {S.summ(g[g.rung == r].R)}" for r in (1, 2, 3)) + " | by symbol: " + " | ".join(f"{s}: {S.summ(g[g.sym == s].R)}" for s in S.SYMS))
        import importlib; importlib.reload(S); tr = S.btc_trend()
