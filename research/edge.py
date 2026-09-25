"""Edge search. Rules fixed in advance; judged on 2022 (design), 2023 Q1 and BTC 2017-19 (holdouts)."""
from pathlib import Path
import numpy as np, pandas as pd
from bt import load, ind

FEE_M, FEE_T = 0.0002, 0.0006

def htf(d4: pd.DataFrame) -> pd.Series:
    """Daily trend known at each 4h bar: +1 if last completed daily close > daily EMA50 and EMA50 rising, -1 mirror, 0 otherwise."""
    dd = d4.resample("1D").agg({"close": "last"}).dropna()
    e = dd.close.ewm(span=50, adjust=False).mean()
    s = np.where((dd.close > e) & (e > e.shift(5)), 1, np.where((dd.close < e) & (e < e.shift(5)), -1, 0))
    s = pd.Series(s, index=dd.index + pd.Timedelta(days=1))  # usable from next day
    return s.reindex(d4.index, method="ffill").fillna(0).values

def manage(d, i, side, entry, stop, fee_rt, mode: str) -> tuple[float, int, float]:
    """Return (R net, exit bar, MFE in R). mode: fixed3 | trail (half at 2R, BE, trail 10-bar extreme)."""
    risk = abs(entry - stop)
    H, L, C = d.high.values, d.low.values, d.close.values
    fee_r = fee_rt * entry / risk
    mfe = 0.0
    if mode == "fixed3":
        tgt = entry + side * 3 * risk
        for j in range(i, min(i + 30, len(d))):
            mfe = max(mfe, side * ((H[j] if side == 1 else L[j]) - entry) / risk)
            if (L[j] <= stop) if side == 1 else (H[j] >= stop): return -1 - fee_r, j, mfe
            if j > i and ((H[j] >= tgt) if side == 1 else (L[j] <= tgt)): return 3 - fee_r, j, mfe
        j = min(i + 30, len(d)) - 1
        return side * (C[j] - entry) / risk - fee_r, j, mfe
    # trail
    part = entry + side * 2 * risk; took = False; st = stop; banked = 0.0
    for j in range(i, min(i + 90, len(d))):
        mfe = max(mfe, side * ((H[j] if side == 1 else L[j]) - entry) / risk)
        hit = (L[j] <= st) if side == 1 else (H[j] >= st)
        if hit:
            r = side * (st - entry) / risk
            return (banked + 0.5 * r if took else r) - fee_r, j, mfe
        if not took and j > i and ((H[j] >= part) if side == 1 else (L[j] <= part)):
            took, banked, st = True, 1.0, entry
        if took and j >= 10:
            tr = (L[j-9:j+1].min() if side == 1 else H[j-9:j+1].max())
            st = max(st, tr) if side == 1 else min(st, tr)
    j = min(i + 90, len(d)) - 1
    r = side * (C[j] - entry) / risk
    return (banked + 0.5 * r if took else r) - fee_r, j, mfe

def pivots(H, L, k=3):
    ph = np.full(len(H), False); pl = np.full(len(L), False)
    for i in range(k, len(H) - k):
        ph[i] = H[i] == H[i-k:i+k+1].max(); pl[i] = L[i] == L[i-k:i+k+1].min()
    return ph, pl

def signals(d: pd.DataFrame) -> list[dict]:
    """Candidate entries: (bar, side, entry, stop, fee, setup, tags). No lookahead."""
    d = ind(d); T = htf(d)
    O, H, L, C = d.open.values, d.high.values, d.low.values, d.close.values
    A, E20, E50, R = d.atr.values, d.ema20.values, d.ema50.values, d.rsi.values
    ph, pl = pivots(H, L)
    out = []
    for i in range(210, len(d) - 2):
        for side in (1, -1):
            # pullback to EMA20 in 4h trend, maker
            trend = (E20[i-1] > E50[i-1] and C[i-1] > E50[i-1]) if side == 1 else (E20[i-1] < E50[i-1] and C[i-1] < E50[i-1])
            lvl = E20[i-1]
            if trend and ((side == 1 and L[i] <= lvl < O[i]) or (side == -1 and H[i] >= lvl > O[i])):
                stop = (d.lo10.values[i] - 0.25*A[i-1]) if side == 1 else (d.hi10.values[i] + 0.25*A[i-1])
                if (stop < lvl) if side == 1 else (stop > lvl):
                    out.append(dict(i=i, side=side, entry=lvl, stop=stop, fee=2*FEE_M, setup="pullback", htf=T[i]*side, div=np.nan))
            # failed move at 20-bar extreme, maker at close next bar
            ll, hh = d.ll20.values[i], d.hh20.values[i]
            swept = (L[i] < ll and C[i] > ll) if side == 1 else (H[i] > hh and C[i] < hh)
            if swept and ((L[i+1] <= C[i]) if side == 1 else (H[i+1] >= C[i])):
                stop = (L[i] - 0.25*A[i]) if side == 1 else (H[i] + 0.25*A[i])
                k = i - 20 + (int(np.argmin(L[i-20:i])) if side == 1 else int(np.argmax(H[i-20:i])))
                div = (R[i] > R[k]) if side == 1 else (R[i] < R[k])
                out.append(dict(i=i+1, side=side, entry=C[i], stop=stop, fee=2*FEE_M, setup="failed_move", htf=T[i]*side, div=div))
            # double bottom/top: second pivot within 0.5 ATR of first, 5-40 bars apart, pivot confirmed 3 bars later;
            # then close through neckline within 15 bars; maker retest of neckline within 10 bars
        for side in (1, -1):
            j = i - 3  # pivot confirmed at i
            if j < 50: continue
            if side == 1 and pl[j]:
                prev = [p for p in range(j-40, j-4) if pl[p]]
                if prev and abs(L[prev[-1]] - L[j]) <= 0.5*A[j]:
                    p = prev[-1]; neck = H[p:j+1].max(); low2 = min(L[p], L[j])
                    brk = next((b for b in range(i, min(i+15, len(d)-1)) if C[b] > neck), None)
                    if brk is not None:
                        fill = next((f for f in range(brk+1, min(brk+11, len(d))) if L[f] <= neck), None)
                        if fill is not None and L[brk+1:fill+1].min() > low2:
                            out.append(dict(i=fill, side=1, entry=neck, stop=low2-0.25*A[j], fee=2*FEE_M, setup="double", htf=T[brk], div=R[j] > R[p]))
            if side == -1 and ph[j]:
                prev = [p for p in range(j-40, j-4) if ph[p]]
                if prev and abs(H[prev[-1]] - H[j]) <= 0.5*A[j]:
                    p = prev[-1]; neck = L[p:j+1].min(); hi2 = max(H[p], H[j])
                    brk = next((b for b in range(i, min(i+15, len(d)-1)) if C[b] < neck), None)
                    if brk is not None:
                        fill = next((f for f in range(brk+1, min(brk+11, len(d))) if H[f] >= neck), None)
                        if fill is not None and H[brk+1:fill+1].max() < hi2:
                            out.append(dict(i=fill, side=-1, entry=neck, stop=hi2+0.25*A[j], fee=2*FEE_M, setup="double", htf=-T[brk], div=R[j] < R[p]))
        # Donchian 20 breakout, taker, stop at 10-bar opposite extreme: trend-following reference
        for side in (1, -1):
            if (side == 1 and H[i] > d.hh20.values[i] and H[i-1] <= d.hh20.values[i-1]) or (side == -1 and L[i] < d.ll20.values[i] and L[i-1] >= d.ll20.values[i-1]):
                entry = max(d.hh20.values[i], O[i]) if side == 1 else min(d.ll20.values[i], O[i])
                stop = (d.lo10.values[i] - 0.25*A[i-1]) if side == 1 else (d.hi10.values[i] + 0.25*A[i-1])
                if (stop < entry) if side == 1 else (stop > entry):
                    out.append(dict(i=i, side=side, entry=entry, stop=stop, fee=2*FEE_T, setup="breakout", htf=T[i]*side, div=np.nan))
    return d, out

def run() -> pd.DataFrame:
    rows = []
    for name, d0 in load().items():
        d, sig = signals(d0)
        for mode in ("fixed3", "trail"):
            busy = {}
            for s in sorted(sig, key=lambda x: x["i"]):
                key = (s["setup"], s["side"])
                if busy.get(key, -1) >= s["i"]: continue  # one position per setup and side
                if abs(s["entry"] - s["stop"]) / s["entry"] < 0.002: continue  # too tight to fill honestly
                r, j, mfe = manage(d, s["i"], s["side"], s["entry"], s["stop"], s["fee"], mode)
                busy[key] = j
                t = d.index[s["i"]]
                era = "BTC 2017-19" if name == "BTC_17" else ("2022 design" if t.year == 2022 else "2023 holdout")
                rows.append(dict(pair=name, t=t, era=era, mode=mode, R=r, mfe=mfe, stop_pct=abs(s["entry"]-s["stop"])/s["entry"]*100, **{k: s[k] for k in ("setup", "side", "htf", "div")}))
    return pd.DataFrame(rows)

def summ(t):
    w = t.R > 0
    return pd.Series(dict(n=len(t), win=w.mean(), expR=t.R.mean(), pf=t.R[w].sum() / max(-t.R[~w].sum(), 1e-9)))

if __name__ == "__main__":
    t = run(); t.to_csv(Path(__file__).with_name("edge_trades.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.float_format", "{:.2f}".format)
    t["filt"] = np.where(t.htf > 0, "with daily", np.where(t.htf < 0, "against daily", "daily flat"))
    g = t.groupby(["setup", "mode", "filt", "era"]).apply(summ, include_groups=False).unstack("era")
    print(g[["n", "expR"]].round(2))
    print(t[t.setup.isin(["failed_move", "double"])].groupby(["setup", "mode", "div", "era"]).apply(summ, include_groups=False).unstack("era")[["n", "expR"]].round(2))
