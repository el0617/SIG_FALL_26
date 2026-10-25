import numpy as np
import pandas as pd
import statsmodels.api as sm

def ann_change(s, n=3):
    """n-month change, annualized, in %."""
    return ((s / s.shift(n)) ** (12 / n) - 1) * 100

def zexp(df, min_periods=60):
    """Expanding-window z-score: uses only data available at each date (no look-ahead)."""
    z = (df - df.expanding(min_periods).mean()) / df.expanding(min_periods).std()
    return z.clip(-3, 3)

def composite(df, min_components=3):
    return df.mean(axis=1).where(df.notna().sum(axis=1) >= min_components)

def conditional_returns(labels, rets, bench="SPY"):
    """Next-month excess return vs benchmark, grouped by the state known at month-end t."""
    fwd = rets.shift(-1)
    ex = fwd.drop(columns=bench).sub(fwd[bench], axis=0)
    d = ex.assign(state=labels).dropna(subset=["state"])
    mean = d.groupby("state").mean() * 1200                      # annualized, % points
    hit = d.groupby("state").agg(lambda x: (x.dropna() > 0).mean() * 100)
    n = d.groupby("state").size().rename("months")
    return mean.round(1), hit.round(0), n

ORDER = ["Goldilocks", "Reflation", "Stagflation", "Slowdown", "Transition"]

def zrobust(df, window=120, min_periods=60):
    """Robust z-score: (x - rolling median) / (1.4826 * rolling MAD), clipped at +-3.
    window=120 -> 10-year rolling baseline; window=None -> expanding.
    Uses only data up to each date (no look-ahead)."""
    w = len(df) if window is None else window
    med = df.rolling(w, min_periods=min_periods).median()
    mad = df.rolling(w, min_periods=min_periods).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
    return ((df - med) / (1.4826 * mad.replace(0, np.nan))).clip(-3, 3)

def episode_table(labels, min_len=3):
    """Distinct episodes (consecutive runs) per state, not months."""
    l = labels.dropna()
    run = (l != l.shift()).cumsum()
    g = l.groupby(run)
    runs = pd.DataFrame({"state": g.first(), "len": g.size()})
    return runs.groupby("state")["len"].agg(
        episodes="count", avg_len="mean", longest="max",
        ep_3m_plus=lambda x: int((x >= min_len).sum())).round(1)

def conditional_stats(labels, rets, bench="SPY", maxlags=3):
    """Next-month excess return vs bench, by state known at month-end t.
    edge = state mean - all-months mean (annualized, pp).
    t    = HAC t-stat from regressing excess return on a state dummy
           (same t-stat applies to 'edge', since edge = dummy coef x (1 - share))."""
    fwd = rets.shift(-1)
    ex = fwd.drop(columns=bench).sub(fwd[bench], axis=0)
    d = ex.assign(state=labels).dropna(subset=["state"])
    d = d.dropna(subset=list(ex.columns), how="all")
    rows = []
    for sec in ex.columns:
        y = d[sec].dropna()
        st = d.loc[y.index, "state"]
        base = y.mean()
        for s in sorted(st.unique()):
            D = (st == s).astype(float)
            if D.sum() < 6 or D.sum() == len(D):
                continue
            res = sm.OLS(y.values, sm.add_constant(D.values)).fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags})
            m = y[D == 1]
            rows.append(dict(sector=sec, state=s, months=int(D.sum()),
                             raw=m.mean() * 1200, base=base * 1200,
                             edge=(m.mean() - base) * 1200, t=res.tvalues[1],
                             hit=(m > 0).mean() * 100, hit_all=(y > 0).mean() * 100))
    return pd.DataFrame(rows)