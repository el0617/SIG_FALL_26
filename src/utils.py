import numpy as np
import pandas as pd

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