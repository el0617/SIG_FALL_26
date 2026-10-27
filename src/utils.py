"""Shared transforms and statistics.

Module 2 (cycle.py) still uses ann_change, zexp, composite and conditional_returns, which are unchanged.
Everything below the v2 marker is used by the regime model.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ---------------------------------------------------------------- v1 (kept for Module 2)
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
    mean = d.groupby("state").mean() * 1200
    hit = d.groupby("state").agg(lambda x: (x.dropna() > 0).mean() * 100)
    n = d.groupby("state").size().rename("months")
    return mean.round(1), hit.round(0), n


# ---------------------------------------------------------------- v2
ORDER = ["Goldilocks", "Reflation", "Stagflation", "Slowdown", "Transition"]

# Slide-facing names. Growth is measured vs its own 10-year trend; inflation vs a 2%-PCE-consistent anchor.
DISPLAY = {
    "Goldilocks":  "Strong growth / low inflation",
    "Reflation":   "Strong growth / high inflation",
    "Stagflation": "Weak growth / high inflation",
    "Slowdown":    "Weak growth / low inflation",
    "Transition":  "Transition (no clear regime)",
}
DIR_ORDER = ["G up / I down", "G up / I up", "G down / I up", "G down / I down"]


def chg(s, n):
    """n-period percent change without pandas' implicit forward-fill."""
    return (s / s.shift(n) - 1) * 100


def lagged(s, n):
    """Move observations n month-ends forward (publication lag) WITHOUT dropping the newest one.
    (v1 used s.shift(n), which pushed the latest release off the end of the series.)
    Requires a month-end index: on month-start dates freq="ME" would only roll to the same month's end (no lag)."""
    if not s.index.is_month_end.all():
        raise ValueError("lagged() needs a month-end DatetimeIndex; resample with data.monthly() first")
    return s.shift(n, freq="ME") if n else s


def _mad(x):
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else np.median(np.abs(x - np.median(x)))


def robust_scale(df, window, min_periods=60):
    """1.4826 * rolling MAD; where the MAD is 0 (more than half the values identical), 1.0 * rolling std."""
    w = len(df) if window is None else window
    sc = 1.4826 * df.rolling(w, min_periods=min_periods).apply(_mad, raw=True)
    return sc.where(sc != 0, df.rolling(w, min_periods=min_periods).std())


def zscore(df, window=120, min_periods=60, center=None, floor_window=360, floor_frac=0.75, soft=True):
    """Robust z-score using only data up to each date.

    centre : rolling median over `window` (None -> expanding), or a fixed anchor per column via
             center={"col": value} (used to measure inflation vs the Fed-consistent rate).
    scale  : 1.4826 * rolling MAD (rolling std where the MAD is 0), floored at floor_frac * a 30-year MAD, so a quiet decade
             (2010s) can't shrink the scale and turn ordinary moves into 3-sigma readings.
    squash : 3*tanh(z/3) instead of a hard clip at +-3. Ordering is preserved, so the 2022 peak
             and the 2023 disinflation no longer both read exactly 3.0.
    """
    w = len(df) if window is None else window
    med = df.rolling(w, min_periods=min_periods).median()
    for col, anchor in (center or {}).items():
        if col in med:
            med[col] = med[col].where(med[col].isna(), anchor)
    sc = robust_scale(df, w, min_periods)
    if floor_frac:
        fw = len(df) if window is None else max(floor_window, w)
        fl = robust_scale(df, fw, min_periods) * floor_frac
        sc = sc.where(~(sc < fl), fl)
    z = (df - med) / sc.where(sc > 0)
    return 3 * np.tanh(z / 3) if soft else z.clip(-3, 3)


def zrobust(df, window=120, min_periods=60):
    """v1 behaviour (rolling median/MAD, hard clip). Kept only for comparison."""
    return zscore(df, window, min_periods, floor_frac=None, soft=False)


def quadrant(g, i):
    if g > 0:
        return "Goldilocks" if i <= 0 else "Reflation"
    return "Stagflation" if i > 0 else "Slowdown"


def classify(g, i, band=0.15, sticky=True):
    """Regime label per month.
    - Both axes outside the band -> that quadrant.
    - Inside the band: if sticky and the point is still on the same side of both axes as the last
      confirmed regime, keep that regime (hysteresis, adds no lag); otherwise 'Transition'.
    - Any axis missing -> NaN (v1 wrongly returned 'Transition').
    """
    d = pd.concat([g, i], axis=1, keys=["g", "i"], sort=True)
    out, last = [], None
    for gv, iv in zip(d["g"].values, d["i"].values):
        if np.isnan(gv) or np.isnan(iv):
            out.append(np.nan); last = None; continue
        q = quadrant(gv, iv)
        if abs(gv) >= band and abs(iv) >= band:
            out.append(q); last = q
        elif sticky and last == q:
            out.append(q)
        else:
            out.append("Transition")
    return pd.Series(out, index=d.index, dtype=object)


def classify_direction(g, i, n=3):
    """Direction of travel: sign of the n-month change in each composite."""
    d = pd.concat([g, i], axis=1, keys=["g", "i"], sort=True)
    dg, di = d["g"].diff(n), d["i"].diff(n)
    lab = np.where(dg > 0, np.where(di <= 0, DIR_ORDER[0], DIR_ORDER[1]),
                   np.where(di > 0, DIR_ORDER[2], DIR_ORDER[3]))
    return pd.Series(lab, index=d.index, dtype=object).where(dg.notna() & di.notna())


def run_ids(labels):
    """Integer id for each consecutive run (episode) of the same label, indexed by the non-missing months.
    A NaN label is a break: runs on either side of a gap get different ids."""
    prev = labels.shift()
    new = (labels != prev) | prev.isna()
    return new.cumsum()[labels.notna()].astype(int)


def episode_table(labels, min_len=3, order=ORDER):
    l = labels.dropna()
    rid = run_ids(labels)
    runs = pd.DataFrame({"state": l.groupby(rid).first(), "len": l.groupby(rid).size()})
    t = runs.groupby("state")["len"].agg(episodes="count", avg_len="mean", longest="max",
                                         ep_3m_plus=lambda x: int((x >= min_len).sum()))
    t.insert(0, "months", l.value_counts())
    return t.reindex([s for s in order if s in t.index]).round(1)


def forward_excess(rets, bench="SPY", spreads=None):
    """Row t = return of month t+1 (so it pairs with the label known at month-end t).
    Sectors are in excess of the benchmark; spreads are equal-weight long minus equal-weight short."""
    fwd = rets.shift(-1)
    ex = fwd.drop(columns=bench).sub(fwd[bench], axis=0)
    for name, (long, short) in (spreads or {}).items():
        ex[name] = fwd[long].mean(axis=1, skipna=False) - fwd[short].mean(axis=1, skipna=False)
    return ex


def bh_qvalues(p):
    """Benjamini-Hochberg false-discovery-rate q-values."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    o = np.argsort(pv)
    ranked = pv[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(n)
    qv[o] = np.minimum(q, 1)
    out[ok] = qv
    return out


def conditional_stats(labels, ex, order=ORDER, loeo=True, family=None, q_max=0.10, min_eps=5):
    """For each series x state: next-month return stats conditional on the label at month-end t.

    raw / base / edge : annualized pp; edge = state mean - all-months mean.
    t, p              : SEs clustered by regime episode (persistent regimes make months non-independent);
                        p from t(G-1), G = episodes of the state. NaN when G < 2.
    t_crit            : one-sided 95% critical value of t(G-1), for raw_t.
    t_hac             : HAC (Newey-West, 3 lags) t-stat, for comparison with v1.
    raw_lo / raw_hi   : 95% interval for the in-state mean (episode-clustered, t(G-1)).
    loeo_min/max      : edge after dropping each single episode of that state, one at a time.
    sign_stable       : edge keeps its sign whichever single episode is dropped.
    q                 : Benjamini-Hochberg q-value within each family of tests.
    credible          : q <= q_max AND sign_stable AND episodes >= min_eps.
    """
    lab = labels.reindex(ex.index)
    rid = run_ids(lab).reindex(ex.index)
    rows = []
    for col in ex.columns:
        ok = ex[col].notna() & lab.notna()
        y, st, gid = ex[col][ok], lab[ok], rid[ok].astype(int)
        if len(y) < 36:
            continue
        base = y.mean()
        for s in [o for o in order if o in set(st)]:
            D = (st == s).astype(float)
            n = int(D.sum())
            if n < 6 or n == len(D):
                continue
            X = sm.add_constant(D.values)
            hac = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            m, mg = y[D == 1], gid[D == 1]
            eps = mg.unique()
            G = len(eps)
            if G >= 2:   # one cluster has no between-episode variation: clustered inference undefined
                cl = sm.OLS(y.values, X).fit(cov_type="cluster", cov_kwds={"groups": gid.values})
                raw_fit = sm.OLS(m.values, np.ones(len(m))).fit(cov_type="cluster", cov_kwds={"groups": mg.values})
                raw_se = float(np.atleast_1d(raw_fit.bse)[0])
                t_cl = float(cl.tvalues[1])
                p_cl = float(2 * stats.t.sf(abs(t_cl), G - 1))
                t_crit, t_975 = float(stats.t.ppf(0.95, G - 1)), float(stats.t.ppf(0.975, G - 1))
            else:
                raw_se = t_cl = p_cl = t_crit = t_975 = np.nan
            edge = (m.mean() - base) * 1200
            row = dict(series=col, state=s, months=n, episodes=G,
                       raw=m.mean() * 1200, raw_lo=(m.mean() - t_975 * raw_se) * 1200,
                       raw_hi=(m.mean() + t_975 * raw_se) * 1200, raw_t=m.mean() / raw_se if raw_se > 0 else np.nan,
                       t_crit=t_crit, base=base * 1200, edge=edge,
                       t=t_cl, p=p_cl, t_hac=float(hac.tvalues[1]),
                       hit=(m > 0).mean() * 100, hit_all=(y > 0).mean() * 100,
                       top_episode_share=float(m.groupby(mg).sum().abs().max() / m.groupby(mg).sum().abs().sum()))
            if loeo and len(eps) > 1:
                edges = []
                for e in eps:
                    keep = gid != e
                    yk, Dk = y[keep], D[keep]
                    edges.append((yk[Dk == 1].mean() - yk.mean()) * 1200)
                row.update(loeo_min=min(edges), loeo_max=max(edges),
                           sign_stable=bool(all(np.sign(x) == np.sign(edge) for x in edges)))
            else:
                row.update(loeo_min=np.nan, loeo_max=np.nan, sign_stable=False)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["family"] = out["series"].map(family or {}).fillna("sector")
    out["q"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "q"] = bh_qvalues(out.loc[idx, "p"].values)
    out["credible"] = (out["q"] <= q_max) & out["sign_stable"] & (out["episodes"] >= min_eps)
    return out


def transition_matrix(labels, order=ORDER):
    """Empirical one-month transition probabilities between labels.
    Only pairs of consecutive non-missing months count. A state with no outgoing transition stays in itself."""
    l = labels.dropna()
    a, b = labels.iloc[:-1], labels.iloc[1:]
    ok = a.notna().values & b.notna().values
    m = pd.crosstab(pd.Series(a.values[ok], name="from"), pd.Series(b.values[ok], name="to"))
    states = [s for s in order if s in set(l)]
    m = m.reindex(index=states, columns=states, fill_value=0)
    for s in m.index[m.sum(axis=1) == 0]:
        print(f"  [transition_matrix] '{s}' has no observed outgoing transition -> treated as absorbing (P=1 to itself)")
        m.loc[s, s] = 1
    return m.div(m.sum(axis=1), axis=0)


def forward_distribution(P, current, horizon=3):
    """Distribution over states h = 0..horizon-1 months ahead (row 0 = current state)."""
    v = pd.Series(0.0, index=P.index)
    v[current] = 1.0
    rows = []
    for _ in range(horizon):
        rows.append(v.copy())
        v = v @ P
    return pd.DataFrame(rows, index=[f"t+{h}" for h in range(horizon)])