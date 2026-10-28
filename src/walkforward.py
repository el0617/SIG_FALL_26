"""Module 9 - Walk-forward out-of-sample test of the predictive tilts (v1).

Run (after the modules have been run once; Module 7 needs the Bloomberg valuation files):
    python -m src.walkforward            # all signals, expanding window, plots
    python -m src.walkforward --robust   # training length, rolling window, later start, 1-month horizon

QUESTION
  Modules 2, 3, 5 and 7 report in-sample tilts: the regression saw the whole history, including the
  months it is judged on. Would someone using each signal in real time have forecast the next-3-month
  Defensives-minus-Growth spread better than the plain historical average?

HOW (no information from after month t is used to forecast at t)
  For every month t from the first month with MIN_TRAIN usable rows:
    - training rows s are those whose 3-month outcome was complete by t (s <= t - 3 months);
    - fit the module's own regression on them (same regressors; OLS);
    - forecast y_t from today's signal; benchmark = historical mean of the training outcomes;
    - OOS tilt = forecast - benchmark (what the signal adds).
  The signals themselves are already built in real time (expanding or rolling z-scores, publication lags),
  so the whole chain is out of sample except for data revisions in macro series (Module 2).

PRE-REGISTERED (written 2026-10-28, before any output was seen; do not change after seeing results)
  Statistic: Clark-West (2007) adjusted MSPE test of signal model vs historical mean, HAC(6) errors for the
  overlapping 3-month outcomes. Out-of-sample R^2 (Campbell-Thompson) reported alongside.
    OOS VALUE   CW t >= 1.65 and OOS R^2 > 0
    WEAK        CW t >= 1.65 but OOS R^2 <= 0 (the signal points the right way but is too noisy to use as is)
    NO OOS VALUE otherwise
  Context (no verdict): hit rate of the tilt's sign, average spread when the tilt pointed to defensives vs
  growth, and how stable the slope's sign was through time.

CAVEATS
  - Overlapping monthly outcomes: about one independent observation per quarter.
  - Module 2 uses today's revised macro data (ALFRED vintages not used), which flatters it slightly.
  - Evaluation windows differ by signal (each starts when its own history allows); compare with care.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.cycle import returns_panels, build_cycle

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
H = 3
MIN_TRAIN = 96          # usable training months before the first forecast
HAC = 6
TARGET = "DEF-GRW"
BASE = dict(min_train=MIN_TRAIN, window=None, start=None, h=H)


# ------------------------------------------------------------------ signals (each exactly as its module tests it)
def signals():
    """{name: (DataFrame of regressors, module, what the in-sample test reported)}; missing inputs are skipped."""
    out = {}
    cyc = build_cycle()
    out["Cycle momentum (level + change)"] = (pd.concat([cyc["lvl"], cyc["mom"]], axis=1), "Module 2",
                                              "NEUTRAL, tilt -0.5 (t -0.58)")
    try:
        from src.policy import build_state, impulse
        out["Policy tightening impulse"] = (impulse(build_state())[0].to_frame(), "Module 3",
                                            "SUPPORTS on the tilt, +2.8 (t 1.86)")
    except Exception as e:
        print(f"  [walkforward] policy signal unavailable: {str(e)[:80]}")
    try:
        from src.market import build_signals, composite
        out["Market confirmation composite"] = (composite(build_signals()[0])[0].to_frame(), "Module 5",
                                                "CONTRADICTS on the expected spread; tilt -2.3 (t -1.56)")
    except Exception as e:
        print(f"  [walkforward] market signal unavailable: {str(e)[:80]}")
    try:
        from src.valuation import load, build_signals as val_signals
        V = val_signals(load())
        out["Valuation gap (vs median)"] = (V["gap_dev"].to_frame(), "Module 7", "NEUTRAL, tilt +0.0 (t +0.05)")
        out["Relative EPS revisions"] = (V["rel_rev"].to_frame(), "Module 7", "NEUTRAL, tilt +3.1 (t +1.12)")
    except Exception as e:
        print(f"  [walkforward] valuation signals unavailable (Bloomberg files needed): {str(e)[:70]}")
    return out


# ------------------------------------------------------------------ engine
def walk(X, y, min_train=MIN_TRAIN, window=None, start=None, h=H):
    """Real-time forecasts of y_t from X_t. Row t of y is the outcome over t+1..t+h (annualized pp)."""
    X = X.astype(float)
    d = pd.concat([y.rename("y"), X], axis=1, sort=True)
    cols = list(X.columns)
    rows = []
    for k, t in enumerate(d.index):
        if d.loc[t, cols].isna().any():
            continue
        if start is not None and t < pd.Timestamp(start):
            continue
        known = d.index <= t - pd.offsets.MonthEnd(h)            # outcomes complete by month-end t
        tr = d.loc[known].dropna()
        if window is not None:
            tr = tr.iloc[-window:]
        if len(tr) < min_train:
            continue
        Xtr = np.column_stack([np.ones(len(tr)), tr[cols].values])
        b = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)[0]
        f = float(np.r_[1.0, d.loc[t, cols].values] @ b)
        mbar = float(tr["y"].mean())
        rows.append({"date": t, "y": d.loc[t, "y"], "forecast": f, "hist_mean": mbar, "tilt": f - mbar,
                     "slope": float(b[1]), "n_train": len(tr)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def evaluate(F, hac=HAC):
    """Clark-West test, OOS R^2, hit rate and conditional averages on months whose outcome is known."""
    e = F.dropna(subset=["y"])
    if len(e) < 24:
        return None
    y, f, m = e["y"].values, e["forecast"].values, e["hist_mean"].values
    cw = (y - m) ** 2 - ((y - f) ** 2 - (m - f) ** 2)
    res = sm.OLS(cw, np.ones(len(cw))).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    cw_t = float(np.asarray(res.tvalues)[0])
    r2 = 1 - np.sum((y - f) ** 2) / np.sum((y - m) ** 2)
    tilt_up = e["tilt"] > 0
    hit = float((np.sign(e["tilt"]) == np.sign(e["y"] - e["hist_mean"])).mean())
    if cw_t >= 1.65 and r2 > 0:
        v = "OOS VALUE"
    elif cw_t >= 1.65:
        v = "WEAK"
    else:
        v = "NO OOS VALUE"
    sl = F["slope"] if F["slope"].ndim == 1 else F["slope"].iloc[:, 0]
    return dict(start=e.index[0], end=e.index[-1], n=len(e), quarters=len(e) // 3, oos_r2=r2 * 100, cw_t=cw_t,
                verdict=v, hit=hit * 100,
                when_def=float(e.loc[tilt_up, "y"].mean()) if tilt_up.any() else np.nan,
                when_grw=float(e.loc[~tilt_up, "y"].mean()) if (~tilt_up).any() else np.nan,
                share_def=float(tilt_up.mean()) * 100,
                slope_sign_same=float((np.sign(sl) == np.sign(sl.iloc[-1])).mean()) * 100,
                tilt_now=float(F["tilt"].iloc[-1]), asof=F.index[-1])


def run(params=None):
    p = {**BASE, **(params or {})}
    ex1, ex3 = returns_panels()
    y = (ex1[TARGET] * 1200) if p["h"] == 1 else (ex3[TARGET] * 1200)
    S = signals()
    res, paths = {}, {}
    for name, (X, mod, insample) in S.items():
        F = walk(X, y, p["min_train"], p["window"], p["start"], p["h"])
        if F.empty:
            continue
        ev = evaluate(F, HAC if p["h"] > 1 else 3)
        if ev is None:
            continue
        res[name] = dict(module=mod, in_sample=insample, **ev)
        paths[name] = F
    return pd.DataFrame(res).T, paths


# ------------------------------------------------------------------ plots
def plot_paths(paths, T, path):
    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
    for i, (name, F) in enumerate(paths.items()):
        a, b = axes[i]
        e = F.dropna(subset=["y"])
        err_m = ((e["y"] - e["hist_mean"]) ** 2).cumsum()
        err_f = ((e["y"] - e["forecast"]) ** 2).cumsum()
        a.plot(e.index, err_m - err_f, color="#37474f", lw=1.4)
        a.axhline(0, c="k", lw=0.8)
        a.set_title(f"{name}: cumulative squared-error advantage over the historical mean "
                    f"(rising = signal helping); {T.loc[name, 'verdict']}", fontsize=9)
        b.plot(F.index, F["slope"], color="#1565c0", lw=1.4)
        b.axhline(0, c="k", lw=0.8)
        b.set_title(f"Slope estimated in real time (first regressor), pp/yr per unit", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=" * 90 + "\nMODULE 9 - WALK-FORWARD OUT-OF-SAMPLE TEST OF THE TILTS\n" + "=" * 90)
    T, paths = run()
    if T.empty:
        print("  no signal could be evaluated"); return
    cols = ["module", "start", "end", "quarters", "oos_r2", "cw_t", "verdict", "hit", "when_def", "when_grw",
            "share_def", "slope_sign_same", "tilt_now", "in_sample"]
    show = T[cols].copy()
    for c in ["start", "end"]:
        show[c] = pd.to_datetime(show[c]).dt.strftime("%Y-%m")
    for c in ["oos_r2", "cw_t", "hit", "when_def", "when_grw", "share_def", "slope_sign_same", "tilt_now"]:
        show[c] = show[c].astype(float).round(2)
    print("\nOut-of-sample results (forecasts from month t use only outcomes complete by t):\n", show.to_string())
    show.to_csv(OUT / "walkforward_summary.csv")
    pd.concat({k: v for k, v in paths.items()}).round(3).to_csv(OUT / "walkforward_paths.csv")
    plot_paths(paths, T, OUT / "walkforward_paths.png")

    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    for name, r in T.iterrows():
        print(f"  {name} ({r['module']}): {r['verdict']}  OOS R2 {float(r['oos_r2']):+.1f}%, Clark-West t {float(r['cw_t']):+.2f}, "
              f"{int(r['quarters'])} quarters {pd.Timestamp(r['start']):%Y-%m} to {pd.Timestamp(r['end']):%Y-%m}; "
              f"tilt sign right {float(r['hit']):.0f}% of months; spread {float(r['when_def']):+.1f} when it pointed to "
              f"defensives vs {float(r['when_grw']):+.1f} when it pointed to growth; today's real-time tilt "
              f"{float(r['tilt_now']):+.1f} pp/yr (in-sample: {r['in_sample']})")
    n_ok = int((T["verdict"] == "OOS VALUE").sum())
    print(f"  {n_ok} of {len(T)} signals add value out of sample under the pre-registered rule.")
    print("=" * 90)


def robustness():
    specs = [("baseline", {}), ("training 60 months", {"min_train": 60}), ("training 120 months", {"min_train": 120}),
             ("rolling 120-month window", {"window": 120}), ("forecasts from 2013", {"start": "2013-01-31"}),
             ("1-month horizon", {"h": 1})]
    rows = []
    for name, prm in specs:
        print(f"  running: {name}")
        T, _ = run(prm)
        for sig, r in T.iterrows():
            rows.append({"spec": name, "signal": sig, "verdict": r["verdict"], "oos_r2": round(float(r["oos_r2"]), 1),
                         "cw_t": round(float(r["cw_t"]), 2), "quarters": int(r["quarters"])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "walkforward_robustness.csv", index=False)
    print("\n", out.pivot(index="signal", columns="spec", values="verdict").to_string())
    print("\nClark-West t:\n", out.pivot(index="signal", columns="spec", values="cw_t").to_string())
    print("\nOOS R2 (%):\n", out.pivot(index="signal", columns="spec", values="oos_r2").to_string())


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()
