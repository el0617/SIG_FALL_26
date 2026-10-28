"""Module 12 - Historical analogs: when did markets last look like today, and what happened next? (v1)

Run (after Modules 1, 2 and 3 have been run, so their state files in outputs/ are current):
    python -m src.analogs            # analog table, outcomes, permutation test, plot
    python -m src.analogs --robust   # number of analogs, variable sets, distance, sample

HOW
  State vector at each month-end t (2003+, when the real 10y begins), all known at t:
    rates      real 10y level and 6m change, 10y breakeven level and 6m change
    policy     2y minus fed funds (market-implied path), 6m change in fed funds, NFCI
    macro      Module 1 growth and inflation z-scores, Module 2 leading momentum, recession probit
  Each variable is standardized over 2003+; distance = root mean squared difference across variables
  (each GROUP weighted equally, so four rate variables don't outvote one recession probability).
  Analogs = the K closest months, at least SPACING months apart (one month per episode), ending at least
  EXCLUDE months before today so their outcomes are known and don't overlap today's window.

PRE-REGISTERED (written 2026-10-28, before any output was seen; do not change after seeing results)
  K = 8, SPACING = 6, EXCLUDE = 12, horizon 3 months (6 months is context).
  Analogs FAVOR DEFENSIVES if the median next-3-month DEF-GRW is > 0 and at least 6 of 8 are positive;
  FAVOR GROWTH if the median is < 0 and at least 6 of 8 are negative; MIXED otherwise.
  Permutation test: share of 5,000 random sets of 8 spaced months whose average DEF-GRW is at least as far
  from the full-sample average as the analogs' (two-sided). Reported, not part of the verdict.

CAVEATS
  - Eight analogs are anecdotes with structure, not a statistical test; they illustrate, the modules decide.
  - Standardizing over the full sample uses later data to set the scale (descriptive, not a forecast).
  - Macro z-scores use today's revised data.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio import etf_returns
from src.scenarios import candidate_weights, active_3m, thesis_3m, CURRENT
from src.portfolio import RECOMMENDED

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
START = "2003-01-31"
K, SPACING, EXCLUDE = 8, 6, 12
N_PERM = 5000
GROUPS = {
    "rates": ["real10y", "d_real6", "breakeven10y", "d_be6"],
    "policy": ["path", "d_ff6", "nfci"],
    "macro": ["growth_z", "inflation_z", "momentum_z", "p_recession"],
}
LABEL = {"real10y": "Real 10y (%)", "d_real6": "Real 10y 6m chg (bp)", "breakeven10y": "Breakeven (%)",
         "d_be6": "Breakeven 6m chg (bp)", "path": "2y - FF (bp)", "d_ff6": "Fed funds 6m chg (bp)", "nfci": "NFCI",
         "growth_z": "Growth z (M1)", "inflation_z": "Inflation z (M1)", "momentum_z": "Lead momentum (M2)",
         "p_recession": "Recession prob. (M2)"}
BASE = dict(k=K, spacing=SPACING, exclude=EXCLUDE, start=START, groups=None, metric="group")


# ------------------------------------------------------------------ state
def _read(name):
    f = OUT / name
    if not f.exists():
        raise FileNotFoundError(f"outputs/{name} not found: run the module that writes it first")
    return pd.read_csv(f, index_col=0, parse_dates=True)


def state():
    P, Rg, C = _read("policy_state.csv"), _read("regime_state.csv"), _read("cycle_state.csv")
    X = pd.DataFrame({
        "real10y": P["real10y"], "d_real6": P["real10y"].diff(6) * 100,
        "breakeven10y": P["breakeven10y"], "d_be6": P["breakeven10y"].diff(6) * 100,
        "path": P["path_2y_minus_ff"] * 100, "d_ff6": P["fed_funds"].diff(6) * 100, "nfci": P["nfci"],
        "growth_z": Rg["growth_z"], "inflation_z": Rg["inflation_z"],
        "momentum_z": C["momentum_z"], "p_recession": C["p_recession_12m"],
    })
    return X


def outcomes(R):
    W = candidate_weights(R)
    g = lambda x, h: ((1 + x / 100).rolling(h).apply(np.prod, raw=True) - 1).shift(-h) * 100
    out = {"DEF-GRW 3m": thesis_3m(R), "SPY 3m": g(R["SPY"], 3),
           "Current active 3m": active_3m(R, W[CURRENT]), "XLP->XLF active 3m": active_3m(R, W[RECOMMENDED]),
           "Add-defensives active 3m": active_3m(R, W["Team direction: add defensives"]),
           "XLE-SPY 3m": g(R["XLE"], 3) - g(R["SPY"], 3)}
    d6 = (g(R[["XLP", "XLU", "XLV"]].mean(axis=1, skipna=False), 6) - g(R[["VGT", "XLY"]].mean(axis=1, skipna=False), 6))
    out["DEF-GRW 6m"] = d6
    return pd.DataFrame(out)


# ------------------------------------------------------------------ distance and selection
def distances(X, today, groups=None, metric="group"):
    groups = groups or GROUPS
    cols = [c for g in groups.values() for c in g]
    Z = (X[cols] - X[cols].mean()) / X[cols].std()
    diff2 = (Z - Z.loc[today]) ** 2
    if metric == "group":     # each group counts equally
        parts = [diff2[c].mean(axis=1) for c in groups.values()]
        d = np.sqrt(sum(parts) / len(parts))
    else:                     # each variable counts equally
        d = np.sqrt(diff2.mean(axis=1))
    return d.where(Z.notna().all(axis=1)), Z


def pick(d, today, k=K, spacing=SPACING, exclude=EXCLUDE):
    cand = d.drop(today, errors="ignore")
    cand = cand[cand.index <= today - pd.offsets.MonthEnd(exclude)].dropna().sort_values()
    chosen = []
    for t, v in cand.items():
        if all(abs((t.year - c.year) * 12 + t.month - c.month) >= spacing for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return pd.Series(d.loc[chosen], name="distance")


def permutation(Y, valid, k=K, spacing=SPACING, n=N_PERM, seed=7):
    """Averages of random sets of k months, spaced like the analogs."""
    rng = np.random.default_rng(seed)
    idx = list(valid)
    out = np.empty(n)
    for i in range(n):
        chosen = []
        for t in rng.permutation(idx):
            if all(abs((t.year - c.year) * 12 + t.month - c.month) >= spacing for c in chosen):
                chosen.append(t)
            if len(chosen) == k:
                break
        out[i] = Y.loc[chosen].mean()
    return out


def verdict(v):
    med, pos, neg = v.median(), int((v > 0).sum()), int((v < 0).sum())
    need = int(np.ceil(0.75 * len(v)))
    if med > 0 and pos >= need:
        return "FAVOR DEFENSIVES"
    if med < 0 and neg >= need:
        return "FAVOR GROWTH"
    return "MIXED"


def build(params=None):
    p = {**BASE, **(params or {})}
    R = etf_returns()
    X = state().loc[p["start"]:]
    today = R.index[-1]                                   # last complete month
    if today not in X.index or X.loc[today].isna().any():
        today = X.dropna().index[X.dropna().index <= today][-1]
    d, Z = distances(X, today, p["groups"], p["metric"])
    O = outcomes(R).reindex(X.index)
    A = pick(d, today, p["k"], p["spacing"], p["exclude"])
    A = A[O.loc[A.index, "DEF-GRW 3m"].notna()]
    return dict(p=p, X=X, Z=Z, d=d, today=today, A=A, O=O, R=R)


def describe(b):
    X, A, O, today = b["X"], b["A"], b["O"], b["today"]
    T = pd.concat([A.round(2), X.loc[A.index, ["real10y", "breakeven10y", "path", "p_recession", "growth_z", "inflation_z"]].round(2),
                   O.loc[A.index].round(2)], axis=1)
    ref = pd.concat([pd.Series({"distance": 0.0}), X.loc[today, ["real10y", "breakeven10y", "path", "p_recession", "growth_z", "inflation_z"]].round(2)])
    return T, ref


# ------------------------------------------------------------------ plots
def plot(b, path):
    X, Z, A, O, today = b["X"], b["Z"], b["A"], b["O"], b["today"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    cols = [c for g in GROUPS.values() for c in g]
    y = np.arange(len(cols))[::-1]
    for t in A.index:
        ax.scatter(Z.loc[t, cols], y, color="0.6", s=18, zorder=2)
    ax.scatter(Z.loc[today, cols], y, color="#c62828", s=60, zorder=3, label=f"today ({today:%b %Y})")
    ax.scatter(Z.loc[A.index, cols].mean(), y, marker="|", color="k", s=300, zorder=4, label="analog average")
    ax.axvline(0, c="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in cols], fontsize=9)
    ax.set_xlabel("standardized value (2003+)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("How close the analogs are to today", fontsize=10)
    ax = axes[1]
    v = O.loc[A.index, "DEF-GRW 3m"]
    ax.bar(range(len(v)), v.values, color=["#2e7d32" if x > 0 else "#c62828" for x in v.values])
    ax.axhline(O["DEF-GRW 3m"].loc[b["p"]["start"]:].mean(), color="k", ls="--", lw=0.8, label="all months, average")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(range(len(v))); ax.set_xticklabels([f"{t:%b %Y}" for t in v.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Defensives minus growth, next 3 months (%)"); ax.legend(fontsize=8)
    ax.set_title("What happened next", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
    print("=" * 90 + "\nMODULE 12 - HISTORICAL ANALOGS\n" + "=" * 90)
    b = build()
    A, O, today = b["A"], b["O"], b["today"]
    T, ref = describe(b)
    print(f"Today = {today:%b %Y} (last complete month). Today's state: " +
          ", ".join(f"{LABEL[c]} {b['X'].loc[today, c]:+.2f}" for c in b["X"].columns))
    print(f"\nThe {len(A)} closest months (each from a different episode, outcomes known):\n", T.to_string())
    T.to_csv(OUT / "analogs_table.csv")

    base = O.loc[b["p"]["start"]:].dropna(subset=["DEF-GRW 3m"])
    cols = ["DEF-GRW 3m", "DEF-GRW 6m", "SPY 3m", "Current active 3m", "XLP->XLF active 3m", "Add-defensives active 3m", "XLE-SPY 3m"]
    S = pd.DataFrame({"analog average": O.loc[A.index, cols].mean(), "analog median": O.loc[A.index, cols].median(),
                      "analogs positive": (O.loc[A.index, cols] > 0).sum(), "all months average": base[cols].mean()})
    print("\nOutcomes (%, cumulative over the horizon):\n", S.round(2).to_string())
    S.round(3).to_csv(OUT / "analogs_summary.csv")

    valid = base.index[base.index <= today - pd.offsets.MonthEnd(EXCLUDE)]
    sims = permutation(base["DEF-GRW 3m"], valid)
    mu = base.loc[valid, "DEF-GRW 3m"].mean()
    obs = O.loc[A.index, "DEF-GRW 3m"].mean()
    pval = float((np.abs(sims - mu) >= abs(obs - mu)).mean())
    v = verdict(O.loc[A.index, "DEF-GRW 3m"])
    plot(b, OUT / "analogs.png")

    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Closest analogs: " + ", ".join(f"{t:%b %Y}" for t in A.index))
    print(f"  Next 3 months, defensives minus growth: average {obs:+.1f}%, median {O.loc[A.index, 'DEF-GRW 3m'].median():+.1f}%, "
          f"{int((O.loc[A.index, 'DEF-GRW 3m'] > 0).sum())} of {len(A)} positive (all months: {mu:+.1f}% average)")
    print(f"  Permutation p-value (two-sided, {N_PERM} random sets): {pval:.2f}")
    print(f"  Candidates, next 3 months vs SPY (analog average): current {S.loc['Current active 3m', 'analog average']:+.2f}%, "
          f"XLP->XLF {S.loc['XLP->XLF active 3m', 'analog average']:+.2f}%, add defensives {S.loc['Add-defensives active 3m', 'analog average']:+.2f}%")
    print(f"  Verdict (pre-registered): {v}")
    print("  Analogs illustrate; they are not a forecast. 8 episodes, in-sample scaling.")
    print("=" * 90)


def robustness():
    specs = [("baseline", {}), ("K = 5", {"k": 5}), ("K = 12", {"k": 12}), ("each variable equal", {"metric": "var"}),
             ("rates + policy only", {"groups": {k: GROUPS[k] for k in ("rates", "policy")}}),
             ("macro only", {"groups": {"macro": GROUPS["macro"]}}), ("2010+ sample", {"start": "2010-01-31"}),
             ("spacing 12 months", {"spacing": 12})]
    rows = []
    for name, prm in specs:
        print(f"  running: {name}")
        b = build(prm)
        v = b["O"].loc[b["A"].index, "DEF-GRW 3m"]
        rows.append({"spec": name, "analogs": ", ".join(f"{t:%Y-%m}" for t in b["A"].index), "avg DEF-GRW 3m": round(v.mean(), 2),
                     "positive": f"{int((v > 0).sum())}/{len(v)}", "verdict": verdict(v),
                     "XLP->XLF minus current": round((b["O"].loc[b["A"].index, "XLP->XLF active 3m"] - b["O"].loc[b["A"].index, "Current active 3m"]).mean(), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "analogs_robustness.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()
