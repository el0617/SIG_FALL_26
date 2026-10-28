"""Module 8 - Probability-weighted scenario tree for the next 3 months (v1).

Run (after Modules 2 and 6 have been run at least once):
    python -m src.scenarios            # tree, payoffs, expected active return per candidate, plots
    python -m src.scenarios --robust   # thresholds, sample, smoothing and payoff-model variants

WHAT IT DOES
  1. Splits every historical 3-month window (months t+1..t+3, 2003+) into six mutually exclusive
     scenarios from three observable moves: real 10y change, 10y breakeven change, SPY return.
  2. Gives each scenario three probabilities:
       A  base rate      - unconditional frequency
       B  policy analog  - windows that started when the market priced the same Fed direction as today
                           (2y minus fed funds; WIRP replaces it when data/raw/wirp.csv exists)
       C  recession risk - windows that started with the real-time recession probit (Module 2) in the
                           same tercile as today
     Conditional sets are shrunk toward the base rate (pseudo-count K) so thin cells can't dominate.
  3. Payoff of each candidate (Module 6 CANDIDATES) in each scenario = its active return vs SPY over 3 months:
       empirical (main): the average realized active return of today's weights in that scenario's windows
       model (check)   : ETF sensitivities x the scenario's median moves (Module 6 formula)
  4. Expected active return, worst scenario, and break-even sell-off probabilities.

PRE-REGISTERED (written 2026-10-28, before any output was seen; do not change after seeing results)
  - Scenario rules, priority order:
      1 Stagflationary sell-off  SPY <= -5% and breakeven change > 0
      2 Growth scare             SPY <= -5% (and breakeven change <= 0)
      3 Inflation scare          breakeven >= +20bp
      4 Real-yield squeeze       real 10y >= +25bp
      5 Easing rally             real 10y <= -25bp
      6 Steady state             everything else
  - Policy analog bucket: hikes priced if 2y - FF >= +25bp, cuts if <= -25bp, else flat.
    WIRP (when present): hikes if cumulative priced change to the last meeting in the horizon > +10bp,
    cuts if < -10bp, else flat. Recession tercile: terciles of the real-time probit, 1997+.
  - Shrinkage K = 12 pseudo-windows (4 non-overlapping quarters) toward the base rate.
  - Decision rule for a candidate X vs "Current (drifted)": X is ROBUSTLY BETTER if its expected active
    return is higher in at least 5 of the 6 (probability set x payoff method) combinations AND its worst
    scenario (empirical) is no more than 0.5pp worse. ROBUSTLY WORSE is the mirror image. Else MIXED.

CAVEATS (state them with the results)
  - In-sample: scenario payoffs and frequencies come from the same 2003+ history; no forecast skill is claimed.
  - Overlapping monthly windows: the n of independent quarters is about n/3; no t-stats are reported.
  - Today's weights are back-cast: XLC has no history before mid-2018 and is set to SPY (zero active) before.
  - Probabilities describe how often each outcome happened in similar conditions, not a forecast.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import RAW
from src.policy import build_state
from src.portfolio import (load_portfolio, etf_returns, drift, apply_trades, etf_sensitivities,
                           CANDIDATES, RECOMMENDED, HOLDINGS)

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
START = "2003-01-31"     # first month-end with DFII10 (real 10y)
H = 3
SCEN = ["Stagflationary sell-off", "Growth scare", "Inflation scare", "Real-yield squeeze",
        "Easing rally", "Steady state"]
SHORT = {"Stagflationary sell-off": "Stagflation\nsell-off", "Growth scare": "Growth\nscare",
         "Inflation scare": "Inflation\nscare", "Real-yield squeeze": "Real-yield\nsqueeze",
         "Easing rally": "Easing\nrally", "Steady state": "Steady\nstate"}
BASE = dict(dd=-5.0, be=20.0, real=25.0, path_band=25.0, wirp_band=10.0, K=12, start=START, model_tag="full")
SETS = ["A base rate", "B policy analog", "C recession risk"]
CURRENT = "Current (drifted)"


# ------------------------------------------------------------------ inputs
def windows(R, S, start):
    """One row per window starting at month-end t: 3-month changes over t -> t+3 and conditioning info at t."""
    idx = R.index
    spy = ((1 + R["SPY"] / 100).rolling(H).apply(np.prod, raw=True) - 1).shift(-H) * 100
    S = S.reindex(idx)
    W = pd.DataFrame({
        "d_real": (S["real10y"].shift(-H) - S["real10y"]) * 100,
        "d_be": (S["breakeven10y"].shift(-H) - S["breakeven10y"]) * 100,
        "spy": spy,
        "path": S["path_2y_minus_ff"] * 100,       # known at t
    }, index=idx)
    return W.loc[start:].dropna(subset=["d_real", "d_be", "spy"])


def classify(W, p):
    sell = W["spy"] <= p["dd"]
    lab = np.select(
        [sell & (W["d_be"] > 0), sell, W["d_be"] >= p["be"], W["d_real"] >= p["real"], W["d_real"] <= -p["real"]],
        SCEN[:5], default=SCEN[5])
    return pd.Series(lab, index=W.index, name="scenario")


def probit_series():
    f = OUT / "cycle_state.csv"
    if not f.exists():
        raise FileNotFoundError("outputs/cycle_state.csv not found: run python -m src.cycle first")
    return pd.read_csv(f, index_col=0, parse_dates=True)["p_recession_12m"]


def wirp_direction(band):
    """'hikes' / 'cuts' / 'flat' from data/raw/wirp.csv if present (columns: meeting, cum_bp), else None."""
    f = RAW / "wirp.csv"
    if not f.exists():
        return None, None
    w = pd.read_csv(f, parse_dates=["meeting"]).sort_values("meeting")
    cum = float(w["cum_bp"].iloc[-1])
    return ("hikes" if cum > band else "cuts" if cum < -band else "flat"), cum


def path_bucket(x, band):
    return np.where(x >= band, "hikes", np.where(x <= -band, "cuts", "flat"))


# ------------------------------------------------------------------ probabilities
def probabilities(lab, W, prob, asof, p):
    base = lab.value_counts(normalize=True).reindex(SCEN, fill_value=0.0)

    def shrunk(mask):
        sub = lab[mask.reindex(lab.index, fill_value=False)]
        cnt = sub.value_counts().reindex(SCEN, fill_value=0)
        return (cnt + p["K"] * base) / (len(sub) + p["K"]), len(sub)

    # B: policy analog
    wd, cum = wirp_direction(p["wirp_band"])
    today_path = W["path"].reindex([asof]).iloc[0] if asof in W.index else np.nan
    if np.isnan(today_path):
        S_last = build_state()["path_2y_minus_ff"].loc[:asof].dropna()
        today_path = float(S_last.iloc[-1]) * 100
    direction = wd or str(path_bucket(np.array([today_path]), p["path_band"])[0])
    bucket = pd.Series(path_bucket(W["path"].values, p["path_band"]), index=W.index)
    pB, nB = shrunk(bucket == direction)

    # C: recession-risk tercile (probit known at t)
    pr = prob.reindex(W.index)
    edges = prob.loc["1997":].dropna().quantile([1 / 3, 2 / 3]).values
    terc = pd.Series(np.digitize(pr.values, edges), index=W.index).where(pr.notna())
    today_p = float(prob.loc[:asof].dropna().iloc[-1])
    t_today = int(np.digitize([today_p], edges)[0])
    pC, nC = shrunk(terc == t_today)

    P = pd.DataFrame({SETS[0]: base, SETS[1]: pB, SETS[2]: pC})
    info = dict(direction=direction, source="WIRP" if wd else "2y - fed funds", path_bp=today_path, wirp_cum=cum,
                nB=nB, probit=today_p, tercile=["low", "middle", "high"][t_today], nC=nC, n=len(lab),
                edges=edges)
    return P, info


# ------------------------------------------------------------------ payoffs
def candidate_weights(R):
    p = load_portfolio()
    cur, _ = drift(p["weight"], R)
    return {name: apply_trades(cur, tr) for name, tr in CANDIDATES.items()}


def active_3m(R, w):
    """3-month active return vs SPY (%) of fixed weights w for windows starting at each t (XLC pre-2018 = SPY)."""
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    rel["XLC"] = rel["XLC"].fillna(0.0)
    port = R["SPY"] + (rel * (w / 100)).sum(axis=1).where(rel.notna().all(axis=1))
    g = lambda x: ((1 + x / 100).rolling(H).apply(np.prod, raw=True) - 1).shift(-H) * 100
    return g(port) - g(R["SPY"])


def thesis_3m(R):
    """DEF-GRW over the same windows (equal-weight XLP/XLU/XLV minus VGT/XLY), 3-month %, for context."""
    g = lambda x: ((1 + x / 100).rolling(H).apply(np.prod, raw=True) - 1).shift(-H) * 100
    return g(R[["XLP", "XLU", "XLV"]].mean(axis=1, skipna=False)) - g(R[["VGT", "XLY"]].mean(axis=1, skipna=False))


def payoffs(R, W, lab, Wts, sens, tag):
    med = W.groupby(lab)[["d_real", "d_be", "spy"]].median().reindex(SCEN)
    emp, mod, neg = {}, {}, {}
    for name, w in Wts.items():
        a = active_3m(R, w).reindex(W.index)
        emp[name] = a.groupby(lab).mean().reindex(SCEN)
        neg[name] = (a < 0).groupby(lab).mean().reindex(SCEN)
        x = w / 100
        mod[name] = pd.Series({s: float((x * (sens[f"real_{tag}"] * med.loc[s, "d_real"] / 100
                                              + sens[f"be_{tag}"] * med.loc[s, "d_be"] / 100
                                              + (sens["beta"] - 1) * med.loc[s, "spy"])).sum()) for s in SCEN})
    th = thesis_3m(R).reindex(W.index).groupby(lab).mean().reindex(SCEN)
    return pd.DataFrame(emp), pd.DataFrame(mod), pd.DataFrame(neg), med, th


# ------------------------------------------------------------------ decision
def summarize(P, E, M, NEG):
    rows = []
    for c in E.columns:
        for ps in P.columns:
            rows.append({"candidate": c, "prob_set": ps,
                         "exp_empirical": float(P[ps] @ E[c].fillna(0)), "exp_model": float(P[ps] @ M[c]),
                         "p_underperform": float(P[ps] @ NEG[c].fillna(0)),
                         "worst_scenario": E[c].idxmin(), "worst": float(E[c].min())})
    return pd.DataFrame(rows)


def verdict(T, c, ref=CURRENT, tol=0.5):
    a, b = T[T.candidate == c].set_index("prob_set"), T[T.candidate == ref].set_index("prob_set")
    wins = int((a["exp_empirical"] > b["exp_empirical"]).sum() + (a["exp_model"] > b["exp_model"]).sum())
    worst_gap = float(a["worst"].iloc[0] - b["worst"].iloc[0])
    if wins >= 5 and worst_gap >= -tol:
        v = "ROBUSTLY BETTER"
    elif wins <= 1 and worst_gap <= tol:
        v = "ROBUSTLY WORSE"
    else:
        v = "MIXED"
    return v, wins, worst_gap


def breakeven_selloff(P, E, c, ref):
    """Total sell-off probability (scenarios 1+2, others rescaled pro rata) at which c and ref tie (empirical)."""
    d = (E[c] - E[ref]).fillna(0)
    base = P[SETS[0]]
    sell, rest = SCEN[:2], SCEN[2:]
    ds = float((base[sell] / base[sell].sum()) @ d[sell])
    dr = float((base[rest] / base[rest].sum()) @ d[rest])
    if np.isclose(ds, dr):
        return np.nan
    q = -dr / (ds - dr)
    return float(q) if 0 <= q <= 1 else np.nan


# ------------------------------------------------------------------ run
def build(params=None):
    p = {**BASE, **(params or {})}
    R = etf_returns()
    S = build_state()
    W = windows(R, S, p["start"])
    lab = classify(W, p)
    asof = R.index[-1]
    P, info = probabilities(lab, W, probit_series(), asof, p)
    Wts = candidate_weights(R)
    sens = etf_sensitivities(R, S)
    E, M, NEG, med, th = payoffs(R, W, lab, Wts, sens, p["model_tag"])
    T = summarize(P, E, M, NEG)
    return dict(p=p, W=W, lab=lab, P=P, info=info, E=E, M=M, NEG=NEG, med=med, th=th, T=T, asof=asof, Wts=Wts)


def plot_tree(b, path):
    P, E = b["P"], b["E"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(SCEN)); bw = 0.27
    for k, ps in enumerate(P.columns):
        axes[0].bar(x + (k - 1) * bw, P[ps] * 100, bw, label=ps)
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT[s] for s in SCEN], fontsize=8)
    axes[0].set_ylabel("probability (%)"); axes[0].set_title("How often each 3-month outcome happened"); axes[0].legend(fontsize=8)
    cols = list(E.columns); bw2 = 0.8 / len(cols)
    for k, c in enumerate(cols):
        axes[1].bar(x + (k - (len(cols) - 1) / 2) * bw2, E[c], bw2, label=c)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([SHORT[s] for s in SCEN], fontsize=8)
    axes[1].set_ylabel("active return vs SPY, 3 months (%)"); axes[1].set_title("Average active return in each outcome (2003+)")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    print("=" * 90 + "\nMODULE 8 - PROBABILITY-WEIGHTED SCENARIO TREE (next 3 months)\n" + "=" * 90)
    b = build()
    P, E, M, T, info, med, lab = b["P"], b["E"], b["M"], b["T"], b["info"], b["med"], b["lab"]
    print(f"Windows: {lab.index.min():%Y-%m} to {lab.index.max():%Y-%m} start months, n={info['n']} "
          f"(about {info['n'] // 3} non-overlapping quarters). As of {b['asof']:%Y-%m-%d}.")
    wtxt = "" if info["wirp_cum"] is None else f", WIRP {info['wirp_cum']:+.0f}bp"
    print(f"Policy analog: market prices {info['direction'].upper()} (source {info['source']}; "
          f"2y-FF {info['path_bp']:+.0f}bp{wtxt}), {info['nB']} analog windows")
    print(f"Recession risk: probit {info['probit']:.0%} -> {info['tercile']} tercile "
          f"(cut-offs {info['edges'][0]:.0%} / {info['edges'][1]:.0%}), {info['nC']} analog windows")

    tree = pd.concat([med.round(1), lab.value_counts().reindex(SCEN).rename("windows"), (P * 100).round(1),
                      b["th"].round(2).rename("DEF-GRW 3m %")], axis=1)
    print("\nScenario tree (median moves in bp / %, probabilities in %):\n", tree.to_string())
    tree.to_csv(OUT / "scenarios_tree.csv")
    print("\nActive return vs SPY by scenario, 3 months (%), empirical:\n", E.T.round(2).to_string())
    print("\nSame, model (sensitivities x median moves):\n", M.T.round(2).to_string())
    E.T.round(3).to_csv(OUT / "scenarios_payoff_empirical.csv"); M.T.round(3).to_csv(OUT / "scenarios_payoff_model.csv")
    T.round(3).to_csv(OUT / "scenarios_summary.csv", index=False)
    piv = T.pivot(index="candidate", columns="prob_set", values="exp_empirical")
    print("\nExpected 3-month active return (%), empirical payoffs:\n", piv.round(2).to_string())
    print("\nExpected 3-month active return (%), model payoffs:\n",
          T.pivot(index="candidate", columns="prob_set", values="exp_model").round(2).to_string())
    print("\nP(underperform SPY over 3 months):\n",
          (T.pivot(index="candidate", columns="prob_set", values="p_underperform") * 100).round(0).to_string())

    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    for ps in SETS:
        top = P[ps].sort_values(ascending=False)
        print(f"  {ps}: most likely {top.index[0]} {top.iloc[0]:.0%}, then {top.index[1]} {top.iloc[1]:.0%}; "
              f"sell-off (1+2) {P[ps][SCEN[:2]].sum():.0%}")
    rows = []
    for c in E.columns:
        if c == CURRENT:
            continue
        v, wins, gap = verdict(T, c)
        q = breakeven_selloff(P, E, c, CURRENT)
        rows.append({"candidate": c, "verdict": v, "wins_of_6": wins, "worst_vs_current": round(gap, 2),
                     "breakeven_selloff_p": None if np.isnan(q) else round(q, 2)})
        qs = "never within 0-100%" if np.isnan(q) else f"{q:.0%}"
        print(f"  {c} vs current: {v} (higher expected return in {wins}/6; worst scenario {gap:+.2f}pp; "
              f"ties with current at sell-off probability {qs})")
    pd.DataFrame(rows).to_csv(OUT / "scenarios_verdicts.csv", index=False)
    print("  In-sample, overlapping windows: describes history in similar conditions, not a forecast.")
    print("=" * 90)
    plot_tree(b, OUT / "scenarios_tree.png")


def robustness():
    specs = [("baseline", {}), ("drawdown -4%", {"dd": -4.0}), ("drawdown -7%", {"dd": -7.0}),
             ("breakeven 15bp", {"be": 15.0}), ("breakeven 25bp", {"be": 25.0}),
             ("real 20bp", {"real": 20.0}), ("real 35bp", {"real": 35.0}),
             ("shrinkage K=6", {"K": 6}), ("shrinkage K=24", {"K": 24}),
             ("sample 2010+", {"start": "2010-01-31"}), ("model betas post-2020", {"model_tag": "post"}),
             ("policy band 50bp", {"path_band": 50.0})]
    rows = []
    for name, prm in specs:
        print(f"  running: {name}")
        b = build(prm)
        v, wins, gap = verdict(b["T"], RECOMMENDED)
        tv, twins, _ = verdict(b["T"], "Team direction: add defensives")
        e = b["T"][b["T"].candidate == RECOMMENDED].set_index("prob_set")["exp_empirical"]
        c = b["T"][b["T"].candidate == CURRENT].set_index("prob_set")["exp_empirical"]
        rows.append({"spec": name, "sell-off p (C)": round(b["P"][SETS[2]][SCEN[:2]].sum() * 100, 1),
                     "rec - cur (A)": round(e[SETS[0]] - c[SETS[0]], 2), "rec - cur (C)": round(e[SETS[2]] - c[SETS[2]], 2),
                     "recommended": f"{v} ({wins}/6)", "team direction": f"{tv} ({twins}/6)"})
    out = pd.DataFrame(rows)
    print("\n", out.to_string(index=False))
    out.to_csv(OUT / "scenarios_robustness.csv", index=False)


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()