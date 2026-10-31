"""Thesis tournament: five theses scored on one common, pre-registered set of rules.

Run:
    python -m src.tournament            # baseline scores, context, robustness (21 settings), chart

Rules: docs/tournament_preregistration.md (committed before this file). Each thesis is a long basket minus a
short basket (equal weights, monthly rebalanced). Six criteria score +1 / 0 / -1:
  C1 pays on average (HAC t), C2 pays beyond market beta (CAPM alpha HAC t), C3 same sign in both halves,
  C4 expected 3m return > 0 under Module 8's data-based probability sets A/B/C, C5 > 0 under all four judgment
  theses, C6 pays in sell-offs (t with n_eff = windows / 3).
Valuation (Module 7, Bloomberg) is context only and is written to an ignored file (licensed data).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.scenarios import build, SCEN, SETS, H

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

DEF, INF, GRW = ["XLP", "XLU", "XLV"], ["XLE", "XLB"], ["VGT", "XLY"]
THESES = {   # code: (name, {ticker: long weight}, {ticker: short weight})
    "A": ("Defensive tilt (team thesis)", {t: 1 / 3 for t in DEF}, {t: 1 / 2 for t in GRW}),
    "B": ("Inflation-hedge tilt", {t: 1 / 2 for t in INF}, {"SPY": 1.0}),
    "C": ("Barbell (TOPDOWN2 structure)", {**{t: 1 / 6 for t in DEF}, **{t: 1 / 4 for t in INF}}, {t: 1 / 2 for t in GRW}),
    "D": ("Hard-landing insurance", {t: 1 / 3 for t in DEF}, {"SPY": 1.0}),
    "E": ("Growth drift (anti-thesis)", {t: 1 / 2 for t in GRW}, {"SPY": 1.0}),
}
JUDGMENT = pd.DataFrame({   # % weights per scenario, order = SCEN
    "Inflation persists": [8, 5, 30, 17, 5, 35],
    "Higher for longer": [6, 10, 12, 30, 5, 37],
    "Hard landing": [8, 25, 7, 8, 20, 32],
    "Soft landing": [2, 5, 10, 10, 18, 55]}, index=SCEN) / 100
CRIT = ["C1", "C2", "C3", "C4", "C5", "C6"]
CRIT_NAME = {"C1": "Pays on average", "C2": "Pays beyond beta", "C3": "Stable in\nboth halves",
             "C4": "Pays in today's\nconditions", "C5": "Robust across\nmacro theses", "C6": "Insurance in\nsell-offs"}
SPLIT = "2013-01-01"
TE_BUDGET = 2.5
SAMPLE_STARTS = ["1999-01-31", "2003-01-31", "2010-01-31"]
SCEN_SPECS = [("baseline", {}), ("drawdown -4%", {"dd": -4.0}), ("drawdown -7%", {"dd": -7.0}),
              ("breakeven 15bp", {"be": 15.0}), ("breakeven 25bp", {"be": 25.0}),
              ("real 20bp", {"real": 20.0}), ("real 35bp", {"real": 35.0})]


# ------------------------------------------------------------------ returns
def basket(R, w):
    w = pd.Series(w)
    return (R[w.index] * w).sum(axis=1, min_count=len(w))


def monthly_tilt(R, code):
    _, L, S = THESES[code]
    return basket(R, L) - basket(R, S)


def tilt_3m(R, code):
    """3-month compounded long minus compounded short, for windows starting at month-end t (months t+1..t+3)."""
    _, L, S = THESES[code]
    g = lambda x: ((1 + x / 100).rolling(H).apply(np.prod, raw=True) - 1).shift(-H) * 100
    return g(basket(R, L)) - g(basket(R, S))


def hac_t(y, X=None, lags=3):
    X = np.ones((len(y), 1)) if X is None else sm.add_constant(X)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(np.asarray(res.params)[0]), float(np.asarray(res.tvalues)[0]), res


def sign_score(t, crit=1.65):
    return 1 if t >= crit else (-1 if t <= -crit else 0)


def all_sign(x):
    x = np.asarray(x, float)
    return 1 if (x > 0).all() else (-1 if (x < 0).all() else 0)


# ------------------------------------------------------------------ scoring
def score(R, b, start):
    """Score every thesis. R = monthly % returns, b = scenarios.build() output, start = return-sample start."""
    W, lab, P = b["W"], b["lab"], b["P"]
    rows, scen_tab = [], {}
    spy = R["SPY"].loc[start:]
    for code, (name, _, _) in THESES.items():
        m = monthly_tilt(R, code).loc[start:].dropna()
        mu, t1, _ = hac_t(m.values)
        d = pd.concat([m, spy], axis=1, keys=["y", "m"]).dropna()
        a, t2, res = hac_t(d["y"].values, d["m"].values)
        beta = float(np.asarray(res.params)[1])
        h1, h2 = m.loc[:"2012-12-31"].mean(), m.loc[SPLIT:].mean()
        x3 = tilt_3m(R, code).reindex(W.index)
        sm_ = x3.groupby(lab).mean().reindex(SCEN)
        scen_tab[code] = sm_
        evA = P[SETS].T @ sm_.fillna(0)                      # data-based sets A, B, C
        evJ = JUDGMENT.T @ sm_.fillna(0)                     # judgment theses
        sell = x3[lab.isin(SCEN[:2])].dropna()
        t6 = sell.mean() / (sell.std() / np.sqrt(len(sell) / 3)) if len(sell) > 3 else np.nan
        vol = m.std() * np.sqrt(12)
        s = dict(C1=sign_score(t1), C2=sign_score(t2), C3=1 if (h1 > 0 and h2 > 0) else (-1 if (h1 < 0 and h2 < 0) else 0),
                 C4=all_sign(evA.values), C5=all_sign(evJ.values), C6=sign_score(t6) if np.isfinite(t6) else 0)
        rows.append(dict(code=code, thesis=name, **s, total=sum(s.values()),
                         mean_pp_yr=mu * 12, t_mean=t1, alpha_pp_yr=a * 12, t_alpha=t2, beta=beta,
                         half1_pp_yr=h1 * 12, half2_pp_yr=h2 * 12, vol_pp_yr=vol, ir=mu * 12 / vol,
                         **{f"ev_{k.split()[0]}": v for k, v in evA.items()},
                         **{f"ev_{k}": v for k, v in evJ.items()},
                         selloff_3m=sell.mean(), t_selloff=t6, n_selloff=len(sell),
                         ev_A_at_TE=evA.iloc[0] * TE_BUDGET / vol, n_months=len(m), start=m.index[0], end=m.index[-1]))
    T = pd.DataFrame(rows).set_index("code")
    T = T.sort_values(["total", "ev_A", "ir"], ascending=False)
    T["rank"] = np.arange(1, len(T) + 1)
    return T, pd.DataFrame(scen_tab)


def robustness(R):
    rows = []
    for st in SAMPLE_STARTS:
        for name, prm in SCEN_SPECS:
            b = build({**prm, "start": max(st, "2003-01-31")})
            T, _ = score(R, b, st)
            top = T["total"].max()
            for code, r in T.iterrows():
                rows.append(dict(sample=st[:4], scen_spec=name, code=code, total=r["total"],
                                 first=r["total"] == top, **{c: r[c] for c in CRIT}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ valuation context (Bloomberg; not committed)
def valuation_context():
    from src.valuation import load
    D = load()
    PE, EPS = D["PE"], D["EPS"]
    m = lambda t: "XLK" if t == "VGT" else t
    rows = {}
    for code, (_, L, S) in THESES.items():
        def hpe(w):
            w = {m(k): v for k, v in w.items()}
            return 1 / sum(v / PE[k] for k, v in w.items())
        def rev(w):
            w = {m(k): v for k, v in w.items()}
            return sum(v * np.log(EPS[k].where(EPS[k] > 0)).diff(3) for k, v in w.items())
        rel = (hpe(L) / hpe(S)).dropna()
        rr = (rev(L) - rev(S)).dropna()
        rows[code] = dict(rel_pe_now=rel.iloc[-1], rel_pe_pct=(rel < rel.iloc[-1]).mean() * 100,
                          rel_rev_3m_pp=rr.iloc[-1] * 100, rev_pct=(rr < rr.iloc[-1]).mean() * 100, asof=rel.index[-1])
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ chart
def plot_scores(T, path):
    M = T[CRIT].astype(float)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(M.values, cmap="RdYlGn", vmin=-1.3, vmax=1.3, aspect="auto")
    sym = {1: "+1", 0: "0", -1: "−1"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, sym[int(M.values[i, j])], ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(CRIT))); ax.set_xticklabels([f"{c}\n{CRIT_NAME[c]}" for c in CRIT], fontsize=8.5)
    ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{T.loc[c, 'thesis']}  (total {int(T.loc[c, 'total']):+d})" for c in M.index], fontsize=9)
    ax.set_title("Thesis tournament: pre-registered scores (+1 supports, −1 contradicts)", fontsize=10.5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    b = build()
    from src.portfolio import etf_returns
    R = etf_returns()
    T, SC = score(R, b, SAMPLE_STARTS[0])
    print("=" * 100 + "\nTHESIS TOURNAMENT (rules: docs/tournament_preregistration.md)\n" + "=" * 100)
    print(f"Returns {T['start'].min():%Y-%m} to {T['end'].max():%Y-%m}; scenario windows {b['W'].index[0]:%Y-%m} to {b['W'].index[-1]:%Y-%m}\n")
    print(T[["thesis"] + CRIT + ["total", "rank"]].to_string())
    stats = ["mean_pp_yr", "t_mean", "alpha_pp_yr", "t_alpha", "beta", "half1_pp_yr", "half2_pp_yr", "vol_pp_yr", "ir",
             "ev_A", "ev_B", "ev_C", "ev_Inflation persists", "ev_Higher for longer", "ev_Hard landing", "ev_Soft landing",
             "selloff_3m", "t_selloff", "n_selloff", "ev_A_at_TE"]
    print("\nStatistics behind the scores:\n", T[stats].round(2).T.to_string())
    print("\nMean 3-month tilt return by scenario (%):\n", SC.round(2).to_string())
    T.to_csv(OUT / "tournament_baseline.csv", float_format="%.4f"); SC.round(4).to_csv(OUT / "tournament_scenarios.csv")
    plot_scores(T, OUT / "tournament_scores.png")

    print("\nRobustness: 3 sample starts x 7 scenario specs = 21 settings")
    Rb = robustness(R)
    Rb.to_csv(OUT / "tournament_robustness.csv", index=False)
    summ = Rb.groupby("code").agg(min_total=("total", "min"), max_total=("total", "max"), median_total=("total", "median"),
                                  first_in=("first", "sum")).reindex(T.index)
    for c in CRIT:
        summ[c + "_range"] = Rb.groupby("code")[c].agg(lambda x: f"{x.min():+d}..{x.max():+d}").reindex(T.index)
    summ.insert(0, "thesis", T["thesis"])
    print(summ.to_string())

    top = T.index[0]
    strongest = summ.loc[top, "first_in"] >= 14            # baseline leader AND first (ties count) in >= 14/21
    print("\nVERDICT (pre-registered language)")
    print(f"  Strongest backing: {T.loc[top, 'thesis'] if strongest else 'none clearly (baseline leader: ' + T.loc[top, 'thesis'] + ')'}"
          f"  [first in {int(summ.loc[top, 'first_in'])}/21 settings]")
    for c, r in T.iterrows():
        v = "SUPPORTED" if r["total"] >= 2 else ("CONTRADICTED" if r["total"] <= -2 else "mixed")
        print(f"  {c} {r['thesis']}: total {int(r['total']):+d} -> {v}")

    try:
        V = valuation_context()
        V.to_csv(OUT / "tournament_valuation_context.csv")
        print("\nValuation context, long vs short basket (Bloomberg; NOT scored; file not committed):\n", V.round(2).to_string())
    except Exception as e:
        print(f"\nValuation context skipped (Bloomberg files needed): {str(e)[:80]}")


if __name__ == "__main__":
    main()
