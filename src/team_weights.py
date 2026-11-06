"""Team final weights (decided 5 Nov 2026) evaluated against today's S&P 500 sector weights.

Run:
    python -m src.team_weights

POST HOC: the team set these weights after seeing the engine's Base result. They follow the direction of Base
(smaller growth underweight, smaller energy overweight) with team modifications. This module reports the same
risk measures and the same five Base-vs-Current checks for context; it is not a pre-registered verdict.

Conventions
- Portfolio: 60% equities, 40% fixed income and cash. Equity weights = portfolio weight x 100/60.
- The November ("old") equity sleeve was 57.93% of the portfolio, so its equity weights sum to 96.55%. The
  2.07pp of the portfolio held outside equities relative to the 60% allocation is treated as cash (zero return)
  in the equity-sleeve measures, so the measures match the weights shown on the slides.
- Benchmark: S&P 500 GICS sector weights at the 4 Nov 2026 close (ChartRow estimate from SPY holdings,
  normalized from 100.3% to 100%), mapped to the eleven sector ETFs.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from src.portfolio import etf_returns, etf_sensitivities, HOLDINGS, load_portfolio
from src.policy import build_state
from src.scenarios import build, summarize, verdict, SCEN, SETS, H
from src.consistency import M8_SPECS
from src.energy import stress
from src import weighting as wg

OUT = Path("outputs")
EQ = 100 / 60
CASH = "CASH"

# % of the $100M portfolio (team decision, 5 Nov 2026)
TEAM = pd.DataFrame({
    "old": {"VGT": 16.47, "XLV": 6.59, "XLC": 4.48, "XLF": 6.12, "XLP": 6.08, "XLY": 3.70, "XLB": 3.70, "XLI": 3.24,
            "XLE": 4.99, "SCHH": 1.14, "XLU": 1.42, "IGSB": 3.00, "IBID": 9.00, "AGZ": 9.00, "VGIT": 13.50,
            "Cash": 4.57, "VCSH": 3.00, "Gold": 4.31, "Silver": 1.24},
    "new": {"VGT": 19.50, "XLV": 7.00, "XLC": 5.50, "XLF": 6.00, "XLP": 4.00, "XLY": 5.00, "XLB": 2.00, "XLI": 4.00,
            "XLE": 3.00, "SCHH": 2.50, "XLU": 1.50, "IGSB": 10.05, "IBID": 6.76, "AGZ": 8.09, "VGIT": 6.79,
            "Cash": 4.00, "VCSH": 0.00, "Gold": 4.31, "Silver": 0.00}})
# 6 Nov: gold (4.31%, held) is funded from IGSB (14.36 -> 10.05); silver (1.24%) is sold. Gold and silver are not in
# the team's current table, which sums to 100% without them, so they are shown on a separate line.
FI = ["IGSB", "IBID", "AGZ", "VGIT", "VCSH", "Cash"]
COMMODITIES = ["Gold", "Silver"]
DURATION = {"IGSB": 2.6, "VCSH": 2.6, "VGIT": 4.8, "AGZ": 3.42, "IBID": 0.54, "Cash": 0.0}   # IGSB, VCSH estimated

# S&P 500 sector weights, 4 Nov 2026 close (ChartRow, SPY holdings), by ETF proxy
SP500_RAW = {"VGT": 36.9, "XLF": 12.6, "XLV": 10.3, "XLY": 9.7, "XLC": 8.5, "XLI": 8.2, "XLP": 4.9, "XLE": 3.6,
             "XLU": 2.1, "SCHH": 1.8, "XLB": 1.7}


def benchmark():
    b = pd.Series(SP500_RAW).reindex(HOLDINGS)
    b = (b / b.sum() * 100).round(2)
    b["VGT"] += round(100 - b.sum(), 2)
    return b


def equity(col):
    w = (TEAM.loc[HOLDINGS, col] * EQ)
    return pd.concat([w, pd.Series({CASH: 100 - w.sum()})])


# ------------------------------------------------------------------ returns with a cash leg
def frame(R):
    X = R[HOLDINGS].copy()
    X["XLC"] = X["XLC"].fillna(R["SPY"])
    X = X.where(R[[h for h in HOLDINGS if h != "XLC"]].notna().all(axis=1))
    X[CASH] = 0.0
    return X


def port(X, w):
    return (X[w.index] * (w / 100)).sum(axis=1, min_count=1).where(X[HOLDINGS].notna().all(axis=1))


def fwd(x):
    return ((1 + x / 100).rolling(H).apply(np.prod, raw=True) - 1).shift(-H) * 100


def spy_sens(R, S):
    dX = S[["real10y", "breakeven10y"]].diff()
    row = {"beta": 0.0}
    for tag, start, drop22 in [("full", "2003-02-28", False), ("post", "2020-01-01", False), ("ex22", "2020-01-01", True)]:
        dd = pd.concat([R["SPY"].rename("y"), dX], axis=1, sort=True).loc[start:].dropna()
        if drop22:
            dd = dd.drop(dd.loc["2022"].index)
        res = sm.OLS(dd["y"].values, sm.add_constant(dd[["real10y", "breakeven10y"]].values)).fit()
        row[f"real_{tag}"], row[f"be_{tag}"] = -res.params[1], -res.params[2]   # cash minus SPY
    return pd.Series(row, name=CASH)


def payoffs_cash(X, R, W, lab, Wts, sens, tag):
    """Module 8 payoffs vs SPY (scenarios.payoffs) for weight vectors that may include a cash leg."""
    med = W.groupby(lab)[["d_real", "d_be", "spy"]].median().reindex(SCEN)
    emp, mod, neg = {}, {}, {}
    for name, w in Wts.items():
        a = (fwd(port(X, w)) - fwd(R["SPY"])).reindex(W.index)
        emp[name] = a.groupby(lab).mean().reindex(SCEN)
        neg[name] = (a < 0).groupby(lab).mean().reindex(SCEN)
        x = w / 100
        s = sens.reindex(w.index)
        mod[name] = pd.Series({k: float((x * (s[f"real_{tag}"] * med.loc[k, "d_real"] / 100
                                           + s[f"be_{tag}"] * med.loc[k, "d_be"] / 100
                                           + (s["beta"] - 1) * med.loc[k, "spy"])).sum()) for k in SCEN})
    return pd.DataFrame(emp), pd.DataFrame(mod), pd.DataFrame(neg)


# ------------------------------------------------------------------ metrics
def metrics(name, w, b, X, R, C, sens, sc):
    bb = pd.concat([b, pd.Series({CASH: 0.0})])
    x = w.reindex(bb.index).fillna(0) - bb
    last = R.index[-wg.WINDOW:]
    p = port(X, w)
    beta = float(np.polyfit(R["SPY"].loc[last], p.loc[last], 1)[0])
    act = (p - port(X, bb)).dropna()
    a3 = act.rolling(3).sum().dropna()
    var3 = -np.percentile(a3, 5)
    cvar3 = -a3[a3 <= -var3].mean()
    xv = (x / 100).reindex(C.index).values
    tev = float(np.sqrt(12 * xv @ C.values @ xv))
    rc = pd.Series(xv * (C.values @ xv) / (xv @ C.values @ xv) * 100, index=C.index)
    Wn, lab, P = sc["W"], sc["lab"], sc["P"]
    pay = (fwd(p) - fwd(port(X, bb))).reindex(Wn.index)
    by = pay.groupby(lab).mean().reindex(SCEN).fillna(0)
    ev = {f"exp_{k.split()[0]}": float(P[k] @ by) for k in SETS}
    s = sens.reindex(x.index)
    st, _, _ = stress(R, float(x["XLE"]))
    m = dict(portfolio=name, te_bench=tev, beta=beta, cvar3=cvar3,
             be_25bp=float((x / 100 * s["be_full"] / 4).sum()),
             selloff_3m=float(pay[lab.isin(SCEN[:2])].mean()), **ev,
             oil_loss=float(st["cost of today's overweight (pp)"].min()), gross_active=float(x.abs().sum() / 2),
             **{f"scen_{k}": float(by[k]) for k in SCEN})
    m.update({f"risk_{g}": float(rc[c].sum()) for g, c in wg.GROUPS.items()})
    m.update({f"act_{g}": float(x[c].sum()) for g, c in wg.GROUPS.items()})
    m["act_Cash"] = float(x[CASH])
    return m, x


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    R = etf_returns(); S = build_state(); sens = etf_sensitivities(R, S)
    sens = pd.concat([sens, spy_sens(R, S).to_frame().T])
    X = frame(R)
    Cm = X[HOLDINGS + [CASH]].loc[R.index[-wg.WINDOW:]].cov()
    b = benchmark()
    sc = build()
    base = pd.read_csv(OUT / "weighting_weights.csv", index_col=0)["Base"].reindex(HOLDINGS)
    Wt = {"Old (5 Nov)": equity("old"), "New (team)": equity("new"),
          "Model Base": pd.concat([base, pd.Series({CASH: 0.0})])}

    # full hedge for the team weights: add the engine's inflation unit bet until breakeven beta is zero
    v2 = wg.unit_bet(b, long=wg.INF)
    xn = Wt["New (team)"].reindex(HOLDINGS) - b
    h = -wg.be_beta(xn, sens.loc[HOLDINGS]) / wg.be_beta(v2, sens.loc[HOLDINGS])
    full = (b + xn + h * v2).round(2)
    Wt["New, full hedge"] = pd.concat([full, pd.Series({CASH: 0.0})])

    rows, A = [], {}
    for n, w in Wt.items():
        m, x = metrics(n, w, b, X, R, Cm, sens, sc)
        rows.append(m); A[n] = x
    M = pd.DataFrame(rows).set_index("portfolio")
    A = pd.DataFrame(A)

    print("Benchmark (S&P 500 sector weights, 4 Nov 2026):\n", b.to_string())
    print("\nEquity weights (% of equities):\n", pd.DataFrame(Wt).round(2).assign(bench=b).to_string())
    print("\nActive weights vs S&P 500 (pp of equities):\n", A.round(2).to_string())
    print(f"\nFull hedge for the team weights: inflation unit bet x {h:.2f}")
    print("\nMetrics:\n", M.round(3).T.to_string())

    o, n = M.loc["Old (5 Nov)"], M.loc["New (team)"]
    checks = pd.DataFrame([
        ("1 TE vs benchmark below Old", f"{n['te_bench']:.2f}% vs {o['te_bench']:.2f}%", n["te_bench"] < o["te_bench"]),
        ("2 Sell-off payoff >= 75% of Old", f"{n['selloff_3m']:+.2f}% vs {o['selloff_3m']:+.2f}%", n["selloff_3m"] >= 0.75 * o["selloff_3m"]),
        ("3 |breakeven beta| below Old", f"{n['be_25bp']:+.3f} vs {o['be_25bp']:+.3f}", abs(n["be_25bp"]) < abs(o["be_25bp"])),
        ("4 Expected (set A) within 0.10pp of Old", f"{n['exp_A']:+.3f} vs {o['exp_A']:+.3f}", n["exp_A"] >= o["exp_A"] - 0.10)],
        columns=["check", "value", "pass"])
    print("\nBase-vs-Current checks applied to New vs Old (context, post hoc):\n", checks.to_string(index=False))

    # Module 8 decision rule vs Old, 12 settings
    sens_x = sens
    res = {k: [] for k in Wt if k != "Old (5 Nov)"}
    wins = {k: [] for k in res}
    for prm in M8_SPECS:
        bb = build(prm)
        E, Mm, NEG = payoffs_cash(X, R, bb["W"], bb["lab"], Wt, sens_x, bb["p"]["model_tag"])
        T = summarize(bb["P"], E, Mm, NEG)
        for k in res:
            v, wn, _ = verdict(T, k, ref="Old (5 Nov)")
            res[k].append(v); wins[k].append(wn)
    print("\nModule 8 decision rule vs Old (12 settings):")
    for k in res:
        print(f"   {k:20s} {pd.Series(res[k]).value_counts().to_dict()}  wins/6 by setting {wins[k]}")

    pd.DataFrame(Wt).round(3).assign(benchmark=b).to_csv(OUT / "team_weights_equity.csv")
    A.round(3).to_csv(OUT / "team_weights_active.csv")
    M.round(4).to_csv(OUT / "team_weights_metrics.csv")
    checks.to_csv(OUT / "team_weights_checks.csv", index=False)
    pd.DataFrame({k: pd.Series(v).value_counts() for k, v in res.items()}).fillna(0).astype(int).to_csv(OUT / "team_weights_m8.csv")


if __name__ == "__main__":
    main()
