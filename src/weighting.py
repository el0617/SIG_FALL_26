"""Transparent risk-budget weighting engine (Task 2).

Run:
    python -m src.weighting

Rules: docs/weighting_preregistration.md (committed before this file). Start from the IVV benchmark, add three
bets with stated reasons, funded pro rata from the other holdings, and scale to a fixed tracking-error budget:
  B1 defensive insurance (XLP, XLU, XLV)         - strong evidence (Task 1: alpha t 2.0, sell-off t 4.9)
  B2 breakeven hedge (XLE, XLB)                  - sized to halve B1's breakeven beta
  B3 growth underweight (VGT, XLY)               - team thesis, conviction version only, half of B1's standalone TE
Versions: Conservative (B1+B2, TE 1.5%), Base (B1+B2, 2.5%), Conviction (B1+B2+B3, 3.5%).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio import (load_portfolio, etf_returns, drift, apply_trades, etf_sensitivities, HOLDINGS,
                           CANDIDATES, RECOMMENDED)
from src.policy import build_state
from src.scenarios import build, active_3m, SCEN, SETS
from src.tournament import JUDGMENT

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
DEF, INF, GRW = ["XLP", "XLU", "XLV"], ["XLE", "XLB"], ["VGT", "XLY"]
OTHER = ["XLC", "XLF", "XLI", "SCHH"]
GROUPS = {"Defensives": DEF, "Inflation hedges": INF, "Growth": GRW, "Other": OTHER}
WINDOW, STAB_WINDOWS = 60, (36, 96)
HEDGE = 0.5
VERSIONS = {"Conservative": (("B1", "B2"), 1.5), "Base": (("B1", "B2"), 2.5), "Conviction": (("B1", "B2", "B3"), 3.5)}
PROPOSAL = "Core + trim energy and growth underweight"
AUM = 100.0   # $ millions


# ------------------------------------------------------------------ bets
def unit_bet(b, long=None, short=None):
    """Active vector (pp) for +1pp long in `long` (equal) or -1pp in `short` (pro rata to benchmark), funded pro rata."""
    v = pd.Series(0.0, index=HOLDINGS)
    if long:
        v[long] = 1.0 / len(long)
        rest = [h for h in HOLDINGS if h not in long]
        v[rest] = -b[rest] / b[rest].sum()
    else:
        v[short] = -b[short] / b[short].sum()
        rest = [h for h in HOLDINGS if h not in short]
        v[rest] = b[rest] / b[rest].sum()
    return v


def cov(R, window):
    return R[HOLDINGS].iloc[-window:].cov()


def te(x, C):
    """Annualized TE (%) of active vector x (pp) given monthly % covariance C."""
    xv = (x / 100).reindex(C.index).values
    return float(np.sqrt(12 * xv @ C.values @ xv))


def be_beta(x, sens):
    return float((x / 100 * sens["be_full"] / 4).sum())       # % vs SPY per +25bp breakeven


def construct(b, sens, C, bets, target):
    v1, v2, v3 = unit_bet(b, long=DEF), unit_bet(b, long=INF), unit_bet(b, short=GRW)
    h = -HEDGE * be_beta(v1, sens) / be_beta(v2, sens)          # halve B1's breakeven beta
    r = te(v1, C) / (2 * te(v3, C)) if "B3" in bets else 0.0  # B1 : B3 standalone TE = 2 : 1
    u = v1 + h * v2 + r * v3
    x = u * target / te(u, C)
    w = (b + x).round(1)
    w[w.idxmax()] += round(100 - w.sum(), 1)
    if (w < 0).any():
        raise ValueError(f"negative weight: {w[w < 0].to_dict()}")
    return w, dict(hedge_ratio=h, b3_ratio=r, s1=float(target / te(u, C)))


# ------------------------------------------------------------------ metrics
def metrics(name, w, b, R, C, sens, sc):
    x = w - b
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    a = w / 100
    last = R.index[-WINDOW:]
    te_spy = float(np.sqrt(12 * a.values @ rel.loc[last].cov().values @ a.values))
    port = (R[HOLDINGS].loc[last] * a).sum(axis=1)
    beta = float(np.polyfit(R["SPY"].loc[last], port, 1)[0])
    act = (rel.fillna({"XLC": 0.0}) * (x / 100)).sum(axis=1).where(rel.drop(columns="XLC").notna().all(axis=1)).dropna()
    a3 = act.rolling(3).sum().dropna()
    var3 = -np.percentile(a3, 5)
    cvar3 = -a3[a3 <= -var3].mean()
    xv = (x / 100).reindex(C.index).values
    rc = pd.Series(xv * (C.values @ xv) / (xv @ C.values @ xv) * 100, index=C.index)
    W, lab, P = sc["W"], sc["lab"], sc["P"]
    pay = (active_3m(R, w) - active_3m(R, b)).reindex(W.index)
    by = pay.groupby(lab).mean().reindex(SCEN).fillna(0)
    ev = {f"exp_{k.split()[0]}": float(P[k] @ by) for k in SETS}
    ev.update({f"exp_{k}": float(JUDGMENT[k] @ by) for k in JUDGMENT})
    sell = pay[lab.isin(SCEN[:2])].mean()
    m = dict(portfolio=name, te_bench=te(x, C), te_spy=te_spy, beta=beta, cvar3=cvar3,
             be_25bp=be_beta(x, sens), real_50bp_post=float((x / 100 * sens["real_post"] / 2).sum()),
             selloff_3m=float(sell), **ev, gross_active=float(x.abs().sum() / 2))
    m.update({f"risk_{g}": float(rc[c].sum()) for g, c in GROUPS.items()})
    return m


# ------------------------------------------------------------------ chart
def plot_active(A, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = list(A.columns); x = np.arange(len(A)); bw = 0.8 / len(cols)
    colors = {"Current": "#90a4ae", "Proposal": "#b0bec5", "Conservative": "#a5d6a7", "Base": "#2e7d32", "Conviction": "#1b5e20"}
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * bw, A[c], bw, label=c, color=colors.get(c))
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(A.index)
    ax.set_ylabel("Active weight vs IVV sector benchmark (pp)")
    ax.set_title("Weighting engine: active weights by version (bets funded pro rata from the other holdings)", fontsize=10)
    ax.legend(fontsize=8, ncol=5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    p = load_portfolio(); b = p["benchmark"].reindex(HOLDINGS)
    R = etf_returns(); S = build_state(); sens = etf_sensitivities(R, S)
    w0, months = drift(p["weight"], R)
    w0 = w0.reindex(HOLDINGS)
    C = cov(R, WINDOW)
    sc = build()

    Wt, info = {"Current": w0, "Proposal": apply_trades(w0, CANDIDATES[PROPOSAL]).reindex(HOLDINGS),
                "XLP->XLF only": apply_trades(w0, CANDIDATES[RECOMMENDED]).reindex(HOLDINGS)}, {}
    for v, (bets, target) in VERSIONS.items():
        Wt[v], info[v] = construct(b, sens, C, bets, target)
    Wt = pd.DataFrame(Wt)
    print("=" * 100 + f"\nWEIGHTING ENGINE (rules: docs/weighting_preregistration.md); returns to {R.index[-1]:%b %Y}\n" + "=" * 100)
    print("Construction:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in info.items()})
    print("\nWeights (%):\n", Wt.assign(benchmark=b).round(1).to_string())
    A = Wt.sub(b, axis=0)
    print("\nActive weights (pp):\n", A.round(1).to_string())
    T = Wt.sub(w0, axis=0)
    print("\nTrades from current (pp; = $m on $100M):\n", T[list(VERSIONS)].round(1).to_string())

    M = pd.DataFrame([metrics(n, Wt[n], b, R, C, sens, sc) for n in Wt.columns]).set_index("portfolio")
    print("\nMetrics:\n", M.round(2).T.to_string())

    # stability of Base across covariance windows
    stab = {f"{WINDOW}m": A["Base"]}
    for wdw in STAB_WINDOWS:
        wb, _ = construct(b, sens, cov(R, wdw), VERSIONS["Base"][0], VERSIONS["Base"][1])
        stab[f"{wdw}m"] = wb - b
        stab[f"TE60_of_{wdw}m"] = pd.Series(te(wb - b, C), index=HOLDINGS)
    St = pd.DataFrame(stab)
    print("\nBase active weights under other covariance windows (pp):\n", St.round(2).to_string())

    # pre-registered checks
    c, bm = M.loc["Current"], M.loc["Base"]
    dev = max((St[f"{wdw}m"] - St[f"{WINDOW}m"]).abs().max() for wdw in STAB_WINDOWS)
    sign_ok = all(((np.sign(St[f"{wdw}m"]) == np.sign(St[f"{WINDOW}m"])) | (St[f"{WINDOW}m"].abs() < 0.05)).all() for wdw in STAB_WINDOWS)
    checks = pd.DataFrame([
        ("1 TE vs benchmark below Current", f"{bm['te_bench']:.2f}% vs {c['te_bench']:.2f}%", bm["te_bench"] < c["te_bench"]),
        ("2 Sell-off payoff >= 75% of Current", f"{bm['selloff_3m']:+.2f}% vs {c['selloff_3m']:+.2f}%", bm["selloff_3m"] >= 0.75 * c["selloff_3m"]),
        ("3 |breakeven beta| below Current", f"{bm['be_25bp']:+.3f} vs {c['be_25bp']:+.3f}", abs(bm["be_25bp"]) < abs(c["be_25bp"])),
        ("4 Expected (set A) within 0.10pp of Current", f"{bm['exp_A']:+.3f} vs {c['exp_A']:+.3f}", bm["exp_A"] >= c["exp_A"] - 0.10),
        ("5 Stable across 36/96m covariances", f"max change {dev:.2f}pp, signs kept {sign_ok}", sign_ok and dev <= 1.5)],
        columns=["check", "value", "pass"])
    print("\nPre-registered checks (Base vs Current):\n", checks.to_string(index=False))
    print(f"\nVERDICT: {'Base is the engine recommendation (5/5 checks)' if checks['pass'].all() else 'no version recommended by the engine: ' + str(int(checks['pass'].sum())) + '/5 checks'}")

    Wt.assign(benchmark=b).round(2).to_csv(OUT / "weighting_weights.csv")
    T.round(2).to_csv(OUT / "weighting_trades.csv")
    M.round(4).to_csv(OUT / "weighting_metrics.csv")
    St.round(3).to_csv(OUT / "weighting_stability.csv")
    checks.to_csv(OUT / "weighting_checks.csv", index=False)
    plot_active(A[["Current", "Proposal", "Conservative", "Base", "Conviction"]], OUT / "weighting_active.png")


if __name__ == "__main__":
    main()
