import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from src.data import fred, monthly, optional_csv, sector_returns
from src.utils import ann_change, zexp, composite, conditional_returns

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
LAG = 1  # conservative publication lag (months) for monthly macro releases

def build_growth():
    c = pd.DataFrame({
        "payrolls":    ann_change(monthly(fred("PAYEMS")), 3).shift(LAG),
        "indpro":      ann_change(monthly(fred("INDPRO")), 3).shift(LAG),
        "real_retail": ann_change(monthly(fred("RRSFS")), 3).shift(LAG),
        "claims_inv":  -monthly(fred("ICSA"), "mean").pct_change(3) * 100,
    })
    ism = optional_csv("ism_new_orders")
    if ism is not None:
        c["ism_no"] = (monthly(ism) - 50).shift(LAG)
    return c

def build_inflation():
    c = pd.DataFrame({
        "core_cpi": ann_change(monthly(fred("CPILFESL")), 3).shift(LAG),
        "core_pce": ann_change(monthly(fred("PCEPILFE")), 3).shift(LAG),
        "be_5y5y":  monthly(fred("T5YIFR"), "mean"),
    })
    ism_p = optional_csv("ism_prices")
    if ism_p is not None:
        c["ism_prices"] = (monthly(ism_p) - 50).shift(LAG)
    return c

def classify(g, i):
    conds = [(g > 0) & (i <= 0), (g > 0) & (i > 0), (g <= 0) & (i > 0), (g <= 0) & (i <= 0)]
    names = ["Goldilocks", "Reflation", "Stagflation", "Slowdown"]
    return pd.Series(np.select(conds, names, default=""), index=g.index).replace("", np.nan)

def markov_probs(g):
    """2-state switching model on the growth composite. Filtered (not smoothed) probabilities = real-time."""
    y = g.dropna()
    res = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
    low = 0 if res.params["const[0]"] < res.params["const[1]"] else 1
    return res.filtered_marginal_probabilities[low].rename("p_weak_growth"), res

def plot_quadrant(g, i, path):
    tr = pd.concat([g, i], axis=1, keys=["g", "i"]).dropna().iloc[-24:]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(tr.i, tr.g, "-o", alpha=0.5)
    ax.scatter(tr.i.iloc[-1], tr.g.iloc[-1], s=180, c="red", zorder=5, label=tr.index[-1].strftime("%b %Y"))
    ax.axhline(0, c="k", lw=0.8); ax.axvline(0, c="k", lw=0.8)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for txt, x, y in [("Goldilocks", xl[0], yl[1]), ("Reflation", xl[1], yl[1]),
                      ("Stagflation", xl[1], yl[0]), ("Slowdown", xl[0], yl[0])]:
        ax.text(x, y, txt, ha="left" if x == xl[0] else "right",
                va="top" if y == yl[1] else "bottom", fontsize=12, alpha=0.5)
    ax.set_xlabel("Inflation z-score (3m avg)"); ax.set_ylabel("Growth z-score (3m avg)")
    ax.set_title("Macro regime: last 24 months"); ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")

def main():
    G, I = build_growth(), build_inflation()
    g = composite(zexp(G)).rolling(3).mean().rename("growth_z")
    i = composite(zexp(I)).rolling(3).mean().rename("inflation_z")
    regime = classify(g, i).rename("regime")

    state = pd.concat([g, i, regime, g.diff(3).rename("growth_mom")], axis=1)
    state.to_csv(OUT / "regime_state.csv")
    print("\nCurrent state (last 6 months):\n", state.dropna().tail(6).round(2))

    rets = sector_returns()
    mean, hit, n = conditional_returns(regime, rets)
    print("\nAnnualized excess return vs SPY by regime (%):\n", mean)
    print("\nHit rate (% of months outperforming):\n", hit)
    print("\nSample size (months):\n", n)
    mean.to_csv(OUT / "regime_excess_returns.csv"); hit.to_csv(OUT / "regime_hit_rates.csv")

    p, res = markov_probs(g)
    p.to_csv(OUT / "markov_prob_weak_growth.csv")
    print("\nMarkov P(weak-growth regime), latest:", round(float(p.iloc[-1]), 2))

    plot_quadrant(g, i, OUT / "regime_quadrant.png")

if __name__ == "__main__":
    main()