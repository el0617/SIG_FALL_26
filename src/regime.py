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

def plot_quadrant(g, i, path, months=60):
    tr = pd.concat([g, i], axis=1, keys=["g", "i"]).dropna().iloc[-months:]
    reg = classify(tr["g"], tr["i"])
    share = reg.value_counts(normalize=True).mul(100)

    # Symmetric limits -> four identical squares
    L = max(tr["i"].abs().max(), tr["g"].abs().max()) * 1.15
    L = np.ceil(L * 4) / 4

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")

    # Quadrant shading (axes-fraction bounds; 0.5 == zero because limits are symmetric)
    shades = {
        "Goldilocks":  (0.0, 0.5, 0.5, 1.0, "#2e7d32"),   # x: left,  y: top
        "Reflation":   (0.5, 1.0, 0.5, 1.0, "#f9a825"),   # x: right, y: top
        "Stagflation": (0.5, 1.0, 0.0, 0.5, "#c62828"),   # x: right, y: bottom
        "Slowdown":    (0.0, 0.5, 0.0, 0.5, "#1565c0"),   # x: left,  y: bottom
    }
    for name, (x0, x1, y0, y1, col) in shades.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes,
                                   color=col, alpha=0.08, zorder=0))

    # Quadrant labels with share of the window spent there
    pos = {"Goldilocks": (0.02, 0.98, "left", "top"), "Reflation": (0.98, 0.98, "right", "top"),
           "Stagflation": (0.98, 0.02, "right", "bottom"), "Slowdown": (0.02, 0.02, "left", "bottom")}
    for name, (x, y, ha, va) in pos.items():
        ax.text(x, y, f"{name}\n{share.get(name, 0):.0f}% of months", transform=ax.transAxes,
                ha=ha, va=va, fontsize=12, fontweight="bold", alpha=0.6)

    ax.axhline(0, c="k", lw=1); ax.axvline(0, c="k", lw=1)

    # Month-to-month path: every month is a point, linked in sequence
    ax.plot(tr["i"], tr["g"], "-", color="grey", lw=0.8, alpha=0.6, zorder=2)
    c = np.arange(len(tr))
    sc = ax.scatter(tr["i"], tr["g"], c=c, cmap="viridis", s=38, edgecolor="white",
                    linewidth=0.5, zorder=3)
    ax.scatter(tr["i"].iloc[-1], tr["g"].iloc[-1], s=260, facecolors="none",
               edgecolors="red", linewidths=2.5, zorder=4)

    # Date labels: first, every 6th month, and latest
    for k in sorted(set([0, len(tr) - 1] + list(range(0, len(tr), 6)))):
        ax.annotate(tr.index[k].strftime("%b-%y"), (tr["i"].iloc[k], tr["g"].iloc[k]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    ticks = np.linspace(0, len(tr) - 1, 6, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([tr.index[k].strftime("%b %Y") for k in ticks])

    ax.set_xlabel("Inflation z-score (3m avg)  →  higher inflation pressure")
    ax.set_ylabel("Growth z-score (3m avg)  →  stronger growth")
    ax.set_title(f"Macro regime, last {len(tr)} months "
                 f"({tr.index[0]:%b %Y} to {tr.index[-1]:%b %Y}); red ring = latest")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

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