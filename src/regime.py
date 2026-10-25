import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from src.data import fred, monthly, optional_csv, sector_returns
from src.utils import (ann_change, composite, zrobust, conditional_stats,
                       episode_table, ORDER)

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
LAG = 1
BAND = 0.25      # neutral zone half-width, in composite z units (calibrate: see Step 4)
WINDOW = 120     # 10-year rolling baseline for z-scores
SMOOTH = 3       # trailing months of smoothing

CORE_G = ["payrolls", "indpro", "real_retail", "claims_inv", "ism_no"]
CORE_I = ["core_cpi", "core_pce", "ism_prices"]

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

def markov_probs(g):
    """2-state switching model on the growth composite. Filtered (not smoothed) probabilities = real-time."""
    y = g.dropna()
    res = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
    low = 0 if res.params["const[0]"] < res.params["const[1]"] else 1
    return res.filtered_marginal_probabilities[low].rename("p_weak_growth"), res

def plot_quadrant(g, i, path, months=60, band=BAND):
    tr = pd.concat([g, i], axis=1, keys=["g", "i"]).dropna().iloc[-months:]
    reg = classify(tr["g"], tr["i"], band)
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

    ax.axhspan(-band, band, color="grey", alpha=0.12, zorder=1)
    ax.axvspan(-band, band, color="grey", alpha=0.12, zorder=1)

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
             f"({tr.index[0]:%b %Y} to {tr.index[-1]:%b %Y}); red ring = latest"
             f" | {share.get('Transition', 0):.0f}% in transition band (grey)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def classify(g, i, band=BAND):
    conds = [(g.abs() < band) | (i.abs() < band),
             (g > 0) & (i <= 0), (g > 0) & (i > 0),
             (g <= 0) & (i > 0), (g <= 0) & (i <= 0)]
    names = ["Transition", "Goldilocks", "Reflation", "Stagflation", "Slowdown"]
    return pd.Series(np.select(conds, names, default=""), index=g.index).replace("", np.nan)

def make_composite(z, core, smooth=SMOOTH):
    cols = [c for c in core if c in z.columns]
    print(f"  composite uses: {cols}")
    return composite(z[cols], min_components=len(cols)).rolling(smooth).mean()

def build_regime(window=WINDOW, smooth=SMOOTH, band=BAND, core_g=CORE_G, core_i=CORE_I):
    G, I = build_growth(), build_inflation()
    g = make_composite(zrobust(G, window), core_g, smooth).rename("growth_z")
    i = make_composite(zrobust(I, window), core_i, smooth).rename("inflation_z")
    return g, i, classify(g, i, band).rename("regime")

def plot_edge_heatmap(stats, eps, path):
    e = stats.pivot(index="sector", columns="state", values="edge")
    t = stats.pivot(index="sector", columns="state", values="t")
    months = stats.groupby("state")["months"].first()
    cols = [c for c in ORDER if c in e.columns]
    e, t = e[cols], t[cols]
    v = np.nanmax(np.abs(e.values))
    fig, ax = plt.subplots(figsize=(1.7 * len(cols) + 2, 0.65 * len(e) + 2.2))
    ax.imshow(e.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    for r in range(e.shape[0]):
        for c in range(e.shape[1]):
            val, tv = e.values[r, c], t.values[r, c]
            if np.isnan(val):
                continue
            sig = abs(tv) >= 2
            ax.text(c, r, f"{val:+.1f}\n(t={tv:.1f})", ha="center", va="center", fontsize=8,
                    fontweight="bold" if sig else "normal", alpha=1 if sig else 0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{int(eps.loc[c, 'episodes'])} episodes / {months[c]} mo" for c in cols],
                       fontsize=8)
    ax.set_yticks(range(len(e))); ax.set_yticklabels(e.index)
    ax.set_title("Annualized excess return vs SPY, relative to all-months average (pp)\n"
                 "bold = |t| >= 2 (HAC);  faded = not distinguishable from noise", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

def main():
    print("Building regime (robust z, 10y window, fixed composition):")
    g, i, regime = build_regime()
    state = pd.concat([g, i, regime, g.diff(3).rename("growth_mom")], axis=1)
    state.to_csv(OUT / "regime_state.csv")
    print("\nCurrent state (last 6 months):\n", state.dropna().tail(6).round(2))

    # Diagnostics for fixes #2 and #5
    ii = i.dropna()
    print("\nShare of months with inflation_z > 0, by decade (should no longer be ~0 pre-2020):\n",
          (ii > 0).groupby(ii.index.year // 10 * 10).mean().round(2))
    print(f"\nTransition share (|z| < {BAND}): {(regime == 'Transition').sum() / regime.notna().sum():.0%}")

    # Fixes #4 and #6: edges, t-stats, episodes
    rets = sector_returns()
    lab = regime.reindex(rets.index).iloc[:-1]       # last month has no forward return
    stats = conditional_stats(lab, rets)
    eps = episode_table(lab)
    print("\nEpisodes per state:\n", eps)
    for name in ["edge", "t", "hit"]:
        print(f"\n{name} (sector x regime):\n",
              stats.pivot(index="sector", columns="state", values=name)[
                  [c for c in ORDER if c in stats.state.unique()]].round(1))
    stats.to_csv(OUT / "regime_stats_long.csv", index=False)
    eps.to_csv(OUT / "regime_episodes.csv")
    plot_edge_heatmap(stats, eps, OUT / "regime_edge_heatmap.png")

    p, _ = markov_probs(g)
    p.to_csv(OUT / "markov_prob_weak_growth.csv")
    print("\nMarkov P(weak-growth), latest:", round(float(p.iloc[-1]), 2))
    plot_quadrant(g, i, OUT / "regime_quadrant.png")

def robustness():
    """How much do labels change under reasonable alternative specs?"""
    print("Baseline:"); base = build_regime()[2]
    rows = []
    for w in [120, 180, None]:
        for s in [1, 3]:
            print(f"Spec window={w}, smooth={s}:")
            r = build_regime(window=w, smooth=s)[2]
            both = pd.concat([base, r], axis=1, keys=["b", "r"]).dropna()
            rows.append({"spec": f"window={w or 'expanding'}, smooth={s}",
                         "agree_%": round((both.b == both.r).mean() * 100),
                         "latest": r.dropna().iloc[-1]})
    print("With breakevens (starts ~2008):")
    r = build_regime(core_i=CORE_I + ["be_5y5y"])[2]
    both = pd.concat([base, r], axis=1, keys=["b", "r"]).dropna()
    rows.append({"spec": "+ be_5y5y (2008+ only)", "agree_%": round((both.b == both.r).mean() * 100),
                 "latest": r.dropna().iloc[-1]})
    print("\n", pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()