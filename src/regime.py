"""Module 1 - Macro regime model (v2).

Run:
    python -m src.regime            # main run: state, playbook, thesis test, nowcast, plots
    python -m src.regime --robust   # specification robustness table (takes 1-3 minutes)

What changed vs v1 (see the deficiency list from our review):
  1. Publication lag no longer drops the newest release (lagged() instead of shift()).
  2. Oct-2025 shutdown hole filled by one-month interpolation (logged); ragged edge carried 1 month.
  3. Z-scores: scale floored at 3/4 of 30-year dispersion + soft squash (no more pinning at 3.0).
  4. Inflation measured vs a Fed-consistent anchor (2.0% core PCE / 2.4% core CPI), not a lagging
     10-year median; the rolling-median version is a robustness spec.
  5. CPI and PCE merged into one 'realized inflation' block (no double counting).
  6. Survey slot: ISM (Bloomberg CSV) if present, else Philly Fed (free FRED) - swapped automatically.
  7. classify(): NaN-safe, band 0.15, hysteresis ('sticky') to cut flicker without adding lag.
  8. Statistics: episode-clustered t, Benjamini-Hochberg q, leave-one-episode-out, credibility flag.
  9. Thesis test: Defensives (XLP, XLU, XLV) minus Growth (XLK, XLY) spread, pre-registered verdict rule.
 10. Timely nowcast (claims, Philly/Empire surveys, breakevens, UMich) plotted next to the lagged composite.
 11. Renamed regimes on all visuals; stale v1 output files removed.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import (fred, fred_optional, level, monthly, optional_csv, sector_returns,
                      complete_months, current_month_end, FILL_LOG)
from src.utils import (ann_change, chg, lagged, zscore, classify, classify_direction, episode_table,
                       forward_excess, conditional_stats, transition_matrix, forward_distribution,
                       ORDER, DISPLAY, DIR_ORDER)

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ specification (frozen before
# looking at returns; every alternative is reported in --robust)
LAG = 1            # months between reference month and month-end availability (hard data)
WINDOW = 120       # 10-year rolling baseline (growth centre, all scales)
SMOOTH = 3         # trailing months of smoothing of each composite
BAND = 0.15        # neutral-zone half-width (composite z units)
STICKY = True      # hysteresis inside the band
INFL_CENTER = "target"                       # "target" (Fed-consistent anchor) or "rolling"
TARGETS = {"core_cpi": 2.4, "core_pce": 2.0}  # core CPI runs ~0.3-0.5pp above core PCE
SURVEY = "auto"    # "auto" = ISM CSVs if present, else Philly Fed; or force "ism" / "fred"
SAMPLE_START = "1999-01-31"                  # sector ETF history

G_COLS = ["payrolls", "indpro", "real_retail", "claims_inv", "survey_orders"]

# Thesis universe. Proxies: VGT -> XLK (VGT starts 2004), SCHH -> IYR (SCHH starts 2011).
# XLC (2018+) is excluded from the regime playbook: too few regime episodes, and it was carved out of
# XLK/XLY, so its history isn't comparable.
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR"]
ROW_NAME = {"XLK": "XLK (VGT)", "IYR": "IYR (SCHH)"}
DEFENSIVE, GROWTH = ["XLP", "XLU", "XLV"], ["XLK", "XLY"]
SPREAD = "DEF-GRW"
SPREADS = {SPREAD: (DEFENSIVE, GROWTH)}

EDGE_LOG = {}  # component -> months carried forward at the ragged edge
COLORS = {"Goldilocks": "#2e7d32", "Reflation": "#f9a825", "Stagflation": "#c62828",
          "Slowdown": "#1565c0", "Transition": "#9e9e9e"}


# ------------------------------------------------------------------ inputs
def finalize(df, name):
    """Common month-end index up to the current month; carry each component's last value forward
    by at most one month at the ragged edge (latest releases arrive on different days)."""
    end = current_month_end()
    df = df.loc[:end]
    df = df.reindex(pd.date_range(df.index.min(), end, freq="ME"))
    for c in df.columns:
        lv = df[c].last_valid_index()
        if lv is not None and lv < df.index[-1]:
            tail = df.index > lv
            filled = df[c].ffill(limit=1)
            df.loc[tail, c] = filled[tail]
            carried = [d.strftime("%Y-%m") for d in df.index[tail] if pd.notna(filled[d])]
            if carried:
                EDGE_LOG[f"{name}.{c}"] = carried
    return df


def survey_orders(source):
    if source in ("auto", "ism"):
        s = optional_csv("ism_new_orders")
        if s is not None:
            return lagged(monthly(s), 1), "ISM manufacturing new orders (Bloomberg, lag 1)"
        if source == "ism":
            raise FileNotFoundError("data/raw/ism_new_orders.csv not found")
    return level("NOCDFSA066MSFRBPHI"), "Philly Fed new orders (FRED, released mid-month, lag 0)"


def survey_prices(source):
    if source in ("auto", "ism"):
        s = optional_csv("ism_prices")
        if s is not None:
            return lagged(monthly(s), 1), "ISM prices paid (Bloomberg, lag 1)"
        if source == "ism":
            raise FileNotFoundError("data/raw/ism_prices.csv not found")
    pp = level("PPCDFSA066MSFRBPHI", optional=True)
    if pp is not None and pp.dropna().index.min() < pd.Timestamp("1990-01-01"):
        return pp, "Philly Fed prices paid (FRED, lag 0)"
    return level("MICH"), "UMich 1-year inflation expectations (FRED, lag 0)"


def build_growth(source=SURVEY):
    so, src = survey_orders(source)
    G = pd.concat({
        "payrolls":      lagged(ann_change(level("PAYEMS")), LAG),
        "indpro":        lagged(ann_change(level("INDPRO")), LAG),
        "real_retail":   lagged(ann_change(level("RRSFS")), LAG),
        "claims_inv":    -chg(level("ICSA", "mean"), 3),          # weekly -> monthly avg, timely
        "survey_orders": so,
    }, axis=1, sort=True)
    return finalize(G, "growth"), src


def build_inflation(source=SURVEY):
    sp, src = survey_prices(source)
    I = pd.concat({
        "core_cpi":      lagged(ann_change(level("CPILFESL")), LAG),
        "core_pce":      lagged(ann_change(level("PCEPILFE")), LAG),
        "survey_prices": sp,
        "be_5y5y":       monthly(fred("T5YIFR"), "mean"),         # robustness only (starts 2003)
    }, axis=1, sort=True)
    return finalize(I, "inflation"), src


def build_regime(window=WINDOW, smooth=SMOOTH, band=BAND, sticky=STICKY, infl_center=INFL_CENTER,
                 survey=SURVEY, with_be=False):
    G, gsrc = build_growth(survey)
    I, isrc = build_inflation(survey)

    zG = zscore(G, window)
    g_raw = zG[G_COLS].mean(axis=1).where(zG[G_COLS].notna().all(axis=1))

    zI = zscore(I, window, center=TARGETS if infl_center == "target" else None)
    blocks = {"realized": zI[["core_cpi", "core_pce"]].mean(axis=1),   # one block, either may lag
              "survey": zI["survey_prices"]}
    if with_be:
        blocks["breakevens"] = zI["be_5y5y"]
    B = pd.concat(blocks, axis=1)
    i_raw = B.mean(axis=1).where(B.notna().all(axis=1))

    g = g_raw.rolling(smooth).mean().rename("growth_z")
    i = i_raw.rolling(smooth).mean().rename("inflation_z")
    return dict(g=g, i=i, g_raw=g_raw, i_raw=i_raw, zG=zG, zI=zI,
                regime=classify(g, i, band, sticky).rename("regime"), sources=(gsrc, isrc))


def build_nowcast(window=WINDOW):
    """Timely read, no smoothing. Monthly surveys published inside the reference month (Philly, Empire,
    UMich) enter at lag 0; ISM (published on the 1st business day of the next month) at lag 1;
    market prices (claims, HY spreads, breakevens) as the month's average."""
    ism_no, ism_inv, ism_p = optional_csv("ism_new_orders"), optional_csv("ism_inventories"), optional_csv("ism_prices")
    hy = optional_csv("hy_oas")
    G = pd.concat({k: v for k, v in {
        "claims_inv": -chg(level("ICSA", "mean"), 3),
        "philly_activity": level("GACDFSA066MSFRBPHI", optional=True),
        "philly_orders": level("NOCDFSA066MSFRBPHI", optional=True),
        "empire_activity": level("GACDISA066MSFRBNY", optional=True),
        "ism_orders_minus_inv": (lagged(monthly(ism_no) - monthly(ism_inv), 1)
                                 if ism_no is not None and ism_inv is not None else None),
        "hy_oas_inv": -monthly(hy, "mean") if hy is not None else None,
    }.items() if v is not None}, axis=1, sort=True)
    be = fred_optional("T5YIE")
    I = pd.concat({k: v for k, v in {
        "breakeven_5y": monthly(be, "mean") if be is not None else None,
        "umich_1y": level("MICH", optional=True),
        "philly_prices": level("PPCDFSA066MSFRBPHI", optional=True),
        "ism_prices": lagged(monthly(ism_p), 1) if ism_p is not None else None,
    }.items() if v is not None}, axis=1, sort=True)
    G, I = finalize(G, "nowcast_g"), finalize(I, "nowcast_i")
    zG, zI = zscore(G, window), zscore(I, window)
    g = zG.mean(axis=1).where(zG.notna().sum(axis=1) >= 2).rename("nowcast_growth")
    i = zI.mean(axis=1).where(zI.notna().sum(axis=1) >= 2).rename("nowcast_inflation")
    return g, i, list(G.columns), list(I.columns)


def markov_probs(g_raw):
    """2-state switching model on the UNSMOOTHED growth composite (robustness check only)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    y = g_raw.dropna()
    res = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
    low = 0 if res.params["const[0]"] < res.params["const[1]"] else 1
    return res.filtered_marginal_probabilities[low].rename("p_weak_growth")


# ------------------------------------------------------------------ returns side
def returns_panel():
    r = complete_months(sector_returns())
    missing = [t for t in ["SPY"] + SECTORS if t not in r.columns]
    if missing:
        raise KeyError(f"prices_monthly.csv lacks {missing}; delete it and rerun to re-download")
    r = r[["SPY"] + SECTORS].loc[SAMPLE_START:]
    return forward_excess(r, spreads=SPREADS)


def thesis_verdict(row):
    """Pre-registered rule: one-sided 5% test on the in-state DEF-GRW mean (episode-clustered).
    Critical value is t(G-1) when the row carries t_crit (conditional_stats rows), else 1.65."""
    if row is None or np.isnan(row["raw_t"]):
        return "NO EVIDENCE"
    crit = row.get("t_crit", 1.65)
    if row["raw_t"] >= crit:
        return "SUPPORTS"
    if row["raw_t"] <= -crit:
        return "CONTRADICTS"
    return "NEUTRAL"


# ------------------------------------------------------------------ plots
def _square_axes(ax, L):
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    shades = {"Goldilocks": (0.0, 0.5), "Reflation": (0.5, 0.5),
              "Stagflation": (0.5, 0.0), "Slowdown": (0.0, 0.0)}
    for name, (x0, y0) in shades.items():
        ax.add_patch(plt.Rectangle((x0, y0), 0.5, 0.5, transform=ax.transAxes,
                                   color=COLORS[name], alpha=0.08, zorder=0))


def plot_quadrant(g, i, labels, path, months=60, band=BAND, nowcast=None):
    tr = pd.concat([g, i], axis=1, keys=["g", "i"]).dropna().iloc[-months:]
    share = labels.reindex(tr.index).value_counts(normalize=True).mul(100)
    L = np.ceil(max(tr.abs().max().max() * 1.15, 1.0) * 4) / 4
    if nowcast is not None:
        L = max(L, np.ceil(max(abs(nowcast[0]), abs(nowcast[1])) * 1.15 * 4) / 4)

    fig, ax = plt.subplots(figsize=(9.5, 9))
    _square_axes(ax, L)
    pos = {"Goldilocks": (0.02, 0.98, "left", "top"), "Reflation": (0.98, 0.98, "right", "top"),
           "Stagflation": (0.98, 0.02, "right", "bottom"), "Slowdown": (0.02, 0.02, "left", "bottom")}
    for name, (x, y, ha, va) in pos.items():
        ax.text(x, y, f"{DISPLAY[name]}\n{share.get(name, 0):.0f}% of months", transform=ax.transAxes,
                ha=ha, va=va, fontsize=11, fontweight="bold", color=COLORS[name], alpha=0.85)
    ax.axhline(0, c="k", lw=1); ax.axvline(0, c="k", lw=1)
    ax.axhspan(-band, band, color="grey", alpha=0.12, zorder=1)
    ax.axvspan(-band, band, color="grey", alpha=0.12, zorder=1)

    ax.plot(tr["i"], tr["g"], "-", color="grey", lw=0.8, alpha=0.6, zorder=2)
    sc = ax.scatter(tr["i"], tr["g"], c=np.arange(len(tr)), cmap="viridis", s=38,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(tr["i"].iloc[-1], tr["g"].iloc[-1], s=260, facecolors="none",
               edgecolors="red", linewidths=2.5, zorder=4, label=f"Composite, {tr.index[-1]:%b %Y}")
    for k in sorted(set([0, len(tr) - 1] + list(range(0, len(tr), 6)))):
        ax.annotate(tr.index[k].strftime("%b-%y"), (tr["i"].iloc[k], tr["g"].iloc[k]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=5)
    if nowcast is not None and not np.isnan(nowcast[0]) and not np.isnan(nowcast[1]):
        ax.scatter(nowcast[1], nowcast[0], marker="*", s=420, c="red", edgecolor="k",
                   zorder=6, label=f"Nowcast (timely data), {nowcast[2]:%b %Y}")
        ax.annotate("", xy=(nowcast[1], nowcast[0]), xytext=(tr["i"].iloc[-1], tr["g"].iloc[-1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2, ls="--"), zorder=5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    ticks = np.linspace(0, len(tr) - 1, 6, dtype=int)
    cbar.set_ticks(ticks); cbar.set_ticklabels([tr.index[k].strftime("%b %Y") for k in ticks])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
    ax.set_xlabel("Inflation pressure, z-score vs Fed-consistent anchor (3m avg)  →")
    ax.set_ylabel("Growth, z-score vs 10-year trend (3m avg)  →")
    ax.set_title(f"Macro regime, {tr.index[0]:%b %Y} to {tr.index[-1]:%b %Y} (monthly)\n"
                 f"grey band = transition zone (|z| < {band}); "
                 f"{share.get('Transition', 0):.0f}% of months in Transition", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_edge_heatmap(stats, eps, path, order=ORDER, display=DISPLAY, what="regime", horizon="Next-month"):
    s = stats[stats["family"] == "sector"]
    cols = [c for c in order if c in set(s["state"])]
    piv = lambda v: s.pivot(index="series", columns="state", values=v).reindex(columns=cols)
    e, t, q, cred = piv("edge"), piv("t"), piv("q"), piv("credible")
    v = np.nanmax(np.abs(e.values))
    fig, ax = plt.subplots(figsize=(2.0 * len(cols) + 2, 0.7 * len(e) + 2.6))
    ax.imshow(e.values.astype(float), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    for r in range(e.shape[0]):
        for c in range(e.shape[1]):
            val = e.values[r, c]
            if pd.isna(val):
                continue
            star = bool(cred.values[r, c])
            loud = star or abs(t.values[r, c]) >= 2
            ax.text(c, r, f"{'★ ' if star else ''}{val:+.1f}\nt={t.values[r, c]:.1f}  q={q.values[r, c]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold" if star else "normal",
                    alpha=1 if loud else 0.45)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{display[c]}\n{int(eps.loc[c, 'episodes'])} episodes / {int(eps.loc[c, 'months'])} mo"
                        for c in cols], fontsize=8)
    ax.set_yticks(range(len(e))); ax.set_yticklabels([ROW_NAME.get(x, x) for x in e.index])
    ax.set_title(f"{horizon} return vs SPY in each {what}, relative to the sector's all-months average "
                 f"(annualized pp)\nt clustered by {what} episode · q = Benjamini-Hochberg FDR · "
                 "★ = q ≤ 0.10, sign survives dropping any one episode, ≥ 5 episodes", fontsize=9.5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_thesis(stats, current, path, expected=None):
    s = stats[stats["series"] == SPREAD].set_index("state").reindex([c for c in ORDER])
    s = s.dropna(subset=["raw"])
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cols = ["#d32f2f" if st == current else "#90a4ae" for st in s.index]
    ax.bar(x, s["raw"], color=cols, alpha=0.85)
    ax.errorbar(x, s["raw"], yerr=[s["raw"] - s["raw_lo"], s["raw_hi"] - s["raw"]],
                fmt="none", ecolor="k", capsize=5, lw=1)
    ax.set_ylim(min(s["raw_lo"].min(), 0) * 1.15 - 1, max(s["raw_hi"].max(), 0) * 1.15 + 6)
    ax.axhline(0, c="k", lw=1)
    ax.axhline(s["base"].iloc[0], c="k", lw=1, ls="--", label=f"All-months average ({s['base'].iloc[0]:+.1f})")
    for k, (st, row) in enumerate(s.iterrows()):
        ax.text(k, row["raw_hi"] + 0.5, f"t={row['raw_t']:.1f}\n{int(row['episodes'])} ep · hit {row['hit']:.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[c].replace(" / ", "\n") for c in s.index], fontsize=9)
    ax.set_ylabel("Defensives minus Growth, annualized next-month return (pp)")
    sub = f"red = current regime ({DISPLAY.get(current, current)})"
    if expected is not None:
        sub += f" · probability-weighted next 3 months: {expected:+.1f} pp/yr"
    ax.set_title("Thesis test: (XLP+XLU+XLV)/3 minus (XLK+XLY)/2, by regime, 95% CI clustered by episode\n" + sub,
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_history(g, i, labels, path, start="1997-01-01"):
    d = pd.concat([g, i, labels], axis=1, keys=["g", "i", "r"]).loc[start:].dropna(subset=["g", "i"])
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 0.6]})
    usrec = fred_optional("USREC")
    if usrec is not None:
        rec = monthly(usrec).reindex(d.index).fillna(0)
        ax[0].fill_between(d.index, -3, 3, where=rec > 0, color="grey", alpha=0.25, label="NBER recession")
    ax[0].plot(d.index, d["g"], lw=1.4, label="Growth z")
    ax[0].plot(d.index, d["i"], lw=1.4, label="Inflation z")
    ax[0].axhline(0, c="k", lw=0.8); ax[0].set_ylim(-3.1, 3.1); ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title("Regime composites and labels (validation: growth should dip in every recession)")
    for t, r in d["r"].items():
        if isinstance(r, str):
            ax[1].axvspan(t - pd.offsets.MonthEnd(1), t, color=COLORS[r], lw=0)
    ax[1].set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ORDER]
    ax[1].legend(handles, [DISPLAY[k] for k in ORDER], fontsize=7, ncol=5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.4))
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def _clean_stale():
    for f in ["regime_hit_rates.csv", "regime_excess_returns.csv", "markov_prob_weak_growth.csv"]:
        (OUT / f).unlink(missing_ok=True)


def main():
    _clean_stale()
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    R = build_regime()
    g, i, lab = R["g"], R["i"], R["regime"]
    end = lab.dropna().index[-1]
    print("=" * 90, "\nMODULE 1 - MACRO REGIME (v2.1)")
    print(f"  growth survey slot   : {R['sources'][0]}")
    print(f"  inflation survey slot: {R['sources'][1]}")
    print(f"  interpolated months  : {FILL_LOG or 'none'}")
    print(f"  ragged-edge carries  : {EDGE_LOG or 'none'}")
    print(f"  latest row {end:%b %Y} is PROVISIONAL (month in progress)" if end >= current_month_end() else "")

    # ---- state
    direction = classify_direction(g, i)
    ng, ni, ng_cols, ni_cols = build_nowcast()
    state = pd.concat([g, i, lab, lab.map(DISPLAY).rename("regime_label"),
                       g.diff(3).rename("growth_mom"), i.diff(3).rename("inflation_mom"),
                       direction.rename("direction"), ng, ni], axis=1).loc["1997":]
    state.to_csv(OUT / "regime_state.csv")
    print("\nCurrent state (last 8 months):\n", state.drop(columns="regime_label").tail(8).round(2))

    # ---- label diagnostics (no returns involved -> safe to look at before the playbook)
    lab_s = lab.loc[SAMPLE_START:]
    print(f"\nTransition share since 1999: {(lab_s == 'Transition').mean():.0%}  (target 15-30%)")
    print("Band sensitivity (Transition share since 1999, labels only):")
    for b in [0.10, 0.15, 0.20, 0.25]:
        print(f"  band {b:.2f}: {(classify(g, i, b, STICKY).loc[SAMPLE_START:] == 'Transition').mean():.0%}")
    ii = i.dropna()
    print("Share of months with inflation_z > 0, by decade:\n",
          (ii > 0).groupby(ii.index.year // 10 * 10).mean().round(2).to_string())

    # ---- playbook
    ex = returns_panel()
    lab_r = lab.reindex(ex.index)
    stats = conditional_stats(lab_r, ex, family={SPREAD: "thesis"})
    eps = episode_table(lab_r.iloc[:-1])
    stats.to_csv(OUT / "regime_stats_long.csv", index=False)
    eps.to_csv(OUT / "regime_episodes.csv")
    print("\nEpisodes per regime (return sample):\n", eps)
    sec = stats[stats.family == "sector"]
    for name in ["edge", "t", "q"]:
        print(f"\n{name} (sector x regime):\n",
              sec.pivot(index="series", columns="state", values=name)
                 .reindex(columns=[c for c in ORDER if c in set(sec.state)]).round(2 if name == "q" else 1))
    cred = sec[sec.credible]
    print("\nCredible cells (q<=0.10, sign-stable, >=5 episodes):",
          "none" if cred.empty else "\n" + cred[["series", "state", "edge", "t", "q", "episodes"]].round(2).to_string())

    # ---- thesis test
    th = stats[stats.series == SPREAD].set_index("state")
    print("\nTHESIS TEST - Defensives minus Growth by regime (annualized pp):\n",
          th[["months", "episodes", "raw", "raw_lo", "raw_hi", "raw_t", "edge", "t", "hit",
              "loeo_min", "loeo_max", "sign_stable"]].round(2))

    dir_r = direction.reindex(ex.index)
    dstats = conditional_stats(dir_r, ex[[SPREAD]], order=DIR_ORDER, family={SPREAD: "thesis"})
    dstats.to_csv(OUT / "direction_stats_long.csv", index=False)
    print("\nDirection lens (3m change in composites) - DEF-GRW:\n",
          dstats.set_index("state")[["months", "episodes", "raw", "raw_t", "edge", "t", "hit"]].round(2))

    # ---- forward-looking: transition matrix -> probability-weighted expectation
    P = transition_matrix(lab.loc["1997":])
    P.round(3).to_csv(OUT / "regime_transition_matrix.csv")
    current = lab.dropna().iloc[-1]
    fwd = forward_distribution(P, current, horizon=3)
    fwd.round(3).to_csv(OUT / "regime_forward_3m.csv")
    mu = th["raw"].reindex(P.index)
    expected = float((fwd @ mu.fillna(th["base"].iloc[0])).mean())
    print("\nOne-month transition matrix:\n", P.round(2))
    print(f"\nRegime distribution for the next 3 return months (starting from {current}):\n", fwd.round(2))

    # ---- robustness side-checks
    try:
        p = markov_probs(R["g_raw"])
        p.to_csv(OUT / "markov_prob_weak_growth.csv")
        print(f"\nMarkov P(weak-growth state), latest: {p.iloc[-1]:.2f}  (robustness check only)")
    except Exception as e:
        print(f"\nMarkov model skipped: {e}")

    both = pd.concat([ng, ni, R["g_raw"], R["i_raw"]], axis=1, keys=["ng", "ni", "g", "i"], sort=True).loc["2003":]
    print(f"\nNowcast inputs: growth {ng_cols}; inflation {ni_cols}")
    print("Nowcast validation - correlation of nowcast(t) with the lagged composite at t+k:")
    for k in range(4):
        print(f"  k={k}: growth {both['ng'].corr(both['g'].shift(-k)):.2f}   "
              f"inflation {both['ni'].corr(both['i'].shift(-k)):.2f}")

    # ---- plots
    ng_last = ng.dropna()
    nc = (ng_last.iloc[-1], ni.reindex(ng_last.index).iloc[-1], ng_last.index[-1]) if len(ng_last) else None
    plot_quadrant(g, i, lab, OUT / "regime_quadrant.png", nowcast=nc)
    plot_edge_heatmap(stats, eps, OUT / "regime_edge_heatmap.png")
    plot_thesis(stats, current, OUT / "regime_thesis_spread.png", expected)
    plot_history(g, i, lab, OUT / "regime_history.png")

    # ---- evidence summary
    row = th.loc[current] if current in th.index else None
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Regime, lagged composite ({end:%b %Y}): {DISPLAY[current]}  "
          f"[growth {g.loc[end]:+.2f}, inflation {i.loc[end]:+.2f}]")
    print(f"  Direction of travel (3m): {direction.loc[end]}")
    if nc:
        print(f"  Nowcast ({nc[2]:%b %Y}): growth {nc[0]:+.2f}, inflation {nc[1]:+.2f}")
    print("  Next-3-month regime odds: " + ", ".join(f"{DISPLAY[k]} {v:.0%}" for k, v in fwd.mean().items() if v >= 0.05))
    if row is not None:
        print(f"  DEF-GRW in current regime: {row['raw']:+.1f} pp/yr (95% CI {row['raw_lo']:+.1f} to "
              f"{row['raw_hi']:+.1f}), t={row['raw_t']:.2f}, {int(row['episodes'])} episodes, hit {row['hit']:.0f}%")
    print(f"  Probability-weighted DEF-GRW, next 3 months: {expected:+.1f} pp/yr")
    print(f"  Verdict (pre-registered, one-sided 5%): {thesis_verdict(row)}")
    if optional_csv("ism_new_orders") is not None and optional_csv("ism_prices") is not None:
        F = build_regime(survey="fred")
        f_cur = F["regime"].dropna().iloc[-1]
        f_st = conditional_stats(F["regime"].reindex(ex.index), ex[[SPREAD]], loeo=False).set_index("state")
        f_row = f_st.loc[f_cur] if f_cur in f_st.index else None
        same = (F["regime"].loc[SAMPLE_START:] == lab.loc[SAMPLE_START:])[lab.loc[SAMPLE_START:].notna()].mean()
        print(f"  Input check - same model with free FRED surveys instead of ISM: {DISPLAY[f_cur]}, "
              f"verdict {thesis_verdict(f_row)}; labels agree {same:.0%} of months since 1999")
    print("=" * 90)


def robustness():
    """Does the current read and the thesis evidence survive reasonable alternative specs?"""
    ex = returns_panel()
    base = build_regime()["regime"]
    specs = [("baseline", {}), ("window 180m", {"window": 180}), ("expanding window", {"window": None}),
             ("no smoothing", {"smooth": 1}), ("band 0.10", {"band": 0.10}), ("band 0.25", {"band": 0.25}),
             ("no hysteresis", {"sticky": False}), ("inflation vs rolling median", {"infl_center": "rolling"}),
             ("+ 5y5y breakevens (2008+)", {"with_be": True}), ("FRED surveys (Philly)", {"survey": "fred"})]
    if optional_csv("ism_new_orders") is not None and optional_csv("ism_prices") is not None:
        specs.append(("ISM surveys (Bloomberg)", {"survey": "ism"}))
    rows = []
    for name, kw in specs:
        print(f"  running: {name}")
        R = build_regime(**kw)
        lab = R["regime"]
        cmp = pd.concat([base, lab], axis=1, keys=["b", "r"]).loc[SAMPLE_START:].dropna()
        cur = lab.dropna().iloc[-1]
        st = conditional_stats(lab.reindex(ex.index), ex[[SPREAD]], loeo=False).set_index("state")
        r = st.loc[cur] if cur in st.index else None
        rows.append({"spec": name, "agree_%": round((cmp.b == cmp.r).mean() * 100),
                     "transition_%": round((lab.loc[SAMPLE_START:] == "Transition").mean() * 100),
                     "episodes": int((lab.loc[SAMPLE_START:].dropna() != lab.loc[SAMPLE_START:].dropna().shift()).sum()),
                     "latest": cur, "g": round(R["g"].dropna().iloc[-1], 2), "i": round(R["i"].dropna().iloc[-1], 2),
                     "DEF-GRW raw": None if r is None else round(r["raw"], 1),
                     "t": None if r is None else round(r["raw_t"], 2),
                     "verdict": thesis_verdict(r)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "regime_robustness.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()