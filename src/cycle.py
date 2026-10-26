"""Module 2 - Business cycle / leading indicators (v2).
 
Run:
    python -m src.cycle            # main run: momentum, phase, thesis tests, recession risk, plots
    python -m src.cycle --robust   # specification robustness table
 
ROLE (vs Module 1)
  Module 1 = where the economy IS (coincident data: payrolls, IP, retail sales, claims, ISM new orders).
  Module 2 = where it is HEADING over the next 2-4 quarters, using forward-looking inputs only.
  No input is shared with Module 1, so the two modules are separate evidence in the scorecard.
 
PRE-REGISTERED HEADLINE TEST (decided before looking at any return result)
  "Slope test": regress the next-3-month Defensives-minus-Growth return on the leading-momentum
  level and its 3-month change (HAC errors). Fitted value at today's readings, one-sided 5%:
  t >= 1.65 SUPPORTS the defensive thesis, t <= -1.65 CONTRADICTS, otherwise NEUTRAL.
  The phase tables are illustration and robustness, not the headline.
 
INPUTS (all oriented so HIGHER = BETTER for growth)
  Momentum block (6-month changes; z-scored, equal-weighted, all required):
    ism_nmi   ISM new orders minus inventories (Bloomberg, lag 1). Fallback: Philly Fed new orders.
    permits   6m % change in building permits (FRED PERMIT, lag 1)
    hours     6m change in average weekly hours, manufacturing (FRED AWHMAN, lag 1)
    sentiment 6m change in UMich consumer sentiment (FRED UMCSENT, lag 0)
    credit    minus the 6m change in the HY spread (Bloomberg LF98OAS; fallback Baa-Treasury)
  Risk block (slow levels, used only in the recession probit):
    curve     10y minus 3m Treasury (GS10 - TB3MS)
    baa       Baa corporate minus 10y Treasury (BAA - GS10), history back to 1953
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
 
from src.data import (fred, fred_optional, level, monthly, optional_csv, sector_returns,
                      complete_months, current_month_end, FILL_LOG)
from src.utils import (chg, lagged, zscore, run_ids, episode_table, forward_excess, conditional_stats,
                       transition_matrix, forward_distribution, _mad)
from src.regime import (finalize, EDGE_LOG, thesis_verdict, plot_edge_heatmap, build_regime,
                        SECTORS, ROW_NAME, DEFENSIVE, GROWTH, SAMPLE_START)
 
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
 
# ------------------------------------------------------------------ specification (frozen)
WINDOW = 120        # 10-year rolling baseline for z-scores
SMOOTH = 3          # trailing smoothing of the composite
CHG = 6             # months over which momentum inputs are measured
BAND_LEVEL = 0.15   # neutral zone on the composite level (z units)
BAND_MOM = 0.25     # neutral zone on momentum (in units of momentum's own robust dispersion)
STICKY = True       # hysteresis inside the bands
CREDIT = "auto"     # "auto" = Bloomberg HY OAS if present, else Baa-Treasury; or "hy" / "baa"
M_COLS = ["ism_nmi", "permits", "hours", "sentiment", "credit"]
PROBIT_START = "1970-01-31"   # first month with a real-time probit forecast
NBER_LAG = 12                 # months before NBER dates are known (training cut-off)
 
PHASES = ["Early", "Mid", "Late", "Contraction", "Transition"]
PDISPLAY = {"Early": "Recovery (below avg, improving)", "Mid": "Expansion (above avg, improving)",
            "Late": "Late cycle (above avg, fading)", "Contraction": "Contraction (below avg, fading)",
            "Transition": "Transition (no clear phase)"}
PCOLORS = {"Early": "#1565c0", "Mid": "#2e7d32", "Late": "#f9a825", "Contraction": "#c62828",
           "Transition": "#9e9e9e"}
 
CYCLICAL = ["XLI", "XLB", "XLF", "XLE", "XLY"]
SPREADS = {"DEF-GRW": (DEFENSIVE, GROWTH), "CYC-DEF": (CYCLICAL, DEFENSIVE)}
HEADLINE = "DEF-GRW"
 
 
# ------------------------------------------------------------------ inputs
def credit_series(source=CREDIT):
    """Returns (HY spread or None, Baa-Treasury spread, description).
    'auto': HY where available (1994+), Baa before it. Each is z-scored on its own history and the
    two are spliced in z-space, so the long Baa history extends the sample back to the 1960s."""
    baa = level("BAA") - level("GS10")
    hy = optional_csv("hy_oas") if source in ("auto", "hy") else None
    if hy is not None:
        return monthly(hy, "mean"), baa, "Bloomberg HY OAS (LF98OAS) from 1999, Baa-Treasury before (spliced in z-space)"
    if source == "hy":
        raise FileNotFoundError("data/raw/hy_oas.csv not found")
    return None, baa, "Baa corporate minus 10y Treasury (FRED)"
 
 
def orders_series():
    no, inv = optional_csv("ism_new_orders"), optional_csv("ism_inventories")
    if no is not None and inv is not None:
        return lagged(monthly(no) - monthly(inv), 1), "ISM new orders minus inventories (Bloomberg, lag 1)"
    return level("NOCDFSA066MSFRBPHI"), "Philly Fed new orders (FRED fallback, lag 0)"
 
 
def build_inputs(chg_n=CHG, credit=CREDIT):
    orders, osrc = orders_series()
    hy, baa, csrc = credit_series(credit)
    M = pd.concat({k: v for k, v in {
        "ism_nmi":   orders,
        "permits":   lagged(chg(level("PERMIT"), chg_n), 1),
        "hours":     lagged(level("AWHMAN").diff(chg_n), 1),
        "sentiment": level("UMCSENT").diff(chg_n),
        "credit_hy":  -hy.diff(chg_n) if hy is not None else None,
        "credit_baa": -baa.diff(chg_n),
    }.items() if v is not None}, axis=1, sort=True)
    return finalize(M, "cycle"), (osrc, csrc)
 
 
def robust_unit(s, min_periods=60):
    """Divide by real-time (expanding) robust dispersion so bands are scale-free."""
    sc = 1.4826 * s.expanding(min_periods).apply(_mad, raw=True)
    return s / sc.where(sc > 0)
 
 
def quadrant_phase(lvl, mom):
    if mom > 0:
        return "Mid" if lvl > 0 else "Early"
    return "Late" if lvl > 0 else "Contraction"
 
 
def classify_phase(lvl, mom, band_l=BAND_LEVEL, band_m=BAND_MOM, sticky=STICKY):
    d = pd.concat([lvl, mom], axis=1, keys=["l", "m"], sort=True)
    out, last = [], None
    for l, m in zip(d["l"].values, d["m"].values):
        if np.isnan(l) or np.isnan(m):
            out.append(np.nan); last = None; continue
        q = quadrant_phase(l, m)
        if abs(l) >= band_l and abs(m) >= band_m:
            out.append(q); last = q
        elif sticky and last == q:
            out.append(q)
        else:
            out.append("Transition")
    return pd.Series(out, index=d.index, dtype=object)
 
 
def build_cycle(window=WINDOW, smooth=SMOOTH, chg_n=CHG, band_l=BAND_LEVEL, band_m=BAND_MOM,
                sticky=STICKY, credit=CREDIT, cols=None):
    M, src = build_inputs(chg_n, credit)
    cols = cols or M_COLS
    ZA = zscore(M, window)
    ZA["credit"] = ZA["credit_hy"].combine_first(ZA["credit_baa"]) if "credit_hy" in ZA else ZA["credit_baa"]
    Z = ZA[cols]
    raw = Z.mean(axis=1).where(Z.notna().all(axis=1))
    lvl = raw.rolling(smooth).mean().rename("momentum_z")
    mom = robust_unit(lvl.diff(3)).rename("momentum_chg")
    dz = Z.diff(3)
    diffusion = ((dz > 0).sum(axis=1) / len(cols) * 100).where(dz.notna().all(axis=1)).rename("diffusion")
    phase = classify_phase(lvl, mom, band_l, band_m, sticky).rename("phase")
    return dict(M=M, Z=Z, raw=raw, lvl=lvl, mom=mom, diffusion=diffusion, phase=phase, sources=src)
 
 
# ------------------------------------------------------------------ recession probit (real time)
def risk_block():
    curve = (level("GS10") - level("TB3MS")).rename("curve")
    baa = (level("BAA") - level("GS10")).rename("baa_spread")
    return pd.concat([curve, baa], axis=1, sort=True)
 
 
def recession_target(h=12):
    usrec = monthly(fred("USREC"))
    return pd.concat([usrec.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1, skipna=False).rename("y")
 
 
def _probit(y, X):
    try:
        return sm.Probit(y, X).fit(disp=False, cov_type="HAC", cov_kwds={"maxlags": 12})
    except Exception:
        return sm.Probit(y, X).fit(disp=False)
 
 
def recession_probit():
    """Probit of 'any NBER recession month in t+1..t+12' on curve and Baa spread.
    Real-time version: refit every January using only targets that were knowable at the time
    (target month t+12 plus a 12-month NBER announcement lag => train on t <= T - 24)."""
    from sklearn.metrics import roc_auc_score
    R = risk_block()
    y = recession_target()
    d = pd.concat([y, R], axis=1, sort=True).dropna(subset=["curve", "baa_spread"])
    X_all = np.column_stack([np.ones(len(d)), d["curve"].values, d["baa_spread"].values])
 
    train = d["y"].notna()
    full = _probit(d.loc[train, "y"].values, X_all[train.values])
    coefs = pd.DataFrame({"coef": np.asarray(full.params), "t": np.asarray(full.tvalues)},
                         index=["const", "curve", "baa_spread"])
 
    rt = pd.Series(np.nan, index=d.index, name="p_recession_12m")
    for year in sorted(set(d.index[d.index >= PROBIT_START].year)):
        T = pd.Timestamp(f"{year}-01-31")
        cut = T - pd.offsets.MonthEnd(12 + NBER_LAG)
        tr = (d.index <= cut) & d["y"].notna().values
        if tr.sum() < 120 or d.loc[tr, "y"].sum() < 12:
            continue
        res = _probit(d.loc[tr, "y"].values, X_all[tr])
        pred = (d.index.year == year)
        rt[pred] = res.predict(X_all[pred])
    ev = pd.concat([rt, d["y"]], axis=1).dropna()
    auc_rt = roc_auc_score(ev["y"], ev["p_recession_12m"]) if ev["y"].nunique() == 2 else np.nan
    auc_in = roc_auc_score(d.loc[train, "y"], full.predict(X_all[train.values]))
    return rt, coefs, auc_rt, auc_in, len(ev)
 
 
# ------------------------------------------------------------------ returns and tests
def returns_panels():
    r = complete_months(sector_returns())
    missing = [t for t in ["SPY"] + SECTORS if t not in r.columns]
    if missing:
        raise KeyError(f"prices_monthly.csv lacks {missing}; delete it and rerun to re-download")
    ex1 = forward_excess(r[["SPY"] + SECTORS].loc[SAMPLE_START:], spreads=SPREADS)
    ex3 = ex1.rolling(3).mean().shift(-2)      # row t = average of months t+1..t+3
    return ex1, ex3
 
 
def slope_test(lvl, mom, ex3, series=HEADLINE):
    """Regress next-3-month spread (annualized pp) on DEMEANED momentum level and change; HAC(6) errors.
    With demeaned regressors the constant is the sample-average spread (the secular drift), so:
      fit  = expected spread at today's readings          -> what the trade earns (headline verdict)
      tilt = fit - average = the cycle's contribution     -> does the cycle point toward defensives?"""
    d = pd.concat([ex3[series] * 1200, lvl, mom], axis=1, keys=["y", "lvl", "mom"], sort=True).dropna()
    mu = d[["lvl", "mom"]].mean()
    X = np.column_stack([np.ones(len(d)), d["lvl"].values - mu["lvl"], d["mom"].values - mu["mom"]])
    res = sm.OLS(d["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    now = pd.concat([lvl, mom], axis=1, sort=True).dropna().iloc[-1]
    x0 = np.array([1.0, now.iloc[0] - mu["lvl"], now.iloc[1] - mu["mom"]])
    b = np.asarray(res.params)
    V = np.asarray(res.cov_params()) if hasattr(res, "cov_params") else np.diag(np.asarray(res.bse) ** 2)
    fit, se = float(x0 @ b), float(np.sqrt(x0 @ V @ x0))
    x1 = x0.copy(); x1[0] = 0.0
    tilt, tse = float(x1 @ b), float(np.sqrt(x1 @ V @ x1))
    coefs = pd.DataFrame({"coef": b, "t": np.asarray(res.tvalues)}, index=["average", "level", "change"])
    return dict(coefs=coefs, fit=fit, se=se, t=fit / se, lo=fit - 1.96 * se, hi=fit + 1.96 * se,
                tilt=tilt, tilt_t=tilt / tse if tse > 0 else np.nan, avg=float(b[0]),
                n=len(d), start=d.index[0], end=d.index[-1], asof=now.name, series=series)
 
 
def slope_verdict(st):
    return thesis_verdict({"raw_t": st["t"]})
 
 
def lead_profile(x, target, max_lead=12):
    d = pd.concat([x, target], axis=1, keys=["x", "t"], sort=True)
    return pd.Series({k: d["x"].corr(d["t"].shift(-k)) for k in range(0, max_lead + 1)}, name="corr")
 
 
# ------------------------------------------------------------------ plots
def plot_dashboard(C, prob, path, start="1997-01-01"):
    lvl, diff = C["lvl"].loc[start:], C["diffusion"].loc[start:]
    p = prob.loc[start:]
    x0, x1 = min(lvl.dropna().index.min(), p.dropna().index.min()), max(lvl.index.max(), p.index.max())
    rec = monthly(fred("USREC")).loc[x0:x1]          # shading limited to the plotted range (v1 bug)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})
    ax[0].plot(lvl.index, lvl, lw=1.4, label="Leading momentum composite (z, 3m avg)")
    ax[0].axhline(0, c="k", lw=0.8); ax[0].axhspan(-BAND_LEVEL, BAND_LEVEL, color="grey", alpha=0.15)
    ax[1].plot(diff.index, diff, lw=1, color="#6a1b9a", label="Diffusion: % of inputs improving (3m)")
    ax[1].axhline(50, c="k", lw=0.8, ls="--"); ax[1].set_ylim(-5, 105)
    ax[2].plot(p.index, p * 100, lw=1.4, color="#c62828", label="P(recession within 12m), real-time %")
    ax[2].set_ylim(0, 100)
    for a in ax:
        lo, hi = a.get_ylim()
        a.fill_between(rec.index, lo, hi, where=rec > 0, color="grey", alpha=0.25, step="pre")
        a.set_ylim(lo, hi); a.legend(fontsize=8, loc="upper left")
    ax[0].set_title("Module 2: leading momentum, breadth and recession risk (grey bars = NBER recessions)")
    ax[2].set_xlim(x0, x1)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
 
 
def plot_clock(lvl, mom, labels, path, months=60):
    tr = pd.concat([mom, lvl], axis=1, keys=["x", "y"], sort=True).dropna().iloc[-months:]
    share = labels.reindex(tr.index).value_counts(normalize=True).mul(100)
    L = np.ceil(max(tr.abs().max().max() * 1.15, 1.0) * 4) / 4
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    quads = {"Early": (0.5, 0.0), "Mid": (0.5, 0.5), "Late": (0.0, 0.5), "Contraction": (0.0, 0.0)}
    pos = {"Mid": (0.98, 0.98, "right", "top"), "Late": (0.02, 0.98, "left", "top"),
           "Contraction": (0.02, 0.02, "left", "bottom"), "Early": (0.98, 0.02, "right", "bottom")}
    for k, (x0, y0) in quads.items():
        ax.add_patch(plt.Rectangle((x0, y0), 0.5, 0.5, transform=ax.transAxes, color=PCOLORS[k], alpha=0.08, zorder=0))
        x, y, ha, va = pos[k]
        ax.text(x, y, f"{PDISPLAY[k]}\n{share.get(k, 0):.0f}% of months", transform=ax.transAxes, ha=ha, va=va,
                fontsize=10.5, fontweight="bold", color=PCOLORS[k], alpha=0.85)
    ax.axhline(0, c="k", lw=1); ax.axvline(0, c="k", lw=1)
    ax.axhspan(-BAND_LEVEL, BAND_LEVEL, color="grey", alpha=0.12); ax.axvspan(-BAND_MOM, BAND_MOM, color="grey", alpha=0.12)
    ax.plot(tr["x"], tr["y"], "-", color="grey", lw=0.8, alpha=0.6, zorder=2)
    sc = ax.scatter(tr["x"], tr["y"], c=np.arange(len(tr)), cmap="viridis", s=38, edgecolor="white", lw=0.5, zorder=3)
    ax.scatter(tr["x"].iloc[-1], tr["y"].iloc[-1], s=260, facecolors="none", edgecolors="red", lw=2.5, zorder=4)
    for k in sorted(set([0, len(tr) - 1] + list(range(0, len(tr), 6)))):
        ax.annotate(tr.index[k].strftime("%b-%y"), (tr["x"].iloc[k], tr["y"].iloc[k]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    t = np.linspace(0, len(tr) - 1, 6, dtype=int)
    cb.set_ticks(t); cb.set_ticklabels([tr.index[k].strftime("%b %Y") for k in t])
    ax.set_xlabel("Momentum: 3-month change of the composite (robust units)  →  improving")
    ax.set_ylabel("Level: leading composite vs 10-year baseline (z, 3m avg)  →  above average")
    ax.set_title(f"Cycle clock, {tr.index[0]:%b %Y} to {tr.index[-1]:%b %Y} (monthly; red ring = latest)\n"
                 f"Counter-clockwise: Recovery → Expansion → Late → Contraction; "
                 f"{share.get('Transition', 0):.0f}% of months in Transition", fontsize=10.5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
 
 
def plot_thesis(stats3, current, slope, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, name, ttl in [(axes[0], "DEF-GRW", "Defensives (XLP,XLU,XLV) minus Growth (XLK,XLY)"),
                          (axes[1], "CYC-DEF", "Cyclicals (XLI,XLB,XLF,XLE,XLY) minus Defensives")]:
        s = stats3[stats3.series == name].set_index("state").reindex(PHASES).dropna(subset=["raw"])
        x = np.arange(len(s))
        ax.bar(x, s["raw"], color=["#d32f2f" if k == current else "#90a4ae" for k in s.index], alpha=0.85)
        ax.errorbar(x, s["raw"], yerr=[s["raw"] - s["raw_lo"], s["raw_hi"] - s["raw"]], fmt="none",
                    ecolor="k", capsize=5, lw=1)
        ax.axhline(0, c="k", lw=1)
        ax.axhline(s["base"].iloc[0], c="k", ls="--", lw=1, label=f"All-months average ({s['base'].iloc[0]:+.1f})")
        for k, (_, r) in enumerate(s.iterrows()):
            ax.text(k, max(r["raw_hi"], 0) + 0.8, f"t={r['raw_t']:.1f}\n{int(r['episodes'])} ep",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([PDISPLAY[k].split(" (")[0] for k in s.index], fontsize=9)
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Average return over the next 3 months, annualized pp")
    lo = min(stats3["raw_lo"].min(), 0) * 1.15 - 1
    hi = max(stats3[stats3.series.isin(SPREADS)]["raw_hi"].max(), 0) * 1.15 + 6
    axes[0].set_ylim(lo, hi)
    fig.suptitle(f"Thesis tests by cycle phase (red = current: {PDISPLAY.get(current, current)}); 95% CI clustered "
                 f"by phase episode\nHeadline slope test: next-3-month DEF-GRW at today's readings "
                 f"{slope['fit']:+.1f} pp/yr (95% CI {slope['lo']:+.1f} to {slope['hi']:+.1f}, t={slope['t']:.2f}) "
                 f"= average {slope['avg']:+.1f} + cycle tilt {slope['tilt']:+.1f} (t={slope['tilt_t']:.2f})",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
 
 
def plot_lead(prof_m1, prof_ip, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(prof_m1.index, prof_m1.values, "-o", label="vs Module 1 growth composite k months later")
    ax.plot(prof_ip.index, prof_ip.values, "-s", label="vs 6m industrial production growth k months later")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("k (months ahead)"); ax.set_ylabel("Correlation")
    ax.set_title("Lead validation: does the composite move BEFORE the economy?\n"
                 "(peak to the right of 0 = it leads)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
 
 
# ------------------------------------------------------------------ runs
def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    C = build_cycle()
    lvl, mom, phase = C["lvl"], C["mom"], C["phase"]
    end = phase.dropna().index[-1]
    print("=" * 90, "\nMODULE 2 - BUSINESS CYCLE / LEADING INDICATORS (v2)")
    print(f"  orders input : {C['sources'][0]}")
    print(f"  credit input : {C['sources'][1]}")
    print(f"  interpolated months : {FILL_LOG or 'none'}")
    print(f"  ragged-edge carries : {{{', '.join(f'{k}: {v}' for k, v in EDGE_LOG.items() if k.startswith('cycle'))}}}")
    print(f"  composite available from {lvl.dropna().index[0]:%b %Y}; latest row {end:%b %Y}"
          + (" is PROVISIONAL (month in progress)" if end >= current_month_end() else ""))
 
    # ---- input diagnostics
    Zs = C["Z"].loc[SAMPLE_START:]
    print("\nInput correlations since 1999 (should be positive, not ~1):\n", Zs.corr().round(2))
    m1 = build_regime()
    print(f"Independence from Module 1: corr(momentum, Module 1 growth, same month) = "
          f"{C['raw'].corr(m1['g_raw']):.2f}")
 
    # ---- state
    prob, pcoefs, auc_rt, auc_in, n_ev = recession_probit()
    R = risk_block()
    state = pd.concat([lvl, mom, C["diffusion"], phase, phase.map(PDISPLAY).rename("phase_label"),
                       prob, R], axis=1, sort=True).loc["1997":]
    state.to_csv(OUT / "cycle_state.csv")
    print("\nCurrent state (last 8 months):\n", state.drop(columns="phase_label").tail(8).round(2))
 
    # ---- label diagnostics (before any returns)
    ph = phase.loc[SAMPLE_START:]
    print(f"\nTransition share since 1999: {(ph == 'Transition').mean():.0%}  (target 15-30%)")
    print("Momentum-band sensitivity (Transition share since 1999, labels only):")
    for b in [0.15, 0.25, 0.35]:
        print(f"  band_mom {b:.2f}: {(classify_phase(lvl, mom, BAND_LEVEL, b).loc[SAMPLE_START:] == 'Transition').mean():.0%}")
 
    # ---- lead validation
    ip6 = (chg(level("INDPRO"), 6) * 2).rename("ip6")
    prof_m1, prof_ip = lead_profile(C["raw"], m1["g_raw"]), lead_profile(C["raw"], ip6)
    pd.concat([prof_m1.rename("vs_module1_growth"), prof_ip.rename("vs_ip_6m")], axis=1).round(3).to_csv(OUT / "cycle_lead_profile.csv")
    print(f"\nLead validation: peak corr with Module 1 growth at k={prof_m1.idxmax()} months ({prof_m1.max():.2f}); "
          f"with 6m IP growth at k={prof_ip.idxmax()} months ({prof_ip.max():.2f})")
 
    # ---- recession probit
    print("\nRecession probit (full-sample coefficients, HAC t):\n", pcoefs.round(3))
    print(f"AUC in-sample {auc_in:.2f}; REAL-TIME out-of-sample {auc_rt:.2f} over {n_ev} months")
    print(f"Latest real-time P(recession within 12m): {prob.dropna().iloc[-1]:.0%} ({prob.dropna().index[-1]:%b %Y})")
 
    # ---- thesis tests
    ex1, ex3 = returns_panels()
    fam = {k: "thesis" for k in SPREADS}
    lab1 = phase.reindex(ex1.index)
    stats1 = conditional_stats(lab1, ex1, order=PHASES, family=fam)
    stats3 = conditional_stats(lab1, ex3, order=PHASES, family=fam)
    eps = episode_table(lab1.iloc[:-1], order=PHASES)
    stats1.to_csv(OUT / "cycle_stats_1m_long.csv", index=False)
    stats3.to_csv(OUT / "cycle_stats_3m_long.csv", index=False)
    eps.to_csv(OUT / "cycle_episodes.csv")
    print("\nEpisodes per phase (return sample):\n", eps)
    cols = ["months", "episodes", "raw", "raw_lo", "raw_hi", "raw_t", "edge", "t", "hit", "sign_stable"]
    for name in SPREADS:
        print(f"\n{name} by phase, NEXT 3 MONTHS (annualized pp):\n",
              stats3[stats3.series == name].set_index("state")[cols].round(2))
    sec = stats3[stats3.family == "sector"]
    cred = sec[sec.credible]
    print("\nCredible sector cells, next 3 months (q<=0.10, sign-stable, >=5 episodes):",
          "none" if cred.empty else "\n" + cred[["series", "state", "edge", "t", "q", "episodes"]].round(2).to_string())
 
    slopes = {k: slope_test(lvl, mom, ex3, k) for k in SPREADS}
    pd.concat({k: v["coefs"] for k, v in slopes.items()}).round(3).to_csv(OUT / "cycle_slope_test.csv")
    for k, s in slopes.items():
        print(f"\nSlope test {k} (next-3-month return on momentum level and change, HAC(6)), "
              f"{s['start']:%Y-%m} to {s['end']:%Y-%m}, n={s['n']}:\n", s["coefs"].round(3))
        print(f"  fitted at {s['asof']:%b %Y} readings: {s['fit']:+.1f} pp/yr "
              f"(95% CI {s['lo']:+.1f} to {s['hi']:+.1f}), t={s['t']:.2f}  =  average {s['avg']:+.1f} "
              f"+ cycle tilt {s['tilt']:+.1f} (t={s['tilt_t']:.2f})")
 
    # ---- forward phase odds
    P = transition_matrix(phase.loc["1997":], order=PHASES)
    P.round(3).to_csv(OUT / "cycle_transition_matrix.csv")
    current = phase.dropna().iloc[-1]
    fwd = forward_distribution(P, current, horizon=3)
    fwd.round(3).to_csv(OUT / "cycle_forward_3m.csv")
    th1 = stats1[stats1.series == HEADLINE].set_index("state")
    expected = float((fwd @ th1["raw"].reindex(P.index).fillna(th1["base"].iloc[0])).mean())
    print("\nOne-month phase transition matrix:\n", P.round(2))
 
    # ---- plots
    plot_dashboard(C, prob, OUT / "cycle_dashboard.png")
    plot_clock(lvl, mom, phase, OUT / "cycle_clock.png")
    plot_thesis(stats3, current, slopes[HEADLINE], OUT / "cycle_thesis_spread.png")
    plot_edge_heatmap(stats3, eps, OUT / "cycle_edge_heatmap.png", order=PHASES, display=PDISPLAY,
                      what="cycle phase", horizon="Next-3-month")
    plot_lead(prof_m1, prof_ip, OUT / "cycle_lead.png")
    for f in ["cycle_excess_returns.csv", "cycle_hit_rates.csv"]:
        (OUT / f).unlink(missing_ok=True)
 
    # ---- evidence summary
    s = slopes[HEADLINE]
    r3 = stats3[(stats3.series == HEADLINE) & (stats3.state == current)]
    r3 = r3.iloc[0] if len(r3) else None
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Phase ({end:%b %Y}): {PDISPLAY[current]}  [level {lvl.loc[end]:+.2f}, momentum {mom.loc[end]:+.2f}, "
          f"diffusion {C['diffusion'].loc[end]:.0f}%]")
    print(f"  Leads Module 1 growth by ~{prof_m1.idxmax()} months; real-time recession AUC {auc_rt:.2f}")
    print(f"  P(recession within 12m), real-time: {prob.dropna().iloc[-1]:.0%}")
    print("  Next-3-month phase odds: " + ", ".join(f"{PDISPLAY[k].split(' (')[0]} {v:.0%}"
                                                    for k, v in fwd.mean().items() if v >= 0.05))
    if r3 is not None:
        print(f"  DEF-GRW next 3m in current phase: {r3['raw']:+.1f} pp/yr (95% CI {r3['raw_lo']:+.1f} to "
              f"{r3['raw_hi']:+.1f}), t={r3['raw_t']:.2f}, {int(r3['episodes'])} episodes -> {thesis_verdict(r3)}")
    print(f"  Probability-weighted DEF-GRW (next-month returns, 3 months): {expected:+.1f} pp/yr")
    print(f"  HEADLINE slope test: DEF-GRW next 3m at today's readings {s['fit']:+.1f} pp/yr "
          f"(95% CI {s['lo']:+.1f} to {s['hi']:+.1f}), t={s['t']:.2f}")
    print(f"    = sample average {s['avg']:+.1f} (secular drift) + cycle tilt {s['tilt']:+.1f} pp/yr "
          f"(t={s['tilt_t']:.2f}; positive = cycle points toward defensives)")
    print(f"  Verdict (pre-registered, one-sided 5%, on the expected spread): {slope_verdict(s)}")
    print(f"  Cycle-tilt verdict (same rule, on the tilt alone): {thesis_verdict({'raw_t': s['tilt_t']})}")
    c = slopes["CYC-DEF"]
    print(f"  Cross-check CYC-DEF slope test: {c['fit']:+.1f} pp/yr, t={c['t']:.2f} "
          f"(a defensive thesis expects this to be NEGATIVE)")
    print("=" * 90)
 
 
def robustness():
    """Does the phase read and the headline slope test survive alternative specifications?"""
    ex1, ex3 = returns_panels()
    base = build_cycle()["phase"]
    specs = [("baseline", {}), ("window 180m", {"window": 180}), ("expanding window", {"window": None}),
             ("no smoothing", {"smooth": 1}), ("3m changes", {"chg_n": 3}),
             ("level band 0.10", {"band_l": 0.10}), ("level band 0.25", {"band_l": 0.25}),
             ("momentum band 0.15", {"band_m": 0.15}), ("momentum band 0.35", {"band_m": 0.35}),
             ("no hysteresis", {"sticky": False}), ("credit = Baa spread", {"credit": "baa"})]
    specs += [(f"drop {c}", {"cols": [x for x in M_COLS if x != c]}) for c in M_COLS]
    rows = []
    for name, kw in specs:
        print(f"  running: {name}")
        C = build_cycle(**kw)
        ph = C["phase"]
        cmp = pd.concat([base, ph], axis=1, keys=["b", "r"], sort=True).loc[SAMPLE_START:].dropna()
        s = slope_test(C["lvl"], C["mom"], ex3)
        rows.append({"spec": name, "agree_%": round((cmp.b == cmp.r).mean() * 100),
                     "transition_%": round((ph.loc[SAMPLE_START:] == "Transition").mean() * 100),
                     "latest": ph.dropna().iloc[-1], "level": round(C["lvl"].dropna().iloc[-1], 2),
                     "mom": round(C["mom"].dropna().iloc[-1], 2),
                     "slope fit": round(s["fit"], 1), "t": round(s["t"], 2), "verdict": slope_verdict(s),
                     "tilt": round(s["tilt"], 1), "tilt t": round(s["tilt_t"], 2),
                     "tilt verdict": thesis_verdict({"raw_t": s["tilt_t"]})})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cycle_robustness.csv", index=False)
    print("\n", out.to_string(index=False))
 
 
if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()