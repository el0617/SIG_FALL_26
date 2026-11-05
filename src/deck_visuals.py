"""
Deck visuals for the TOPDOWN2 final presentation (13 Nov 2026).

    python -m src.deck_visuals

Draws ten presentation charts from the FROZEN data (branch freeze-2026-11-04) and writes them to
outputs/deck/. No data are downloaded and no model is re-estimated: every number is read from
data/raw caches or from outputs/*.csv written by the modules in the freeze run.

Each figure is sized to its placeholder on the 1920x1080 slide canvas (pixels / 100 = inches, saved at
dpi 200, i.e. 2x) and uses the deck's background and palette, so it can replace the dashed box as is.

  deck_v01_active_weights.png     Visual 1   current active weights vs IVV       weighting_weights.csv
  deck_v03_wti_2026.png           Visual 3   WTI daily, 2026, conflict events      data/raw/DCOILWTICO.csv
  deck_v04_pe_premium.png         Visual 4   growth vs defensive fwd P/E premium   valuation_signals.csv
  deck_v05_def_minus_grw.png      Visual 5   defensives minus growth since 29 May  data/raw/prices_monthly.csv
  deck_v06_decomposition.png      Visual 6   (def - mkt) and (grw - mkt) legs      tournament_baseline.csv
  deck_v07_valuation_revisions.png Visual 7  sector fwd P/E percentile vs revisions valuation_sectors.csv
  deck_v08_oil_collapse_loss.png  Visual 8   oil-glut loss by XLE active weight    energy_stress.csv, weighting_weights.csv
  deck_v09_weights_before_after.png Visual 9 active weights, current vs Base       weighting_weights.csv
  deck_v10_scenario_returns.png   Visual 10  3-month active return by scenario     consistency_scenario_payoffs.csv
  deck_v11_trigger_readings.png   Visual 11  review triggers: reading vs threshold T10YIE, DCOILWTICO, cycle_state, valuation_sectors

Visuals 4 and 7 restyle Module 7 outputs (outputs/valuation_gap.png, outputs/valuation_sectors.png).
Visual 2 (CPI composition) needs BLS energy-CPI data that are not cached; Visual 12 has no source in the repo.

Conventions follow the models: DEF = XLP/XLU/XLV, GRW = VGT/XLY with XLK as the VGT proxy (the target
series), legs equal-weighted, monthly returns from adjusted closes; month in progress dropped.
"""
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"
DECK = OUT / "deck"
DECIDE = pd.Timestamp("2026-11-04")         # freeze / decision date
ASOF = pd.Timestamp("2026-05-29")           # portfolio positioned (Bloomberg PORT screen)
LAST_MONTH = pd.Timestamp("2026-10-31")     # last complete month in the freeze

# ---- deck style -------------------------------------------------------------------------------------
BG, INK, INK2, MUTED, RULE = "#F6F4EF", "#14213D", "#4A5568", "#5B6270", "#DDD8CC"
BLUE, ORANGE, PURPLE, GREY = "#2B6CB0", "#C2410C", "#6E5A9E", "#A39E93"   # validated: blue/orange/purple
DEF, INF, GRW = ["XLP", "XLU", "XLV"], ["XLE", "XLB"], ["VGT", "XLY"]
GROUP_COLOR = {**{t: BLUE for t in DEF}, **{t: ORANGE for t in GRW}, **{t: PURPLE for t in INF}}
ORDER = ["VGT", "XLY", "XLC", "XLF", "XLI", "SCHH", "XLB", "XLE", "XLP", "XLV", "XLU"]   # growth .. defensives

for p in glob.glob(os.path.expanduser("~/.fonts/*.ttf")) + glob.glob(str(ROOT / "assets" / "fonts" / "*.ttf")):
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
_names = {f.name for f in fm.fontManager.ttflist}
FONT = "IBM Plex Sans" if "IBM Plex Sans" in _names else ("Arial" if "Arial" in _names else "DejaVu Sans")
BOLD = {"fontfamily": "IBM Plex Sans SmBld"} if "IBM Plex Sans SmBld" in _names else {"fontweight": "bold"}

plt.rcParams.update({
    "font.family": FONT, "font.size": 13, "axes.facecolor": BG, "figure.facecolor": BG,
    "savefig.facecolor": BG, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": RULE, "grid.linewidth": 0.8,
    "legend.frameon": False, "axes.unicode_minus": True,
})


def figure(w_px, h_px, **kw):
    return plt.subplots(figsize=(w_px / 100, h_px / 100), **kw)


def title(fig, text, sub=None):
    """Title 14px from the top edge, subtitle 30px below it (pixel offsets on the slide canvas)."""
    h = fig.get_figheight() * 100
    fig.text(0.012, 1 - 12 / h, text, ha="left", va="top", fontsize=15, color=INK, **BOLD)
    if sub:
        fig.text(0.012, 1 - 40 / h, sub, ha="left", va="top", fontsize=12, color=MUTED)


def top_for(fig, sub):
    """Axes top that leaves room for the title block (plus 18px)."""
    return 1 - ((66 if sub else 42) + 18) / (fig.get_figheight() * 100)


def save(fig, name, top=None):
    if top is not None:
        fig.subplots_adjust(top=top)
    DECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK / name, dpi=200)
    plt.close(fig)
    print(f"  saved outputs/deck/{name}")


def pp(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def active_weights():
    W = pd.read_csv(OUT / "weighting_weights.csv", index_col=0)
    return W[["Current", "Base"]].sub(W["benchmark"], axis=0)


def hbar_labels(ax, y, vals, fmt=lambda v: pp(v), pad=0.25, size=12):
    for yi, v in zip(y, vals):
        ax.text(v + (pad if v >= 0 else -pad), yi, fmt(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=size, color=INK2)


# ---- Visual 1 -------------------------------------------------------------------------------------
def v01():
    A = active_weights()["Current"].reindex(ORDER)
    fig, ax = figure(1052, 656)
    y = np.arange(len(A))[::-1]
    ax.barh(y, A.values, height=0.62, color=[GROUP_COLOR.get(t, GREY) for t in A.index], edgecolor=BG, lw=2)
    hbar_labels(ax, y, A.values)
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(A.min() - 2.5, A.max() + 2.5)
    ax.set_xlabel("Active weight vs IVV sector benchmark (pp)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, GREY, PURPLE, BLUE)]
    ax.legend(handles, ["Growth", "Other", "Inflation hedge", "Defensives"], loc="upper right", ncol=1, fontsize=12)
    title(fig, "Current active weights by ETF", "Drifted from 29 May through 31 Oct; percentage points vs IVV")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11)
    save(fig, "deck_v01_active_weights.png", top=top_for(fig, True))


# ---- Visual 3 -------------------------------------------------------------------------------------
EVENTS = [("2026-02-28", "War begins", "left", 0), ("2026-04-08", "Ceasefire", "left", 1),
          ("2026-06-17", "Islamabad memorandum", "right", 0), ("2026-07-08", "Memorandum collapses", "left", 1),
          ("2026-10-25", "Qatar draft agreement", "right", 0)]


def v03():
    w = pd.read_csv(RAW / "DCOILWTICO.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
    w = w.loc["2026-01-01":DECIDE]
    fig, ax = figure(1040, 290)
    ax.plot(w.index, w.values, color=INK, lw=2)
    lo, hi = w.min(), w.max()
    ax.set_ylim(lo - 6, hi + 26)
    for d, lab, ha, lvl in EVENTS:
        d = pd.Timestamp(d)
        ax.axvline(d, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(d, hi + 25 - 11 * lvl, f" {lab} " if ha == "left" else f"{lab} ", fontsize=11, color=INK2,
                va="top", ha=ha)
    ax.plot(w.index[-1], w.iloc[-1], "o", ms=8, color=ORANGE, mec=BG, mew=2)
    ax.text(w.index[-1], w.iloc[-1] - 6, f"${w.iloc[-1]:.0f}", fontsize=12, color=INK, ha="right", va="top")
    pk = pd.Timestamp(w.idxmax())
    ax.text(matplotlib.dates.date2num(pk) - 3, w.max() + 1.0, f"Peak ${w.max():.0f} ", fontsize=11, color=INK2, ha="right", va="bottom")
    ax.set_ylabel("$ per barrel"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    title(fig, f"WTI crude oil, daily, 2 Jan – {w.index[-1]:%-d %b} 2026")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11)
    save(fig, "deck_v03_wti_2026.png", top=top_for(fig, False))
    return w


# ---- Visual 4 -------------------------------------------------------------------------------------
def v04():
    V = pd.read_csv(OUT / "valuation_signals.csv", index_col=0, parse_dates=True)
    prem = (np.exp(V["gap"].dropna()) - 1) * 100            # gap = log(growth fwd P/E / defensive fwd P/E)
    med = prem.expanding(min_periods=36).median()            # median of history known at each date
    fig, ax = figure(952, 656)
    ax.axhline(0, color=INK, lw=1)
    ax.plot(prem.index, prem.values, color=ORANGE, lw=2, label="Growth premium over defensives")
    ax.plot(med.index, med.values, color=INK, lw=1.4, ls=(0, (5, 3)), label="Median of history to date")
    ax.plot(prem.index[-1], prem.iloc[-1], "o", ms=9, color=ORANGE, mec=BG, mew=2, zorder=3)
    end = prem.index[-1] + pd.Timedelta(days=75)
    ax.text(end, prem.iloc[-1] + 1.5, f"{prem.index[-1]:%b %Y}\n{pp(prem.iloc[-1], 0)}%", ha="left", va="bottom",
            fontsize=12, color=INK, **BOLD)
    ax.text(end, med.iloc[-1] - 1.5, f"Median\n{pp(med.iloc[-1], 0)}%", ha="left", va="top", fontsize=12, color=INK2)
    ax.set_xlim(prem.index[0] - pd.Timedelta(days=120), prem.index[-1] + pd.Timedelta(days=900))
    ax.set_ylabel("Forward P/E premium (%)"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1.0), fontsize=12)
    title(fig, "Forward P/E premium of growth over defensives",
          f"XLK and XLY vs XLP, XLU and XLV; monthly {prem.index[0]:%b %Y} – {prem.index[-1]:%b %Y}")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.07)
    save(fig, "deck_v04_pe_premium.png", top=top_for(fig, True))
    return prem, med


# ---- Visual 7 -------------------------------------------------------------------------------------
def v07():
    T = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0).drop(index="SPY", errors="ignore")
    col = {t: (BLUE if t in DEF else ORANGE if t in ("XLK", "XLY") else GREY) for t in T.index}
    fig, ax = figure(672, 656)
    ax.axvline(50, color=RULE, lw=1.2); ax.axhline(0, color=INK, lw=1)
    for t, r in T.iterrows():
        ax.plot(r["pe_pct"], r["rev3"], "o", ms=12 if t in DEF else 10, color=col[t], mec=BG, mew=2, zorder=3)
        lab = f"{t}  {r['fwd_pe']:.1f}x" if t in DEF else t
        if t == "XLV":      # most expensive point: label below it, inside the frame
            ax.text(104, r["rev3"] - 0.55, lab, fontsize=12, color=INK, va="top", ha="right", **BOLD)
            continue
        left = r["pe_pct"] > 85
        ax.text(r["pe_pct"] + (-2 if left else 2), r["rev3"] + 0.35, lab, fontsize=12, color=INK, va="bottom",
                ha="right" if left else "left", **(BOLD if t in DEF else {}))
    ax.set_xlim(0, 105); ax.set_ylim(-1, max(18, T["rev3"].max() + 2))
    ax.set_xlabel("Forward P/E percentile vs own history since 2006")
    ax.set_ylabel("3-month change in forward EPS (%)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=c) for c in (BLUE, ORANGE, GREY)]
    ax.legend(handles, ["Defensives (forward P/E)", "Growth", "Other"], loc="upper left", fontsize=11.5)
    title(fig, "Valuation and earnings revisions by sector", "Bloomberg estimates as of 30 Sep 2026")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.10)
    save(fig, "deck_v07_valuation_revisions.png", top=top_for(fig, True))
    return T


# ---- Visual 5 -------------------------------------------------------------------------------------
def v05():
    r = pd.read_csv(RAW / "prices_monthly.csv", index_col=0, parse_dates=True).loc[:LAST_MONTH]
    D, G = r[DEF].mean(axis=1), r[["XLK", "XLY"]].mean(axis=1)
    win = slice("2026-06-01", LAST_MONTH)
    cd, cg = (1 + D.loc[win]).cumprod() - 1, (1 + G.loc[win]).cumprod() - 1
    s = pd.concat([pd.Series([0.0], index=[ASOF]), (cd - cg) * 100])
    fig, ax = figure(860, 300)
    ax.axvspan(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-31"), color=ORANGE, alpha=0.10, lw=0)
    ax.plot(s.index, s.values, color=BLUE, lw=2.2, marker="o", ms=8, mec=BG, mew=2)
    ax.axhline(0, color=INK, lw=1)
    for d, v in s.iloc[1:].items():
        ax.text(d, v + 1.2, pp(v), ha="center", va="bottom", fontsize=12, color=INK2)
    oct_ = (D - G).loc[LAST_MONTH] * 100
    ax.text(pd.Timestamp("2026-10-15"), 1.0, f"October\n{pp(oct_)}%", ha="center", va="bottom", fontsize=11, color=ORANGE)
    ax.set_ylim(min(-2, s.min() - 2), s.max() + 5)
    ax.set_ylabel("Cumulative difference (pp)"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xticks(s.index, [f"{d:%-d %b}" for d in s.index])
    title(fig, "Defensives minus growth since positioning on 29 May",
          "Cumulative return, XLP/XLU/XLV minus XLK/XLY, equal-weighted; month-end")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.11)
    save(fig, "deck_v05_def_minus_grw.png", top=top_for(fig, True))
    return s, oct_


# ---- Visual 6 -------------------------------------------------------------------------------------
def v06():
    T = pd.read_csv(OUT / "tournament_baseline.csv", index_col=0)
    d, g = T.loc["D"], T.loc["E"]      # D: defensives - market, E: growth - market (A = D - E)
    assert abs((d["mean_pp_yr"] - g["mean_pp_yr"]) - T.loc["A", "mean_pp_yr"]) < 1e-3
    panels = [("Average excess return\n(pp per year)", "mean_pp_yr", "t_mean"),
              ("Alpha beyond beta\n(pp per year)", "alpha_pp_yr", "t_alpha"),
              ("Average sell-off quarter\n(%, SPY down ≥ 5%)", "selloff_3m", "t_selloff")]
    fig, axes = figure(932, 656, ncols=3, sharey=False)
    for ax, (lab, c, t) in zip(axes, panels):
        vals = [d[c], g[c]]
        ax.bar([0, 1], vals, width=0.6, color=[BLUE, ORANGE], edgecolor=BG, lw=2)
        ax.axhline(0, color=INK, lw=1)
        lim = max(6.5, max(abs(v) for v in vals) * 1.35)
        ax.set_ylim(-lim * 0.45, lim)
        for x, v, tt in zip([0, 1], vals, [d[t], g[t]]):
            ax.text(x, v + (0.25 if v >= 0 else -0.25), f"{pp(v)}\n(t {pp(tt)})", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=12, color=INK2)
        ax.set_xticks([0, 1], ["Defensives\nvs market", "Growth\nvs market"], fontsize=12)
        ax.set_title(lab, fontsize=13, color=INK, pad=10)
        ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    title(fig, "Decomposition: defensives minus growth = (defensives − market) − (growth − market)",
          f"Monthly data {pd.Timestamp(d['start']):%b %Y} – {pd.Timestamp(d['end']):%b %Y}; thesis tournament baseline")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, wspace=0.35)
    save(fig, "deck_v06_decomposition.png", top=top_for(fig, True) - 62 / 656)
    return d, g


# ---- Visual 8 -------------------------------------------------------------------------------------
def v08():
    S = pd.read_csv(OUT / "energy_stress.csv", index_col=0)
    glut = S.loc[[i for i in S.index if i.startswith("Oil glut")][0], "XLE minus SPY (%)"] / 100   # -0.4068
    C = pd.read_csv(OUT / "consistency_summary.csv", index_col=0)      # XLE active weight and oil-crash loss
    marks = [("Current", "Current (drifted)", GREY), ("Team proposal", "Proposal", GREY),
             ("Full hedge", "Base (full hedge, post hoc)", GREY), ("Recommended", "Base (half hedge)", BLUE)]
    marks = [(lab, C.loc[k, "xle_active"], C.loc[k, "oil_crash_tail_pp"], c) for lab, k, c in marks]
    x = np.linspace(0, 7.5, 100)
    fig, ax = figure(860, 440)
    ax.axhspan(-1, 0, color=BLUE, alpha=0.07, lw=0)
    ax.plot(x, glut * x, color=INK, lw=2)
    ax.axhline(-1, color=ORANGE, lw=1.8, ls=(0, (5, 3)))
    ax.text(7.45, -0.93, "Limit: −1pp", color=ORANGE, fontsize=12, ha="right", va="bottom")
    offs = {"Current": (-0.2, -0.22, "right"), "Team proposal": (0.15, 0.12, "left"),
            "Full hedge": (-0.15, -0.28, "right"), "Recommended": (0.18, 0.12, "left")}
    for lab, w, y, c in marks:
        ax.plot(w, y, "o", ms=10, color=c, mec=INK if c == GREY else BG, mew=1.5 if c == GREY else 2, zorder=3)
        dx, dy, ha = offs[lab]
        ax.text(w + dx, y + dy, f"{lab}\n{pp(w)}pp → {pp(y, 2)}pp", fontsize=11.5, color=INK, ha=ha,
                va="top" if dy < 0 else "bottom", **(BOLD if lab == "Recommended" else {}))
    ax.set_xlim(0, 7.5); ax.set_ylim(-3.75, 0.6)
    ax.set_xlabel("XLE active weight (pp)"); ax.set_ylabel("Loss (pp)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    title(fig, "Loss in a 2014–16 oil collapse, by XLE active weight",
          f"XLE minus SPY in the Jul 2014 – Jan 2016 episode: {glut * 100:.1f}%".replace("-", "−"))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13)
    save(fig, "deck_v08_oil_collapse_loss.png", top=top_for(fig, True))
    return glut, marks


# ---- Visual 9 -------------------------------------------------------------------------------------
def v09():
    A = active_weights().reindex(ORDER)
    fig, ax = figure(820, 440)
    y = np.arange(len(A))[::-1]; h = 0.38
    ax.barh(y + h / 2, A["Current"], height=h, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - h / 2, A["Base"], height=h, color=BLUE, edgecolor=BG, lw=1.5, label="Recommended")
    hbar_labels(ax, y - h / 2, A["Base"].values, size=10.5, pad=0.2)
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(A.values.min() - 2.2, A.values.max() + 2.2)
    ax.set_xlabel("Active weight vs IVV (pp)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", fontsize=12)
    title(fig, "Active weights: current vs recommended", "Labels: recommended active weight (pp)")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12)
    save(fig, "deck_v09_weights_before_after.png", top=top_for(fig, True))
    return A


# ---- Visual 10 ------------------------------------------------------------------------------------
def v10():
    P = pd.read_csv(OUT / "consistency_scenario_payoffs.csv", index_col=0)
    P = P[["Current (drifted)", "Base (half hedge)"]]
    names = {"Stagflationary sell-off": "Stagflationary\nsell-off"}
    fig, ax = figure(672, 656)
    y = np.arange(len(P))[::-1]; h = 0.38
    ax.barh(y + h / 2, P.iloc[:, 0], height=h, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - h / 2, P.iloc[:, 1], height=h, color=BLUE, edgecolor=BG, lw=1.5, label="Recommended")
    for yy, col in ((y + h / 2, 0), (y - h / 2, 1)):
        hbar_labels(ax, yy, P.iloc[:, col].values, fmt=lambda v: pp(v, 2), pad=0.04, size=11)
    ax.set_yticks(y, [names.get(s, s) for s in P.index]); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.75, 1.55)
    ax.set_xlabel("Active return vs SPY, 3 months (%)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=12)
    title(fig, "Average 3-month return by scenario", "Module 8 scenario tree, 2003 onward")
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.10)
    save(fig, "deck_v10_scenario_returns.png", top=top_for(fig, True))
    return P


# ---- Visual 11 ------------------------------------------------------------------------------------
def v11(wti):
    be = pd.read_csv(RAW / "T10YIE.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
    be_ref = be.loc[:DECIDE].iloc[-1]
    w26 = wti.loc["2026"]; decline = (1 - w26.iloc[-1] / w26.max()) * 100
    cs = pd.read_csv(OUT / "cycle_state.csv", index_col=0, parse_dates=True)["p_recession_12m"].dropna()
    vs = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0)
    rows = [
        ("10-year breakeven", f"{be_ref:.2f}% on {be.loc[:DECIDE].index[-1]:%-d %b}", be_ref, be_ref + 0.25,
         (be_ref - 0.5, be_ref + 0.5), "%", "Trigger: +25bp → full hedge"),
        ("WTI decline from 2026 peak", f"{decline:.0f}% (peak ${w26.max():.0f}, {w26.idxmax():%-d %b})", decline, 25,
         (0, 40), "%", "Trigger: −25% → XLE to about +1pp"),
        ("Recession probability, 12 months", f"{cs.iloc[-1] * 100:.0f}% ({cs.index[-1]:%b})", cs.iloc[-1] * 100, 35,
         (0, 50), "%", "Trigger: above 35% → add defensives"),
        ("Technology EPS revisions, 3 months", f"{pp(vs.loc['XLK', 'rev3'])}% (30 Sep)", vs.loc["XLK", "rev3"], 15,
         (0, 25), "%", "Condition: above +15% → do not add to growth underweight"),
    ]
    fig, axes = figure(672, 656, nrows=4)
    for ax, (name, reading, v, thr, (lo, hi), unit, rule) in zip(axes, rows):
        met = v >= thr
        ax.barh(0, hi - lo, left=lo, height=0.5, color=RULE, lw=0)
        ax.barh(0, (min(v, thr) - lo), left=lo, height=0.5, color=BLUE, alpha=0.35, lw=0)
        ax.plot(v, 0, "o", ms=12, color=ORANGE if met else BLUE, mec=BG, mew=2, zorder=3)
        ax.plot([thr, thr], [-0.42, 0.42], color=INK, lw=2.5)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _, d=(2 if hi - lo < 2 else 0): f"{x:.{d}f}%"))
        ax.text(lo, 0.72, name, fontsize=13, color=INK, va="bottom", transform=ax.transData, **BOLD)
        status = "Met" if met else "Not triggered"
        ax.text(hi, 0.72, f"{reading} · {status}", fontsize=11.5, color=INK2, va="bottom", ha="right")
        ax.text(lo, -1.25, rule, fontsize=11, color=MUTED, va="top")
    title(fig, "Review triggers: current readings vs thresholds", "Dot: reading · vertical line: threshold")
    fig.subplots_adjust(left=0.03, right=0.965, bottom=0.10, hspace=1.6)
    save(fig, "deck_v11_trigger_readings.png", top=top_for(fig, True) - 34 / 656)
    return rows


def main():
    print("\nDeck visuals from frozen data (freeze-2026-11-04)\n")
    v01()
    wti = v03()
    prem, med = v04()
    T7 = v07()
    s, oct_ = v05()
    d, g = v06()
    glut, marks = v08()
    A = v09()
    P = v10()
    rows = v11(wti)
    print("\nNumbers drawn (check against the deck):")
    print(f"  Defensives minus growth: October {oct_:+.2f}%, cumulative since 29 May {s.iloc[-1]:+.2f}pp")
    print(f"  Legs: defensives alpha {d['alpha_pp_yr']:+.2f} (t {d['t_alpha']:.2f}), sell-off {d['selloff_3m']:+.2f} "
          f"(t {d['t_selloff']:.2f}); growth alpha {g['alpha_pp_yr']:+.2f} (t {g['t_alpha']:.2f})")
    print("  Oil-collapse loss: " + ", ".join(f"{l} {w:+.2f}pp -> {y:+.2f}pp" for l, w, y, _ in marks))
    print("  Active weights (Current -> Base): " + ", ".join(f"{t} {a:+.1f}->{b:+.1f}" for t, (a, b) in A.iterrows()))
    print("  Scenario returns (Current / Base): " + ", ".join(f"{s_}: {a:+.2f}/{b:+.2f}" for s_, (a, b) in P.iterrows()))
    print("  Triggers: " + "; ".join(f"{r[0]} = {r[1]} vs {r[3]:.2f}" for r in rows))
    print(f"  P/E premium: {prem.iloc[-1]:+.1f}% vs median {med.iloc[-1]:+.1f}%; defensive fwd P/E "
          + ", ".join(f"{t} {T7.loc[t, 'fwd_pe']:.1f}x (pct {T7.loc[t, 'pe_pct']:.0f})" for t in DEF))
    print(f"\nSaved to {DECK.relative_to(ROOT)}/\n")


if __name__ == "__main__":
    main()
