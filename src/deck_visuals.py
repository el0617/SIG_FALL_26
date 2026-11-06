"""
Deck visuals for the Mean Green Fund / TOPDOWN2 final presentation (13 Nov 2026), team style.

    python -m src.deck_visuals

Draws every chart of the final deck from the FROZEN data (branch freeze-2026-11-04) into outputs/deck/.
No downloads and no re-estimation: numbers come from data/raw caches and outputs/*.csv of the freeze run.
The only hard-coded inputs are the BLS September 2026 CPI component rates (CPI_SEP26 below), which are
not cached; their source is noted next to them.

Style (team template): white background, Arial (Liberation Sans has the same metrics where Arial is not
installed), black text and axes, no gridlines, one text size. Every chart is sized to its box on the
1920x1080 slide canvas (pixels / 100 = inches, saved at 2x) and all chart text is 24px on the slide, the
same size as the slides' body text; titles are the same size in bold. Series colours: dark green #13501B,
red #A82B0B, teal #104862, grey #A6ADA7.

Portfolio weights are the team's final decision (5 Nov 2026), evaluated by src.team_weights against S&P 500
sector weights at the 4 Nov close; equity weights are % of the 60% equity allocation.

Conventions follow the models: DEF = XLP/XLU/XLV, GRW = VGT/XLY with XLK as the VGT proxy (the target
series), legs equal-weighted, monthly total returns, month in progress dropped.
"""
from pathlib import Path
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"
DECK = OUT / "deck"
DECIDE = pd.Timestamp("2026-11-04")
ASOF = pd.Timestamp("2026-05-29")
LAST_MONTH = pd.Timestamp("2026-10-31")

# BLS CPI, September 2026, 12-month change (%), released 14 Oct 2026 (as reported by Securities.io / Schwab)
CPI_SEP26 = {"Gasoline": 34.2, "Energy": 20.4, "All items": 3.7, "Shelter": 3.2, "Core": 2.6}

BG, INK = "#FFFFFF", "#000000"
GREEN, RED, TEAL, LIGHT, GREY, TRACK = "#13501B", "#A82B0B", "#104862", "#8ED974", "#A6ADA7", "#E6E8E6"
FS = 17.28          # 24px on the slide: 1px = 0.01in = 0.72pt
DEF, INF, GRW = ["XLP", "XLU", "XLV"], ["XLE", "XLB"], ["VGT", "XLY"]
GROUP_COLOR = {**{t: GREEN for t in DEF}, **{t: RED for t in GRW}, **{t: TEAL for t in INF}}
ORDER = ["VGT", "XLY", "XLC", "XLF", "XLI", "SCHH", "XLB", "XLE", "XLP", "XLV", "XLU"]

for p in glob.glob(os.path.expanduser("~/.fonts/*.ttf")) + glob.glob(str(ROOT / "assets" / "fonts" / "*.ttf")):
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
_names = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _names), "DejaVu Sans")
BOLD = {"fontweight": "bold"}

plt.rcParams.update({
    "font.family": FONT, "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.titlesize": FS, "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": 1.2,
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.4, "axes.unicode_minus": True,
})


def figure(w_px, h_px, **kw):
    return plt.subplots(figsize=(w_px / 100, h_px / 100), **kw)


def title(fig, text):
    fig.text(0.5, 1 - 6 / (fig.get_figheight() * 100), text, ha="center", va="top", fontsize=FS, **BOLD)


def save(fig, name, extra=0):
    fig.subplots_adjust(top=1 - (46 + extra) / (fig.get_figheight() * 100))
    DECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK / name, dpi=200)
    plt.close(fig)
    print(f"  saved outputs/deck/{name}")


def pp(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def series(name):
    return pd.read_csv(RAW / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()


# Team final weights (5 Nov 2026) vs S&P 500 sector weights (4 Nov), from `python -m src.team_weights`
CUR, NEW = "Old (5 Nov)", "New (team)"
FI_TYPES = {"Treasury and agency": ["VGIT", "AGZ"], "TIPS": ["IBID"], "Investment-grade corporate": ["IGSB", "VCSH"],
            "Cash": ["Cash"]}


def active_weights():
    A = pd.read_csv(OUT / "team_weights_active.csv", index_col=0)
    return A[[CUR, NEW]].rename(columns={CUR: "Current", NEW: "Base"})


def team_metrics():
    return pd.read_csv(OUT / "team_weights_metrics.csv", index_col=0)


def hbar_labels(ax, y, vals, fmt=lambda v: pp(v), pad=0.25, **kw):
    for yi, v in zip(y, vals):
        ax.text(v + (pad if v >= 0 else -pad), yi, fmt(v), va="center", ha="left" if v >= 0 else "right", fontsize=FS, **kw)


def bars_frame(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def date_axis(ax, months=(1, 7), fmt="%b %Y"):
    ax.xaxis.set_major_locator(mdates.MonthLocator(months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# ---- Proposal ----------------------------------------------------------------------------------------------
def weights_active(w=700, h=708):
    A = active_weights().reindex(ORDER)
    fig, ax = figure(w, h)
    y = np.arange(len(A))[::-1]; hh = 0.38
    ax.barh(y + hh / 2, A["Current"], height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, A["Base"], height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-10.5, 6); ax.set_xlabel("Active weight vs S&P 500 (pp)")
    bars_frame(ax); ax.legend(loc="lower left")
    title(fig, "Equity active weights: current vs proposed")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.1)
    save(fig, "deck_weights_active.png")
    return A


# ---- Present -----------------------------------------------------------------------------------------------
def performance():
    from src.portfolio import etf_returns, DEFAULT, HOLDINGS, GROUPS
    R = etf_returns().loc["2026-06-30":LAST_MONTH]
    w = DEFAULT.set_index("ticker")["weight"]; w = w / w.sum()
    b = DEFAULT.set_index("ticker")["benchmark"]; b = b / b.sum()
    rows, contrib = [], pd.Series(0.0, index=HOLDINGS)
    for d, r in R.iterrows():
        rp, rb = (w * r[HOLDINGS]).sum(), (b * r[HOLDINGS]).sum()
        contrib += (w - b) * (r[HOLDINGS] - rb)
        rows.append((d, rp, rb, r["SPY"]))
        w = w * (1 + r[HOLDINGS] / 100); w /= w.sum(); b = b * (1 + r[HOLDINGS] / 100); b /= b.sum()
    M = pd.DataFrame(rows, columns=["date", "TOPDOWN2", "Sector benchmark", "SPY"]).set_index("date")
    C = (1 + M / 100).cumprod() * 100 - 100
    C.loc[ASOF] = 0.0; C = C.sort_index()
    fig, ax = figure(1100, 708)
    ax.axvspan(pd.Timestamp("2026-09-30"), LAST_MONTH, color=LIGHT, alpha=0.25, lw=0)
    for col, c, lw in (("TOPDOWN2", GREEN, 3), ("Sector benchmark", TEAL, 2.2), ("SPY", GREY, 2.2)):
        ax.plot(C.index, C[col], color=c, lw=lw, marker="o", ms=8, mec=BG, mew=1.5, label=f"{col} ({pp(C[col].iloc[-1])}%)")
    ax.axhline(0, color=INK, lw=1.2)
    ax.text(pd.Timestamp("2026-10-15"), -1.25, "October", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(C.index, [f"{d:%-d %b}" for d in C.index])
    ax.set_ylim(-1.4, 5.8)
    ax.set_ylabel("Cumulative total return (%)")
    ax.legend(loc="upper left")
    title(fig, "Cumulative return since 29 May 2026")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.08)
    save(fig, "deck_performance.png")
    groups = {g: contrib[t].sum() for g, t in GROUPS.items()}
    return M, C, groups


def active_current():
    A = active_weights()["Current"].rename({"CASH": "Cash"}).reindex(ORDER + ["Cash"])
    A["Cash"] = np.floor(A["Cash"] * 10 + 1e-6) / 10      # 100 - 96.55: shown as 3.4 so it matches 96.6 + 3.4 = 100
    fig, ax = figure(1100, 708)
    y = np.arange(len(A))[::-1]
    cols = [GROUP_COLOR.get(t, GREY) for t in A.index]; cols[-1] = "#FFFFFF"
    ax.barh(y, A.values, height=0.62, color=cols, edgecolor=[BG] * (len(A) - 1) + [INK], lw=2,
            hatch=[None] * (len(A) - 1) + ["//"])
    hbar_labels(ax, y, A.values)
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-11.5, 7.5); ax.set_xlabel("Active weight vs S&P 500 (pp of the 60% equity allocation)")
    bars_frame(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RED, GREY, TEAL, GREEN)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor=INK, hatch="//"))
    ax.legend(handles, ["Growth", "Other", "Inflation hedge", "Defensives", "Equities below 60%"], loc="lower left")
    title(fig, "Current equity active weights by ETF (5 Nov)")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.1)
    save(fig, "deck_active_current.png")


# ---- Macro ------------------------------------------------------------------------------------------------
def cpi_components():
    S = pd.Series(CPI_SEP26)
    colors = [RED if k in ("Gasoline", "Energy") else (INK if k == "All items" else GREEN) for k in S.index]
    fig, ax = figure(1060, 708)
    y = np.arange(len(S))[::-1]
    ax.barh(y, S.values, height=0.6, color=colors, edgecolor=BG, lw=2)
    hbar_labels(ax, y, S.values, fmt=lambda v: f"{v:.1f}%", pad=0.5)
    ax.set_yticks(y, S.index); ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(2, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.4, -0.62, "Fed target 2%", fontsize=FS, va="center")
    ax.set_xlim(0, 40); ax.set_ylim(-0.85, len(S) - 0.4)
    ax.set_xlabel("Change over 12 months (%)")
    bars_frame(ax)
    title(fig, "Consumer prices by component, September 2026")
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.1)
    save(fig, "deck_cpi_components.png")


def activity():
    ip, rs = series("INDPRO"), series("RRSFS")
    a = ((ip / ip.shift(12) - 1) * 100).loc["2024-01":]
    b = ((rs / rs.shift(12) - 1) * 100).loc["2024-01":]
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot(a.index, a.values, color=GREEN, lw=2.6, label=f"Industrial production ({pp(a.iloc[-1])}%)")
    ax.plot(b.index, b.values, color=RED, lw=2.6, label=f"Real retail sales ({pp(b.iloc[-1])}%)")
    date_axis(ax)
    ax.set_ylabel("Change over 12 months (%)")
    ax.legend(loc="lower right")
    title(fig, f"Industrial production and real retail sales, to {a.index[-1]:%b %Y}")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.08)
    save(fig, "deck_activity.png")
    return a.iloc[-1], b.iloc[-1]


def payrolls():
    p = series("PAYEMS").diff().loc["2025-01":]
    avg3 = p.rolling(3).mean()
    fig, ax = figure(1060, 708)
    ax.bar(p.index, p.values, width=22, color=[GREEN if v >= 0 else RED for v in p.values], edgecolor=BG, lw=1)
    ax.plot(avg3.index, avg3.values, color=INK, lw=2.2, label="3-month average")
    ax.axhline(0, color=INK, lw=1.2)
    date_axis(ax, (1, 7))
    ax.set_ylabel("Monthly change (thousands)")
    ax.legend(loc="lower right")
    title(fig, f"Nonfarm payroll growth, Jan 2025 – {p.index[-1]:%b %Y}")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08)
    save(fig, "deck_payrolls.png")
    return p


def fed_funds():
    f = series("DFF").loc["2022-01-01":DECIDE]
    fig, ax = figure(1060, 708)
    ax.plot(f.index, f.values, color=GREEN, lw=2.6, drawstyle="steps-post")
    notes = [("2022-11-15", 5.65, "2022–23\n+525bp"), ("2024-10-15", 5.65, "2024\n−100bp"),
             ("2025-10-15", 4.65, "2025\n−75bp"), ("2026-09-01", 2.55, "Sep 2026\n+25bp")]
    for d, yv, t in notes:
        ax.text(pd.Timestamp(d), yv, t, fontsize=FS, ha="center", va="bottom", **BOLD)
    ax.plot(f.index[-1], f.iloc[-1], "o", ms=10, color=RED, mec=BG, mew=2, zorder=3)
    ax.text(f.index[-1], f.iloc[-1] + 0.25, f"{f.iloc[-1]:.2f}%", ha="right", va="bottom", fontsize=FS)
    ax.set_ylim(0, 6.8)
    date_axis(ax, (1,), "%Y")
    ax.set_ylabel("Effective federal funds rate (%)")
    title(fig, "Federal funds rate, 2022 – 2026")
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.08)
    save(fig, "deck_fed_funds.png")
    return f


def fed_expected():
    f, t2 = series("DFF").loc["2025-01-01":DECIDE], series("DGS2").loc["2025-01-01":DECIDE]
    fig, ax = figure(1000, 708)
    ax.plot(t2.index, t2.values, color=RED, lw=2, label=f"2-year Treasury yield ({t2.iloc[-1]:.2f}%)")
    ax.plot(f.index, f.values, color=GREEN, lw=2.6, drawstyle="steps-post", label=f"Effective fed funds rate ({f.iloc[-1]:.2f}%)")
    gap = (t2.iloc[-1] - f.iloc[-1]) * 100
    x = t2.index[-1] + pd.Timedelta(days=12)
    ax.annotate("", xy=(x, t2.iloc[-1]), xytext=(x, f.iloc[-1]), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(x - pd.Timedelta(days=10), (t2.iloc[-1] + f.iloc[-1]) / 2, f"{gap:+.0f}bp", ha="right", va="center", fontsize=FS, **BOLD)
    ax.set_xlim(f.index[0], x + pd.Timedelta(days=15)); ax.set_ylim(3.3, 5.3)
    date_axis(ax)
    ax.set_ylabel("Percent")
    ax.legend(loc="upper left")
    title(fig, "Fed funds rate and the 2-year Treasury yield")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08)
    save(fig, "deck_fed_expected.png")
    return f.iloc[-1], t2.iloc[-1], gap


def curve():
    t10, t2, f = series("DGS10"), series("DGS2"), series("DFF")
    df = pd.concat([t10, t2, f], axis=1, keys=["t10", "t2", "ff"]).dropna().loc["2019-01-01":DECIDE]
    s1, s2 = (df.t10 - df.t2) * 100, (df.t10 - df.ff) * 100
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot(s2.index, s2.values, color=RED, lw=1.8, label=f"10-year − fed funds ({s2.iloc[-1]:+.0f}bp)")
    ax.plot(s1.index, s1.values, color=GREEN, lw=1.8, label=f"10-year − 2-year ({s1.iloc[-1]:+.0f}bp)")
    ax.set_ylim(-200, 330)
    date_axis(ax, (1,), "%Y")
    ax.set_ylabel("Spread (basis points)")
    ax.legend(loc="upper left")
    title(fig, "Yield curve spreads, 2019 – 2026")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08)
    save(fig, "deck_curve.png")
    return df.iloc[-1], s1.iloc[-1], s2.iloc[-1]


def real_yields():
    n, r, b = series("DGS10"), series("DFII10"), series("T10YIE")
    df = pd.concat([n, r, b], axis=1, keys=["n", "r", "b"]).dropna().loc["2021-01-01":DECIDE]
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1.2)
    for col, c, lab in (("n", INK, "10-year nominal"), ("r", GREEN, "10-year real (TIPS)"), ("b", RED, "10-year breakeven")):
        ax.plot(df.index, df[col], color=c, lw=2, label=f"{lab} ({df[col].iloc[-1]:.2f}%)")
    date_axis(ax, (1,), "%Y")
    ax.set_ylim(-1.5, 6.2)
    ax.set_ylabel("Percent")
    ax.legend(loc="upper left", ncol=1)
    title(fig, "10-year Treasury yield = real yield + breakeven inflation")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08)
    save(fig, "deck_real_yields.png")
    return df.iloc[-1]


# ---- War and AI --------------------------------------------------------------------------------------------
EVENTS = [("2026-02-28", "War begins", "right", 0), ("2026-04-08", "Ceasefire", "left", 1),
          ("2026-06-17", "Memorandum", "right", 0), ("2026-07-08", "Collapse", "left", 1),
          ("2026-10-25", "Qatar draft", "right", 0)]


def wti(w=1060, h=330):
    s = series("DCOILWTICO").loc["2026-01-01":DECIDE]
    fig, ax = figure(w, h)
    ax.plot(s.index, s.values, color=INK, lw=2)
    ax.set_ylim(50, 178)
    for d, lab, ha, lvl in EVENTS:
        d = pd.Timestamp(d)
        ax.axvline(d, color=INK, lw=1, ls=(0, (3, 3)))
        ax.text(d, 176 - 26 * lvl, f" {lab}" if ha == "left" else f"{lab} ", fontsize=FS, va="top", ha=ha)
    ax.plot(s.index[-1], s.iloc[-1], "o", ms=9, color=RED, mec=BG, mew=2)
    ax.text(s.index[-1], s.iloc[-1] - 9, f"${s.iloc[-1]:.0f}", fontsize=FS, ha="right", va="top")
    ax.set_yticks([60, 80, 100, 120])
    ax.set_ylabel("$ per barrel")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    title(fig, f"WTI crude oil, daily, 2 Jan – {s.index[-1]:%-d %b} 2026")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12)
    save(fig, "deck_wti_2026.png")
    return s


def pe_premium(w=1000, h=708):
    V = pd.read_csv(OUT / "valuation_signals.csv", index_col=0, parse_dates=True)
    prem = (np.exp(V["gap"].dropna()) - 1) * 100
    med = prem.expanding(min_periods=36).median()
    fig, ax = figure(w, h)
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot(prem.index, prem.values, color=RED, lw=2, label=f"Growth premium ({prem.index[-1]:%b %Y}: {pp(prem.iloc[-1], 0)}%)")
    ax.plot(med.index, med.values, color=GREEN, lw=2.2, ls=(0, (5, 3)), label=f"Median of history to date ({pp(med.iloc[-1], 0)}%)")
    ax.plot(prem.index[-1], prem.iloc[-1], "o", ms=10, color=RED, mec=BG, mew=2, zorder=3)
    date_axis(ax, (1,), "%Y"); ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.set_ylim(-15, 112)
    ax.set_ylabel("Forward P/E premium (%)")
    ax.legend(loc="upper right")
    title(fig, "Forward P/E premium of growth over defensives")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07)
    save(fig, "deck_pe_premium.png")
    return prem, med


# ---- Timing ------------------------------------------------------------------------------------------------
def def_minus_grw(w=900, h=330):
    r = pd.read_csv(RAW / "prices_monthly.csv", index_col=0, parse_dates=True).loc[:LAST_MONTH]
    D, G = r[DEF].mean(axis=1), r[["XLK", "XLY"]].mean(axis=1)
    win = slice("2026-06-01", LAST_MONTH)
    cd, cg = (1 + D.loc[win]).cumprod() - 1, (1 + G.loc[win]).cumprod() - 1
    s = pd.concat([pd.Series([0.0], index=[ASOF]), (cd - cg) * 100])
    fig, ax = figure(w, h)
    ax.axvspan(pd.Timestamp("2026-09-30"), LAST_MONTH, color=LIGHT, alpha=0.3, lw=0)
    ax.plot(s.index, s.values, color=GREEN, lw=2.6, marker="o", ms=9, mec=BG, mew=2)
    ax.axhline(0, color=INK, lw=1.2)
    for d, v in s.iloc[1:].items():
        ax.text(d, v + 1.3, pp(v), ha="center", va="bottom", fontsize=FS)
    ax.set_ylim(-3, s.max() + 7); ax.set_yticks([0, 10])
    ax.set_ylabel("pp")
    ax.set_xticks(s.index, [f"{d:%-d %b}" for d in s.index])
    title(fig, "Defensives minus growth since 29 May, cumulative (pp)")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12)
    save(fig, "deck_def_minus_grw.png")
    oct_ = (D - G).loc[LAST_MONTH] * 100
    return s, oct_


# ---- Trades ------------------------------------------------------------------------------------------------
def decomposition(w=960, h=708):
    T = pd.read_csv(OUT / "tournament_baseline.csv", index_col=0)
    d, g = T.loc["D"], T.loc["E"]
    assert abs((d["mean_pp_yr"] - g["mean_pp_yr"]) - T.loc["A", "mean_pp_yr"]) < 1e-3
    panels = [("Excess return\n(pp per year)", "mean_pp_yr", "t_mean"), ("Alpha\n(pp per year)", "alpha_pp_yr", "t_alpha"),
              ("Sell-off quarter\n(%)", "selloff_3m", "t_selloff")]
    fig, axes = figure(w, h, ncols=3)
    for ax, (lab, c, t) in zip(axes, panels):
        vals = [d[c], g[c]]
        ax.bar([0, 1], vals, width=0.6, color=[GREEN, RED], edgecolor=BG, lw=2)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_ylim(-3, 8.2)
        for x, v, tt in zip([0, 1], vals, [d[t], g[t]]):
            ax.text(x, v + (0.25 if v >= 0 else -0.25), f"{pp(v)}\n(t {pp(tt)})", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=FS)
        ax.set_xticks([0, 1], ["Def.", "Growth"])
        ax.set_title(lab, fontsize=FS, pad=8)
        ax.tick_params(axis="x", length=0)
    title(fig, "Defensives and growth vs the market, 1999 – 2026")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, wspace=0.4)
    save(fig, "deck_decomposition.png", extra=70)
    return d, g


LABEL_POS = {"XLC": (-1.5, 0.25, "right"), "XLF": (-1.5, 0.0, "right"), "XLI": (0, -0.55, "center"),
             "XLV": (1.8, 0.0, "left")}


def valuation_revisions(w=700, h=708):
    T = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0).drop(index="SPY", errors="ignore")
    col = {t: (GREEN if t in DEF else RED if t in ("XLK", "XLY") else GREY) for t in T.index}
    fig, ax = figure(w, h)
    ax.axvline(50, color=INK, lw=1, ls=(0, (3, 3))); ax.axhline(0, color=INK, lw=1.2)
    for t, r in T.iterrows():
        ax.plot(r["pe_pct"], r["rev3"], "o", ms=13 if t in DEF else 11, color=col[t], mec=BG, mew=2, zorder=3)
        dx, dy, ha = LABEL_POS.get(t, (1.8, 0.0, "left"))
        ax.text(r["pe_pct"] + dx, r["rev3"] + dy, t, fontsize=FS, ha=ha, va="center", **(BOLD if t in DEF else {}))
    ax.set_xlim(0, 105); ax.set_ylim(-1.2, 18.5)
    ax.set_xlabel("Forward P/E percentile since 2006")
    ax.set_ylabel("3-month EPS revision (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=c) for c in (GREEN, RED, GREY)]
    ax.legend(handles, ["Defensives", "Growth", "Other"], loc="upper left")
    title(fig, "Valuation vs earnings revisions")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.1)
    save(fig, "deck_valuation_revisions.png")
    return T


def oil_collapse_loss(w=900, h=490):
    S = pd.read_csv(OUT / "energy_stress.csv", index_col=0)
    glut = S.loc[[i for i in S.index if i.startswith("Oil glut")][0], "XLE minus SPY (%)"] / 100
    M, A = team_metrics(), pd.read_csv(OUT / "team_weights_active.csv", index_col=0)
    marks = [("Current", CUR, GREY, (-0.12, -0.12, "right", "top")),
             ("Model Base", "Model Base", GREY, (-0.15, -0.1, "right", "top")),
             ("Full hedge", "New, full hedge", GREY, (-0.15, -0.12, "right", "top")),
             ("Proposed", NEW, GREEN, (0.15, 0.1, "left", "bottom"))]
    marks = [(lab, A.loc["XLE", k], M.loc[k, "oil_loss"], c, pos) for lab, k, c, pos in marks]
    x = np.linspace(0, 6, 100)
    fig, ax = figure(w, h)
    ax.axhspan(-1, 0, color=LIGHT, alpha=0.25, lw=0)
    ax.plot(x, glut * x, color=INK, lw=2)
    ax.axhline(-1, color=RED, lw=2, ls=(0, (5, 3)))
    ax.text(5.95, -0.9, "Limit −1pp", color=INK, fontsize=FS, ha="right", va="bottom", **BOLD)
    for lab, wv, y, c, (dx, dy, ha, va) in marks:
        ax.plot(wv, y, "o", ms=11, color=c, mec=INK if c == GREY else BG, mew=1.5 if c == GREY else 2, zorder=3)
        ax.text(wv + dx, y + dy, lab, fontsize=FS, ha=ha, va=va, **(BOLD if lab == "Proposed" else {}))
    ax.set_xlim(0, 6); ax.set_ylim(-2.6, 0.3)
    ax.set_xlabel("XLE active weight (pp)"); ax.set_ylabel("Loss (pp)")
    title(fig, "Loss in a 2014–16 oil collapse, by XLE weight")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15)
    save(fig, "deck_oil_collapse_loss.png")
    return glut, [(m[0], m[1], m[2], m[3]) for m in marks]


# ---- Risks -------------------------------------------------------------------------------------------------
def risk_metrics(w=860, h=460):
    M = team_metrics()
    worst = lambda k: -M.loc[k, [c for c in M.columns if c.startswith("scen_")]].min()
    rows = [("Tracking error (%)", M.loc[CUR, "te_bench"], M.loc[NEW, "te_bench"]),
            ("3-month CVaR (%)", M.loc[CUR, "cvar3"], M.loc[NEW, "cvar3"]),
            ("Oil-collapse loss (pp)", -M.loc[CUR, "oil_loss"], -M.loc[NEW, "oil_loss"]),
            ("Worst scenario (%)", worst(CUR), worst(NEW))]
    fig, ax = figure(w, h)
    y = np.arange(len(rows))[::-1]; hh = 0.38
    cur = [r[1] for r in rows]; new = [r[2] for r in rows]
    ax.barh(y + hh / 2, cur, height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, new, height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    for yy, vals in ((y + hh / 2, cur), (y - hh / 2, new)):
        for yi, v in zip(yy, vals):
            ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=FS)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 4.2); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False); bars_frame(ax)
    ax.legend(loc="lower right")
    title(fig, "Risk measures (losses as positive numbers)")
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.03)
    save(fig, "deck_risk_metrics.png")
    return rows


def scenario_returns(w=700, h=708):
    M = team_metrics()
    P = M[[c for c in M.columns if c.startswith("scen_")]].T.rename(index=lambda c: c[5:])[[CUR, NEW]]
    names = {"Stagflationary sell-off": "Stagflation\nsell-off", "Real-yield squeeze": "Real-yield\nsqueeze",
             "Growth scare": "Growth\nscare", "Inflation scare": "Inflation\nscare", "Easing rally": "Easing\nrally",
             "Steady state": "Steady\nstate"}
    fig, ax = figure(w, h)
    y = np.arange(len(P))[::-1]; hh = 0.38
    ax.barh(y + hh / 2, P.iloc[:, 0], height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, P.iloc[:, 1], height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    for yy, c in ((y + hh / 2, 0), (y - hh / 2, 1)):
        hbar_labels(ax, yy, P.iloc[:, c].values, fmt=lambda v: pp(v, 2), pad=0.04)
    ax.set_yticks(y, [names.get(s, s) for s in P.index]); ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-1.2, 1.85); ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Active return vs S&P 500, 3 months (%)")
    bars_frame(ax); ax.legend(loc="lower right")
    title(fig, "Average 3-month return by scenario")
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.1)
    save(fig, "deck_scenario_returns.png")
    return P


def fixed_income_mix(w=700, h=440):
    from src.team_weights import TEAM
    T = pd.DataFrame({k: TEAM.loc[v].sum() for k, v in FI_TYPES.items()}).T[["old", "new"]]
    fig, ax = figure(w, h)
    y = np.arange(len(T))[::-1]; hh = 0.38
    ax.barh(y + hh / 2, T["old"], height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, T["new"], height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    for yy, c in ((y + hh / 2, "old"), (y - hh / 2, "new")):
        for yi, v in zip(yy, T[c]):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=FS)
    ax.set_yticks(y, ["Treasury\nand agency", "TIPS", "IG\ncorporate", "Cash"])
    ax.set_xlim(0, 27); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False); bars_frame(ax)
    ax.legend(loc="lower right")
    title(fig, "Fixed income by type (% of portfolio)")
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.03)
    save(fig, "deck_fixed_income_mix.png")
    return T


def trigger_readings(w=700, h=708):
    wt = series("DCOILWTICO")
    be = series("T10YIE"); be_ref = be.loc[:DECIDE].iloc[-1]
    w26 = wt.loc["2026"]; decline = (1 - w26.iloc[-1] / w26.max()) * 100
    cs = pd.read_csv(OUT / "cycle_state.csv", index_col=0, parse_dates=True)["p_recession_12m"].dropna()
    vs = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0)
    rows = [(f"10-year breakeven: {be_ref:.2f}%", be_ref, be_ref + 0.25, (1.9, 2.7), 1, "%"),
            (f"WTI below 2026 peak: {decline:.0f}%", decline, 25, (0, 40), 0, "%"),
            (f"Recession probability: {cs.iloc[-1] * 100:.0f}%", cs.iloc[-1] * 100, 35, (0, 50), 0, "%"),
            (f"Tech EPS revisions: {pp(vs.loc['XLK', 'rev3'])}%", vs.loc["XLK", "rev3"], 15, (0, 25), 0, "%")]
    fig, axes = figure(w, h, nrows=4)
    for ax, (name, v, thr, (lo, hi), dec, unit) in zip(axes, rows):
        met = v >= thr
        ax.barh(0, hi - lo, left=lo, height=0.55, color=TRACK, lw=0)
        ax.barh(0, min(v, thr) - lo, left=lo, height=0.55, color=LIGHT, alpha=0.8, lw=0)
        ax.plot([thr, thr], [-0.45, 0.45], color=INK, lw=3)
        ax.plot(v, 0, "o", ms=13, color=RED if met else GREEN, mec=BG, mew=2, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _, d=dec: f"{x:.{d}f}{unit}"))
        ax.set_title(f"{name} · {'met' if met else 'not triggered'}", fontsize=FS, loc="left", pad=6, **BOLD)
    title(fig, "Readings (dot) vs thresholds (line)")
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.05, hspace=1.25)
    save(fig, "deck_trigger_readings.png", extra=40)
    return rows


def main():
    print("\nDeck visuals from frozen data (freeze-2026-11-04), team style\n")
    weights_active()
    M, C, groups = performance()
    active_current(); cpi_components()
    ip, rs = activity(); p = payrolls(); fed_funds()
    ff, t2, gap = fed_expected(); last, s1, s2 = curve(); ry = real_yields()
    w = wti(); prem, med = pe_premium(); s, oct_ = def_minus_grw()
    decomposition(); valuation_revisions(); glut, marks = oil_collapse_loss()
    rows = risk_metrics(); scenario_returns(); trigger_readings(); fixed_income_mix()
    print("\nNumbers drawn (check against the deck):")
    print("  Performance Jun–Oct (%): " + ", ".join(f"{k} {v:+.2f}" for k, v in C.iloc[-1].items()))
    print("  Active return by group vs sector benchmark (pp): " + ", ".join(f"{k} {v:+.2f}" for k, v in groups.items()))
    print(f"  Industrial production y/y {ip:+.2f}%, real retail sales y/y {rs:+.2f}%")
    print(f"  Payrolls: last {p.iloc[-1]:+.0f}k, 2026 average {p.loc['2026'].mean():+.0f}k")
    print(f"  Fed funds {ff:.2f}%, 2-year {t2:.2f}% ({gap:+.0f}bp); 10y {last.t10:.2f}%, 10y-2y {s1:+.0f}bp, 10y-ff {s2:+.0f}bp")
    print(f"  10y nominal {ry.n:.2f}%, real {ry.r:.2f}%, breakeven {ry.b:.2f}%; WTI ${w.iloc[-1]:.2f}")
    print(f"  P/E premium {prem.iloc[-1]:+.1f}% vs median {med.iloc[-1]:+.1f}%; DEF-GRW Oct {oct_:+.2f}%, since 29 May {s.iloc[-1]:+.2f}pp")
    print("  Oil-collapse loss: " + ", ".join(f"{l} {wv:+.2f}pp -> {y:+.2f}pp" for l, wv, y, _ in marks))
    print("  Risk measures (current/proposed): " + "; ".join(f"{r[0]} {r[1]:.2f}/{r[2]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
