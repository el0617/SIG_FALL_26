"""
Deck visuals for the Mean Green Fund / TOPDOWN2 final presentation (13 Nov 2026), team style.

    python -m src.deck_visuals

Draws every chart of the final deck from the FROZEN data (branch freeze-2026-11-04) into outputs/deck/.
No downloads and no re-estimation: numbers come from data/raw caches and outputs/*.csv of the freeze run.
The only hard-coded inputs are the BLS September 2026 CPI component rates (CPI_SEP26 below), which are
not cached; their source is noted next to them.

Style: the team's template (white background, Inter, palette dark green #13501B, red #A82B0B,
teal #104862, light green #8ED974, black). Each figure is sized to its box on the 1920x1080 slide canvas
(pixels / 100 = inches) and saved at dpi 200 (2x).

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
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"
DECK = OUT / "deck"
DECIDE = pd.Timestamp("2026-11-04")
ASOF = pd.Timestamp("2026-05-29")
LAST_MONTH = pd.Timestamp("2026-10-31")

# BLS CPI, September 2026, 12-month change (%), released 14 Oct 2026 (as reported by Securities.io / Schwab)
CPI_SEP26 = {"Gasoline": 34.2, "Energy": 20.4, "All items": 3.7, "Shelter": 3.2, "Core (ex food and energy)": 2.6}

# ---- team style -------------------------------------------------------------------------------------
BG, INK, INK2, MUTED, RULE = "#FFFFFF", "#000000", "#333333", "#595959", "#D9D9D9"
GREEN, RED, TEAL, LIGHT, LIME, GREY = "#13501B", "#A82B0B", "#104862", "#8ED974", "#5F9600", "#A6ADA7"
DEF, INF, GRW = ["XLP", "XLU", "XLV"], ["XLE", "XLB"], ["VGT", "XLY"]
GROUP_COLOR = {**{t: GREEN for t in DEF}, **{t: RED for t in GRW}, **{t: TEAL for t in INF}}
ORDER = ["VGT", "XLY", "XLC", "XLF", "XLI", "SCHH", "XLB", "XLE", "XLP", "XLV", "XLU"]

for p in glob.glob(os.path.expanduser("~/.fonts/*.ttf")) + glob.glob(str(ROOT / "assets" / "fonts" / "*.ttf")):
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
_names = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ("Inter", "Aptos", "Arial", "DejaVu Sans") if f in _names), "DejaVu Sans")
BOLD = {"fontweight": "bold"}

plt.rcParams.update({
    "font.family": FONT, "font.size": 13, "axes.facecolor": BG, "figure.facecolor": BG,
    "savefig.facecolor": BG, "axes.edgecolor": "#8C8C8C", "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": RULE, "grid.linewidth": 0.8,
    "legend.frameon": False, "axes.unicode_minus": True,
})


def figure(w_px, h_px, **kw):
    return plt.subplots(figsize=(w_px / 100, h_px / 100), **kw)


def title(fig, text, sub=None):
    """Centred bold title 10px from the top, grey subtitle under it (team chart style)."""
    h = fig.get_figheight() * 100
    fig.text(0.5, 1 - 10 / h, text, ha="center", va="top", fontsize=15, color=INK, **BOLD)
    if sub:
        fig.text(0.5, 1 - 36 / h, sub, ha="center", va="top", fontsize=12, color=MUTED)


def top_for(fig, sub=True, extra=0):
    return 1 - ((62 if sub else 40) + 16 + extra) / (fig.get_figheight() * 100)


def save(fig, name, top):
    fig.subplots_adjust(top=top)
    DECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK / name, dpi=200)
    plt.close(fig)
    print(f"  saved outputs/deck/{name}")


def pp(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def series(name):
    return pd.read_csv(RAW / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()


def active_weights():
    W = pd.read_csv(OUT / "weighting_weights.csv", index_col=0)
    return W[["Current", "Base"]].sub(W["benchmark"], axis=0)


def hbar_labels(ax, y, vals, fmt=lambda v: pp(v), pad=0.25, size=12):
    for yi, v in zip(y, vals):
        ax.text(v + (pad if v >= 0 else -pad), yi, fmt(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=size, color=INK2)


def end_label(ax, x, y, text, color, dy=0, **kw):
    ax.text(x + pd.Timedelta(days=12), y + dy, text, color=color, fontsize=12, va="center", ha="left", **kw)


def tidy(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True)
    ax.set_axisbelow(True)


# ---- Proposal: active weights, current vs proposed ----------------------------------------------------
def weights_active(w=820, h=708, name="deck_weights_active.png"):
    A = active_weights().reindex(ORDER)
    fig, ax = figure(w, h)
    y = np.arange(len(A))[::-1]; hh = 0.38
    ax.barh(y + hh / 2, A["Current"], height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, A["Base"], height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    hbar_labels(ax, y - hh / 2, A["Base"].values, size=11, pad=0.2)
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(A.values.min() - 2.4, A.values.max() + 2.4)
    ax.set_xlabel("Active weight vs IVV (pp)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", fontsize=12)
    title(fig, "Active weights: current vs proposed", "Percentage points vs IVV; labels show the proposed weight")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.09)
    save(fig, name, top_for(fig))
    return A


# ---- Present: performance since 29 May ------------------------------------------------------------------
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
    M = pd.DataFrame(rows, columns=["date", "TOPDOWN2", "IVV sector benchmark", "SPY"]).set_index("date")
    C = (1 + M / 100).cumprod() * 100 - 100
    C.loc[ASOF] = 0.0; C = C.sort_index()
    fig, ax = figure(1100, 708)
    styles = {"TOPDOWN2": (GREEN, 2.6), "IVV sector benchmark": (TEAL, 2.0), "SPY": (GREY, 2.0)}
    for col, (c, lw) in styles.items():
        ax.plot(C.index, C[col], color=c, lw=lw, marker="o", ms=7, mec=BG, mew=1.5, label=col)
        end_label(ax, C.index[-1], C[col].iloc[-1], f"{col}  {pp(C[col].iloc[-1])}%", c, **(BOLD if col == "TOPDOWN2" else {}))
    ax.axhline(0, color=INK, lw=1)
    ax.axvspan(pd.Timestamp("2026-09-30"), LAST_MONTH, color=LIGHT, alpha=0.18, lw=0)
    ax.set_xticks(C.index, [f"{d:%-d %b}" for d in C.index])
    ax.set_xlim(C.index[0] - pd.Timedelta(days=6), C.index[-1] + pd.Timedelta(days=70))
    ax.set_ylabel("Cumulative total return (%)"); tidy(ax)
    title(fig, "Cumulative return since positioning on 29 May 2026",
          "Monthly total returns; 29 May holdings drifted with returns; benchmark = IVV sector weights")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08)
    save(fig, "deck_performance.png", top_for(fig))
    groups = {g: contrib[t].sum() for g, t in GROUPS.items()}
    return M, C, groups


# ---- Present: current active weights ---------------------------------------------------------------------
def active_current():
    A = active_weights()["Current"].reindex(ORDER)
    fig, ax = figure(1100, 708)
    y = np.arange(len(A))[::-1]
    ax.barh(y, A.values, height=0.62, color=[GROUP_COLOR.get(t, GREY) for t in A.index], edgecolor=BG, lw=2)
    hbar_labels(ax, y, A.values)
    ax.set_yticks(y, A.index); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(A.min() - 2.5, A.max() + 2.5)
    ax.set_xlabel("Active weight vs IVV sector benchmark (pp)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RED, GREY, TEAL, GREEN)]
    ax.legend(handles, ["Growth", "Other", "Inflation hedge", "Defensives"], loc="upper right", fontsize=12)
    title(fig, "Current active weights by ETF", "Drifted from 29 May through 31 Oct; percentage points vs IVV")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09)
    save(fig, "deck_active_current.png", top_for(fig))


# ---- Macro: inflation --------------------------------------------------------------------------------------
def cpi_components():
    S = pd.Series(CPI_SEP26)
    colors = [RED if k in ("Gasoline", "Energy") else (INK if k == "All items" else GREEN) for k in S.index]
    fig, ax = figure(1060, 708)
    y = np.arange(len(S))[::-1]
    ax.barh(y, S.values, height=0.6, color=colors, edgecolor=BG, lw=2)
    hbar_labels(ax, y, S.values, fmt=lambda v: f"{v:.1f}%", pad=0.5, size=13)
    ax.set_yticks(y, S.index, fontsize=13); ax.axvline(0, color=INK, lw=1)
    ax.axvline(2, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.3, y[0] + 0.45, "Fed target 2%", color=MUTED, fontsize=11, va="bottom")
    ax.set_xlim(0, 40); ax.set_xlabel("Change over 12 months (%)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    title(fig, "Consumer prices by component, September 2026", "12-month change; BLS CPI-U, released 14 Oct 2026")
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.09)
    save(fig, "deck_cpi_components.png", top_for(fig))


# ---- Macro: growth -----------------------------------------------------------------------------------------
def activity():
    ip, rs = series("INDPRO"), series("RRSFS")
    a = ((ip / ip.shift(12) - 1) * 100).loc["2024-01":]
    b = ((rs / rs.shift(12) - 1) * 100).loc["2024-01":]
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1)
    ax.plot(a.index, a.values, color=GREEN, lw=2.4)
    ax.plot(b.index, b.values, color=RED, lw=2.4)
    end_label(ax, a.index[-1], a.iloc[-1], f"Industrial production {pp(a.iloc[-1])}%", GREEN, dy=0.35, **BOLD)
    end_label(ax, b.index[-1], b.iloc[-1], f"Real retail sales {pp(b.iloc[-1])}%", RED, dy=-0.35, **BOLD)
    ax.set_xlim(a.index[0], a.index[-1] + pd.Timedelta(days=290))
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Change over 12 months (%)"); tidy(ax)
    title(fig, "Industrial production and real retail sales", f"Monthly, Jan 2024 – {a.index[-1]:%b %Y}; Federal Reserve, Census (FRED)")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08)
    save(fig, "deck_activity.png", top_for(fig))
    return a.iloc[-1], b.iloc[-1]


# ---- Macro: labor market -----------------------------------------------------------------------------------
def payrolls():
    p = series("PAYEMS").diff().loc["2025-01":]
    avg3 = p.rolling(3).mean()
    fig, ax = figure(1060, 708)
    ax.bar(p.index, p.values, width=22, color=[GREEN if v >= 0 else RED for v in p.values], edgecolor=BG, lw=1)
    ax.plot(avg3.index, avg3.values, color=INK, lw=2, label="3-month average")
    ax.axhline(0, color=INK, lw=1)
    ax.text(p.index[-1], p.iloc[-1] + 32, f"{p.iloc[-1]:+.0f}k", ha="center", va="bottom", fontsize=12, color=INK, **BOLD)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y")); ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.set_ylabel("Monthly change in nonfarm payrolls (thousands)"); tidy(ax)
    ax.legend(loc="upper left", fontsize=12)
    title(fig, "Nonfarm payroll growth", f"Monthly change, Jan 2025 – {p.index[-1]:%b %Y}; BLS (FRED)")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11)
    save(fig, "deck_payrolls.png", top_for(fig))
    return p


# ---- Macro: Federal Reserve -------------------------------------------------------------------------------
def fed_funds():
    f = series("DFF").loc["2022-01-01":DECIDE]
    fig, ax = figure(1060, 708)
    ax.plot(f.index, f.values, color=GREEN, lw=2.4, drawstyle="steps-post")
    notes = [("2022-07-15", 5.5, "Mar 2022 – Jul 2023\n+525bp"), ("2024-09-25", 5.7, "Sep – Dec 2024\n−100bp"),
             ("2025-09-25", 4.75, "Sep – Dec 2025\n−75bp"), ("2026-07-01", 4.55, "Sep 2026\n+25bp")]
    for d, yv, t in notes:
        ax.text(pd.Timestamp(d), yv, t, fontsize=12, color=INK, ha="center", va="bottom", **BOLD)
    ax.plot(f.index[-1], f.iloc[-1], "o", ms=9, color=RED, mec=BG, mew=2, zorder=3)
    end_label(ax, f.index[-1], f.iloc[-1] - 0.35, f"{f.iloc[-1]:.2f}%", RED, **BOLD)
    ax.set_ylim(0, 6.6); ax.set_xlim(f.index[0], f.index[-1] + pd.Timedelta(days=120))
    ax.set_ylabel("Effective federal funds rate (%)"); tidy(ax)
    title(fig, "Federal funds rate, 2022 – 2026", f"Daily effective rate to {f.index[-1]:%-d %b %Y}; Federal Reserve (FRED)")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08)
    save(fig, "deck_fed_funds.png", top_for(fig))
    return f


def fed_expected():
    f, t2 = series("DFF").loc["2025-01-01":DECIDE], series("DGS2").loc["2025-01-01":DECIDE]
    fig, ax = figure(1000, 708)
    ax.plot(f.index, f.values, color=GREEN, lw=2.4, drawstyle="steps-post")
    ax.plot(t2.index, t2.values, color=RED, lw=2)
    gap = (t2.iloc[-1] - f.iloc[-1]) * 100
    end_label(ax, t2.index[-1], t2.iloc[-1] + 0.08, f"2-year Treasury {t2.iloc[-1]:.2f}%", RED, **BOLD)
    end_label(ax, f.index[-1], f.iloc[-1] - 0.08, f"Fed funds {f.iloc[-1]:.2f}%", GREEN, **BOLD)
    ax.annotate("", xy=(t2.index[-1], t2.iloc[-1]), xytext=(t2.index[-1], f.iloc[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(t2.index[-1] - pd.Timedelta(days=8), (t2.iloc[-1] + f.iloc[-1]) / 2, f"{gap:+.0f}bp", ha="right",
            va="center", fontsize=12, color=INK, **BOLD)
    ax.set_xlim(f.index[0], f.index[-1] + pd.Timedelta(days=170))
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Percent"); tidy(ax)
    title(fig, "Fed funds rate and the 2-year Treasury yield", f"Daily, Jan 2025 – {t2.index[-1]:%-d %b %Y}; Federal Reserve (FRED)")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08)
    save(fig, "deck_fed_expected.png", top_for(fig))
    return f.iloc[-1], t2.iloc[-1], gap


# ---- Macro: yield curve and real yields --------------------------------------------------------------------
def curve():
    t10, t2, f = series("DGS10"), series("DGS2"), series("DFF")
    df = pd.concat([t10, t2, f], axis=1, keys=["t10", "t2", "ff"]).dropna().loc["2019-01-01":DECIDE]
    s1, s2 = (df.t10 - df.t2) * 100, (df.t10 - df.ff) * 100
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1)
    ax.plot(s1.index, s1.values, color=GREEN, lw=1.8)
    ax.plot(s2.index, s2.values, color=RED, lw=1.8)
    end_label(ax, s1.index[-1], s1.iloc[-1] - 18, f"10-year − 2-year\n{s1.iloc[-1]:+.0f}bp", GREEN, **BOLD)
    end_label(ax, s2.index[-1], s2.iloc[-1] + 22, f"10-year − fed funds\n{s2.iloc[-1]:+.0f}bp", RED, **BOLD)
    ax.set_xlim(s1.index[0], s1.index[-1] + pd.Timedelta(days=640))
    ax.set_ylabel("Spread (basis points)"); tidy(ax)
    title(fig, "Yield curve spreads, 2019 – 2026", f"Daily to {s1.index[-1]:%-d %b %Y}; Treasury, Federal Reserve (FRED)")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08)
    save(fig, "deck_curve.png", top_for(fig))
    return df.iloc[-1], s1.iloc[-1], s2.iloc[-1]


def real_yields():
    n, r, b = series("DGS10"), series("DFII10"), series("T10YIE")
    df = pd.concat([n, r, b], axis=1, keys=["n", "r", "b"]).dropna().loc["2021-01-01":DECIDE]
    fig, ax = figure(1060, 708)
    ax.axhline(0, color=INK, lw=1)
    for col, c, lab, dy in (("n", INK, "10-year nominal", 0.12), ("r", GREEN, "10-year real (TIPS)", 0.0),
                            ("b", RED, "10-year breakeven", -0.12)):
        ax.plot(df.index, df[col], color=c, lw=2)
        end_label(ax, df.index[-1], df[col].iloc[-1] + dy, f"{lab} {df[col].iloc[-1]:.2f}%", c, **BOLD)
    ax.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(days=520))
    ax.set_ylabel("Percent"); tidy(ax)
    title(fig, "10-year nominal yield = real yield + breakeven inflation",
          f"Daily, Jan 2021 – {df.index[-1]:%-d %b %Y}; Treasury, Federal Reserve (FRED)")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08)
    save(fig, "deck_real_yields.png", top_for(fig))
    return df.iloc[-1]


# ---- War and AI ---------------------------------------------------------------------------------------------
EVENTS = [("2026-02-28", "War begins", "left", 0), ("2026-04-08", "Ceasefire", "left", 1),
          ("2026-06-17", "Islamabad memorandum", "right", 0), ("2026-07-08", "Memorandum collapses", "left", 1),
          ("2026-10-25", "Qatar draft agreement", "right", 0)]


def wti(w=1060, h=300):
    s = series("DCOILWTICO").loc["2026-01-01":DECIDE]
    fig, ax = figure(w, h)
    ax.plot(s.index, s.values, color=INK, lw=2)
    hi = s.max()
    ax.set_ylim(s.min() - 6, hi + 26)
    for d, lab, ha, lvl in EVENTS:
        d = pd.Timestamp(d)
        ax.axvline(d, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(d, hi + 25 - 11 * lvl, f" {lab} " if ha == "left" else f"{lab} ", fontsize=11, color=INK2, va="top", ha=ha)
    ax.plot(s.index[-1], s.iloc[-1], "o", ms=8, color=RED, mec=BG, mew=2)
    ax.text(s.index[-1], s.iloc[-1] - 6, f"${s.iloc[-1]:.0f}", fontsize=12, color=INK, ha="right", va="top")
    pk = pd.Timestamp(s.idxmax())
    ax.text(mdates.date2num(pk) - 3, hi + 1.0, f"Peak ${hi:.0f} ", fontsize=11, color=INK2, ha="right", va="bottom")
    ax.set_ylabel("$ per barrel"); tidy(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    title(fig, f"WTI crude oil, daily, 2 Jan – {s.index[-1]:%-d %b} 2026")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12)
    save(fig, "deck_wti_2026.png", top_for(fig, sub=False))
    return s


def pe_premium(w=1000, h=708):
    V = pd.read_csv(OUT / "valuation_signals.csv", index_col=0, parse_dates=True)
    prem = (np.exp(V["gap"].dropna()) - 1) * 100
    med = prem.expanding(min_periods=36).median()
    fig, ax = figure(w, h)
    ax.axhline(0, color=INK, lw=1)
    ax.plot(prem.index, prem.values, color=RED, lw=2, label="Growth premium over defensives")
    ax.plot(med.index, med.values, color=GREEN, lw=1.8, ls=(0, (5, 3)), label="Median of history to date")
    ax.plot(prem.index[-1], prem.iloc[-1], "o", ms=9, color=RED, mec=BG, mew=2, zorder=3)
    end = prem.index[-1] + pd.Timedelta(days=75)
    ax.text(end, prem.iloc[-1] + 1.5, f"{prem.index[-1]:%b %Y}\n{pp(prem.iloc[-1], 0)}%", ha="left", va="bottom", fontsize=12, color=INK, **BOLD)
    ax.text(end, med.iloc[-1] - 1.5, f"Median\n{pp(med.iloc[-1], 0)}%", ha="left", va="top", fontsize=12, color=GREEN)
    ax.set_xlim(prem.index[0] - pd.Timedelta(days=120), prem.index[-1] + pd.Timedelta(days=900))
    ax.set_ylabel("Forward P/E premium (%)"); tidy(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1.0), fontsize=12)
    title(fig, "Forward P/E premium of growth over defensives",
          f"XLK and XLY vs XLP, XLU and XLV; monthly {prem.index[0]:%b %Y} – {prem.index[-1]:%b %Y}; Bloomberg")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.07)
    save(fig, "deck_pe_premium.png", top_for(fig))
    return prem, med


# ---- Timing -----------------------------------------------------------------------------------------------
def def_minus_grw(w=900, h=300):
    r = pd.read_csv(RAW / "prices_monthly.csv", index_col=0, parse_dates=True).loc[:LAST_MONTH]
    D, G = r[DEF].mean(axis=1), r[["XLK", "XLY"]].mean(axis=1)
    win = slice("2026-06-01", LAST_MONTH)
    cd, cg = (1 + D.loc[win]).cumprod() - 1, (1 + G.loc[win]).cumprod() - 1
    s = pd.concat([pd.Series([0.0], index=[ASOF]), (cd - cg) * 100])
    fig, ax = figure(w, h)
    ax.axvspan(pd.Timestamp("2026-09-30"), LAST_MONTH, color=LIGHT, alpha=0.25, lw=0)
    ax.plot(s.index, s.values, color=GREEN, lw=2.4, marker="o", ms=8, mec=BG, mew=2)
    ax.axhline(0, color=INK, lw=1)
    for d, v in s.iloc[1:].items():
        ax.text(d, v + 1.2, pp(v), ha="center", va="bottom", fontsize=12, color=INK2)
    oct_ = (D - G).loc[LAST_MONTH] * 100
    ax.text(pd.Timestamp("2026-10-15"), 1.0, f"October\n{pp(oct_)}%", ha="center", va="bottom", fontsize=11, color=GREEN, **BOLD)
    ax.set_ylim(min(-2, s.min() - 2), s.max() + 5)
    ax.set_ylabel("Cumulative (pp)"); tidy(ax)
    ax.set_xticks(s.index, [f"{d:%-d %b}" for d in s.index])
    title(fig, "Defensives minus growth since positioning on 29 May",
          "Cumulative return, XLP/XLU/XLV minus XLK/XLY, equal-weighted; month-end")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12)
    save(fig, "deck_def_minus_grw.png", top_for(fig))
    return s, oct_


# ---- Trades ------------------------------------------------------------------------------------------------
def decomposition(w=960, h=708):
    T = pd.read_csv(OUT / "tournament_baseline.csv", index_col=0)
    d, g = T.loc["D"], T.loc["E"]
    assert abs((d["mean_pp_yr"] - g["mean_pp_yr"]) - T.loc["A", "mean_pp_yr"]) < 1e-3
    panels = [("Average excess return\n(pp per year)", "mean_pp_yr", "t_mean"),
              ("Alpha beyond beta\n(pp per year)", "alpha_pp_yr", "t_alpha"),
              ("Average sell-off quarter\n(%, SPY down ≥ 5%)", "selloff_3m", "t_selloff")]
    fig, axes = figure(w, h, ncols=3)
    for ax, (lab, c, t) in zip(axes, panels):
        vals = [d[c], g[c]]
        ax.bar([0, 1], vals, width=0.6, color=[GREEN, RED], edgecolor=BG, lw=2)
        ax.axhline(0, color=INK, lw=1)
        lim = max(6.5, max(abs(v) for v in vals) * 1.35)
        ax.set_ylim(-lim * 0.45, lim)
        for x, v, tt in zip([0, 1], vals, [d[t], g[t]]):
            ax.text(x, v + (0.25 if v >= 0 else -0.25), f"{pp(v)}\n(t {pp(tt)})", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=12, color=INK2)
        ax.set_xticks([0, 1], ["Defensives\nvs market", "Growth\nvs market"], fontsize=12)
        ax.set_title(lab, fontsize=13, color=INK, pad=10)
        tidy(ax); ax.tick_params(axis="x", length=0)
    title(fig, "Defensives − growth = (defensives − market) − (growth − market)",
          f"Monthly data {pd.Timestamp(d['start']):%b %Y} – {pd.Timestamp(d['end']):%b %Y}; thesis tournament baseline")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, wspace=0.35)
    save(fig, "deck_decomposition.png", top_for(fig, extra=62))
    return d, g


def valuation_revisions(w=700, h=708):
    T = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0).drop(index="SPY", errors="ignore")
    col = {t: (GREEN if t in DEF else RED if t in ("XLK", "XLY") else GREY) for t in T.index}
    fig, ax = figure(w, h)
    ax.axvline(50, color=RULE, lw=1.2); ax.axhline(0, color=INK, lw=1)
    for t, r in T.iterrows():
        ax.plot(r["pe_pct"], r["rev3"], "o", ms=12 if t in DEF else 10, color=col[t], mec=BG, mew=2, zorder=3)
        lab = f"{t}  {r['fwd_pe']:.1f}x" if t in DEF else t
        if t == "XLV":
            ax.text(104, r["rev3"] - 0.55, lab, fontsize=12, color=INK, va="top", ha="right", **BOLD)
            continue
        left = r["pe_pct"] > 85
        ax.text(r["pe_pct"] + (-2 if left else 2), r["rev3"] + 0.35, lab, fontsize=12, color=INK, va="bottom",
                ha="right" if left else "left", **(BOLD if t in DEF else {}))
    ax.set_xlim(0, 105); ax.set_ylim(-1, max(18, T["rev3"].max() + 2))
    ax.set_xlabel("Forward P/E percentile vs own history since 2006")
    ax.set_ylabel("3-month change in forward EPS (%)"); tidy(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=c) for c in (GREEN, RED, GREY)]
    ax.legend(handles, ["Defensives (forward P/E)", "Growth", "Other"], loc="upper left", fontsize=11.5)
    title(fig, "Valuation and earnings revisions by sector", "Bloomberg estimates as of 30 Sep 2026")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.09)
    save(fig, "deck_valuation_revisions.png", top_for(fig))
    return T


def oil_collapse_loss(w=900, h=490):
    S = pd.read_csv(OUT / "energy_stress.csv", index_col=0)
    glut = S.loc[[i for i in S.index if i.startswith("Oil glut")][0], "XLE minus SPY (%)"] / 100
    C = pd.read_csv(OUT / "consistency_summary.csv", index_col=0)
    marks = [("Current", "Current (drifted)", GREY), ("Team proposal", "Proposal", GREY),
             ("Full hedge", "Base (full hedge, post hoc)", GREY), ("Recommended", "Base (half hedge)", GREEN)]
    marks = [(lab, C.loc[k, "xle_active"], C.loc[k, "oil_crash_tail_pp"], c) for lab, k, c in marks]
    x = np.linspace(0, 7.5, 100)
    fig, ax = figure(w, h)
    ax.axhspan(-1, 0, color=LIGHT, alpha=0.2, lw=0)
    ax.plot(x, glut * x, color=INK, lw=2)
    ax.axhline(-1, color=RED, lw=1.8, ls=(0, (5, 3)))
    ax.text(7.45, -0.93, "Limit: −1pp", color=RED, fontsize=12, ha="right", va="bottom", **BOLD)
    offs = {"Current": (-0.2, -0.22, "right"), "Team proposal": (0.15, 0.12, "left"),
            "Full hedge": (-0.15, -0.28, "right"), "Recommended": (0.18, 0.12, "left")}
    for lab, wv, y, c in marks:
        ax.plot(wv, y, "o", ms=10, color=c, mec=INK if c == GREY else BG, mew=1.5 if c == GREY else 2, zorder=3)
        dx, dy, ha = offs[lab]
        ax.text(wv + dx, y + dy, f"{lab}\n{pp(wv)}pp → {pp(y, 2)}pp", fontsize=11.5, color=INK, ha=ha,
                va="top" if dy < 0 else "bottom", **(BOLD if lab == "Recommended" else {}))
    ax.set_xlim(0, 7.5); ax.set_ylim(-3.75, 0.6)
    ax.set_xlabel("XLE active weight (pp)"); ax.set_ylabel("Loss (pp)"); tidy(ax)
    title(fig, "Loss in a 2014–16 oil collapse, by XLE active weight",
          f"XLE minus SPY in the Jul 2014 – Jan 2016 episode: {glut * 100:.1f}%".replace("-", "−"))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13)
    save(fig, "deck_oil_collapse_loss.png", top_for(fig))
    return glut, marks


# ---- Risks -------------------------------------------------------------------------------------------------
def risk_metrics(w=860, h=440):
    M = pd.read_csv(OUT / "weighting_metrics.csv", index_col=0)
    C = pd.read_csv(OUT / "consistency_summary.csv", index_col=0)
    P = pd.read_csv(OUT / "consistency_scenario_payoffs.csv", index_col=0)
    rows = [("Tracking error vs IVV (%)", M.loc["Current", "te_bench"], M.loc["Base", "te_bench"]),
            ("3-month CVaR, 95% (%)", M.loc["Current", "cvar3"], M.loc["Base", "cvar3"]),
            ("Loss in a 2014–16 oil collapse (pp)", -C.loc["Current (drifted)", "oil_crash_tail_pp"], -C.loc["Base (half hedge)", "oil_crash_tail_pp"]),
            ("Worst scenario, 3 months (%)", -P["Current (drifted)"].min(), -P["Base (half hedge)"].min())]
    fig, ax = figure(w, h)
    y = np.arange(len(rows))[::-1]; hh = 0.38
    cur = [r[1] for r in rows]; new = [r[2] for r in rows]
    ax.barh(y + hh / 2, cur, height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, new, height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    for yy, vals in ((y + hh / 2, cur), (y - hh / 2, new)):
        for yi, v in zip(yy, vals):
            ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=11, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, max(cur) * 1.18)
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=12)
    title(fig, "Risk measures: current vs proposed", "Losses shown as positive numbers; lower is better")
    fig.subplots_adjust(left=0.37, right=0.97, bottom=0.08)
    save(fig, "deck_risk_metrics.png", top_for(fig))
    return rows


def scenario_returns(w=700, h=708):
    P = pd.read_csv(OUT / "consistency_scenario_payoffs.csv", index_col=0)[["Current (drifted)", "Base (half hedge)"]]
    names = {"Stagflationary sell-off": "Stagflationary\nsell-off"}
    fig, ax = figure(w, h)
    y = np.arange(len(P))[::-1]; hh = 0.38
    ax.barh(y + hh / 2, P.iloc[:, 0], height=hh, color=GREY, edgecolor=BG, lw=1.5, label="Current")
    ax.barh(y - hh / 2, P.iloc[:, 1], height=hh, color=GREEN, edgecolor=BG, lw=1.5, label="Proposed")
    for yy, col in ((y + hh / 2, 0), (y - hh / 2, 1)):
        hbar_labels(ax, yy, P.iloc[:, col].values, fmt=lambda v: pp(v, 2), pad=0.04, size=11)
    ax.set_yticks(y, [names.get(s, s) for s in P.index]); ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.75, 1.55); ax.set_xlabel("Active return vs SPY, 3 months (%)")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=12)
    title(fig, "Average 3-month return by scenario", "Module 8 scenario tree, 2003 onward")
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.09)
    save(fig, "deck_scenario_returns.png", top_for(fig))
    return P


def trigger_readings(w=700, h=708):
    wt = series("DCOILWTICO")
    be = series("T10YIE"); be_ref = be.loc[:DECIDE].iloc[-1]
    w26 = wt.loc["2026"]; decline = (1 - w26.iloc[-1] / w26.max()) * 100
    cs = pd.read_csv(OUT / "cycle_state.csv", index_col=0, parse_dates=True)["p_recession_12m"].dropna()
    vs = pd.read_csv(OUT / "valuation_sectors.csv", index_col=0)
    rows = [
        ("10-year breakeven", f"{be_ref:.2f}% on {be.loc[:DECIDE].index[-1]:%-d %b}", be_ref, be_ref + 0.25,
         (be_ref - 0.5, be_ref + 0.5), "Trigger: +25bp → full hedge"),
        ("WTI decline from 2026 peak", f"{decline:.0f}% (peak ${w26.max():.0f}, {w26.idxmax():%-d %b})", decline, 25,
         (0, 40), "Trigger: −25% → XLE to about +1pp"),
        ("Recession probability, 12 months", f"{cs.iloc[-1] * 100:.0f}% ({cs.index[-1]:%b})", cs.iloc[-1] * 100, 35,
         (0, 50), "Trigger: above 35% → add defensives"),
        ("Technology EPS revisions, 3 months", f"{pp(vs.loc['XLK', 'rev3'])}% (30 Sep)", vs.loc["XLK", "rev3"], 15,
         (0, 25), "Condition: above +15% → do not add to growth underweight"),
    ]
    fig, axes = figure(w, h, nrows=4)
    for ax, (name, reading, v, thr, (lo, hi), rule) in zip(axes, rows):
        met = v >= thr
        ax.barh(0, hi - lo, left=lo, height=0.5, color="#E6E8E6", lw=0)
        ax.barh(0, (min(v, thr) - lo), left=lo, height=0.5, color=LIGHT, alpha=0.7, lw=0)
        ax.plot(v, 0, "o", ms=12, color=RED if met else GREEN, mec=BG, mew=2, zorder=3)
        ax.plot([thr, thr], [-0.42, 0.42], color=INK, lw=2.5)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _, d=(2 if hi - lo < 2 else 0): f"{x:.{d}f}%"))
        ax.text(lo, 0.72, name, fontsize=13, color=INK, va="bottom", **BOLD)
        ax.text(hi, 0.72, f"{reading} · {'Met' if met else 'Not triggered'}", fontsize=11.5, color=INK2, va="bottom", ha="right")
        ax.text(lo, -1.25, rule, fontsize=11, color=MUTED, va="top")
    title(fig, "Review triggers: current readings vs thresholds", "Dot: reading · vertical line: threshold")
    fig.subplots_adjust(left=0.03, right=0.965, bottom=0.10, hspace=1.6)
    save(fig, "deck_trigger_readings.png", top_for(fig, extra=34))
    return rows


def main():
    print("\nDeck visuals from frozen data (freeze-2026-11-04), team style\n")
    A = weights_active()
    M, C, groups = performance()
    active_current()
    cpi_components()
    ip, rs = activity()
    p = payrolls()
    f = fed_funds()
    ff, t2, gap = fed_expected()
    last, s1, s2 = curve()
    ry = real_yields()
    w = wti()
    prem, med = pe_premium()
    s, oct_ = def_minus_grw()
    d, g = decomposition()
    valuation_revisions()
    glut, marks = oil_collapse_loss()
    rows = risk_metrics()
    P = scenario_returns()
    trig = trigger_readings()
    print("\nNumbers drawn (check against the deck):")
    print("  Performance Jun–Oct (%): " + ", ".join(f"{k} {v:+.2f}" for k, v in C.iloc[-1].items()))
    print("  Active return by group vs IVV sector mix (pp): " + ", ".join(f"{k} {v:+.2f}" for k, v in groups.items()))
    print(f"  Industrial production y/y {ip:+.2f}%, real retail sales y/y {rs:+.2f}%")
    print(f"  Payrolls: last {p.iloc[-1]:+.0f}k, 2026 average {p.loc['2026'].mean():+.0f}k, 3-month average {p.iloc[-3:].mean():+.0f}k")
    print(f"  Fed funds {ff:.2f}%, 2-year {t2:.2f}% ({gap:+.0f}bp); 10y {last.t10:.2f}%, 10y-2y {s1:+.0f}bp, 10y-ff {s2:+.0f}bp")
    print(f"  10y nominal {ry.n:.2f}%, real {ry.r:.2f}%, breakeven {ry.b:.2f}%")
    print(f"  WTI last ${w.iloc[-1]:.2f}, peak ${w.max():.2f}; P/E premium {prem.iloc[-1]:+.1f}% vs median {med.iloc[-1]:+.1f}%")
    print(f"  Defensives minus growth: October {oct_:+.2f}%, since 29 May {s.iloc[-1]:+.2f}pp")
    print("  Oil-collapse loss: " + ", ".join(f"{l} {wv:+.2f}pp -> {y:+.2f}pp" for l, wv, y, _ in marks))
    print("  Risk measures (current/proposed): " + "; ".join(f"{r[0]} {r[1]:.2f}/{r[2]:.2f}" for r in rows))
    print(f"\nSaved to {DECK.relative_to(ROOT)}/\n")


if __name__ == "__main__":
    main()
