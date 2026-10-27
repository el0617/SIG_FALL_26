"""Module 7 - What's priced in: valuations, earnings revisions and the equity risk premium.

Run (from the repo root):
    python -m src.valuation data/raw/bloomberg_valuation.xlsx   # first time: import the workbook, then run
    python -m src.valuation                                      # later runs use the cached CSVs
    python -m src.valuation --robust                             # specification robustness table

QUESTION
  Modules 1-5 ask whether the macro backdrop favours defensives. This module asks whether the
  market has ALREADY priced that in: is growth expensive or cheap relative to defensives, are
  analysts revising growth earnings up or down, and how much earnings outperformance does the
  growth leg's valuation premium require?

PRE-REGISTERED TESTS (fixed before any real Bloomberg data was loaded)
  Target: next-3-month Defensives-minus-Growth return (XLP, XLU, XLV minus XLK, XLY), annualized pp,
  exactly the series used in Modules 2, 3 and 5.
  1. HEADLINE - valuation gap: log(growth forward P/E / defensive forward P/E), measured as its
     deviation from the expanding-window median (real time: only history known at each date).
     Higher = growth unusually expensive = expected to favour defensives.
  2. SECOND - relative earnings revisions: 3-month log change in forward EPS, defensives minus growth.
     Higher = defensive earnings being revised up relative to growth = expected to favour defensives.
  Each is a one-regressor HAC(6) regression with a demeaned signal. The verdict uses the TILT (today's
  signal times its slope = what the signal adds to the long-run average spread), one-sided 5%:
  t >= 1.65 SUPPORTS the defensive thesis, t <= -1.65 CONTRADICTS, otherwise NEUTRAL.
  The equity risk premium and the 12-month horizon are context, not verdicts.

DATA (Bloomberg, monthly from 12/2005; see the walkthrough for the BDH formulas)
  Sheets PE, EPS, DY (and optional PX) with one column block per index:
    S5INFT S5COND S5TELS S5FINL S5INDU S5MATR S5ENRS S5CONS S5HLTH S5UTIL S5RLST SPX
  PE  = BEST_PE_RATIO   (blended 12-month forward, BEST_FPERIOD_OVERRIDE=BF)
  EPS = BEST_EPS        (blended 12-month forward)
  DY  = EQY_DVD_YLD_12M (trailing dividend yield)
  Real 10y yield for the equity risk premium comes from FRED (DFII10), already cached by Module 3.

TIMING
  Valuations and estimates are market data known at month-end t, so they pair with returns of
  months t+1..t+3 with no publication lag.
"""
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.data import RAW, level, sector_returns, complete_months
from src.utils import forward_excess
from src.regime import thesis_verdict, DEFENSIVE, GROWTH

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

IDX = {"XLK": "S5INFT", "XLY": "S5COND", "XLC": "S5TELS", "XLF": "S5FINL", "XLI": "S5INDU",
       "XLB": "S5MATR", "XLE": "S5ENRS", "XLP": "S5CONS", "XLV": "S5HLTH", "XLU": "S5UTIL",
       "IYR": "S5RLST", "SPY": "SPX"}
NAME = {"XLK": "Info tech (VGT)", "XLY": "Cons. discretionary", "XLC": "Comm. services", "XLF": "Financials",
        "XLI": "Industrials", "XLB": "Materials", "XLE": "Energy", "XLP": "Cons. staples", "XLV": "Health care",
        "XLU": "Utilities", "IYR": "Real estate (SCHH)", "SPY": "S&P 500"}
FIELDS = {"PE": ("val_pe", "BEST_PE_RATIO (forward P/E)", (3, 150)),
          "EPS": ("val_eps", "BEST_EPS (forward EPS)", (-50, 2000)),
          "DY": ("val_dy", "EQY_DVD_YLD_12M (dividend yield, %)", (0, 15)),
          "PX": ("val_px", "PX_LAST (index level)", (10, 20000))}

START = "2005-12-31"
MIN_HIST = 36          # months of history before the expanding median is trusted
REV = 3                # revision window, months
H = 3                  # headline horizon, months
HAC = 6
HEADLINE = "DEF-GRW"
COL_DEF, COL_GRW, COL_OTH = "#2e7d32", "#c62828", "#90a4ae"


# ------------------------------------------------------------------ import
def _to_dates(col):
    s = pd.Series(col)
    num = pd.to_numeric(s, errors="coerce")
    serial = num.between(20000, 80000)
    out = pd.to_datetime(s.where(~serial).astype(str), errors="coerce", format="mixed")
    out[serial] = pd.to_datetime("1899-12-30") + pd.to_timedelta(num[serial], unit="D")
    return out


def _find_sheet(xl, key):
    for s in xl.sheet_names:
        if s.strip().upper() == key:
            return s
    for s in xl.sheet_names:
        if re.search(rf"\b{key}\b", s.strip().upper()):
            return s
    return None


def _parse_sheet(df):
    """Find each index ticker in the first 3 rows. Two layouts are accepted:
    (a) BDH with Dts=S under each ticker: dates in the ticker's column, values in the next one;
    (b) one shared date column A, values under each ticker."""
    out = {}
    tick = {v: k for k, v in IDX.items()}
    for r in range(min(3, len(df))):
        for c in range(df.shape[1]):
            cell = str(df.iat[r, c]).upper().replace("0", "O") if pd.notna(df.iat[r, c]) else ""
            for code, etf in tick.items():
                if etf in out or not re.search(rf"\b{code.replace('0', 'O')}\b", cell):
                    continue
                body = df.iloc[r + 1:]
                d_own = _to_dates(body.iloc[:, c])
                if d_own.notna().mean() > 0.5 and c + 1 < df.shape[1]:
                    dates, vals = d_own, body.iloc[:, c + 1]
                else:
                    dates, vals = _to_dates(body.iloc[:, 0]), body.iloc[:, c]
                s = pd.Series(pd.to_numeric(vals, errors="coerce").values, index=dates.values).dropna()
                s = s[s.index.notna()]
                s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
                out[etf] = s.groupby(level=0).last().sort_index()
    return pd.DataFrame(out)


def import_workbook(path):
    xl = pd.ExcelFile(path)
    print(f"\nImporting {path}  (sheets: {', '.join(xl.sheet_names)})")
    for key, (fname, desc, (lo, hi)) in FIELDS.items():
        sh = _find_sheet(xl, key)
        if sh is None:
            print(f"  {key:<4} not found{' (optional)' if key == 'PX' else '  <-- REQUIRED'}")
            continue
        df = _parse_sheet(pd.read_excel(xl, sheet_name=sh, header=None))
        if df.empty:
            print(f"  {key:<4} sheet '{sh}': no index tickers found in the first 3 rows")
            continue
        df = df[[t for t in IDX if t in df.columns]]
        df.index.name = "Date"
        df.to_csv(RAW / f"{fname}.csv")
        miss = [IDX[t] for t in IDX if t not in df.columns]
        bad = ((df < lo) | (df > hi)).sum()
        print(f"  {key:<4} {desc}: {df.shape[1]} indices, {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}"
              f"{'; MISSING ' + ', '.join(miss) if miss else ''}")
        for t in df.columns:
            c = df[t].dropna()
            if c.empty:
                print(f"         {IDX[t]:<7} EMPTY - no numbers under this ticker (formula not refreshed, or pasted as #N/A?)")
                continue
            out = c[(c < lo) | (c > hi)]
            note = f"   {len(out)} outside {lo}-{hi}: " + ", ".join(f"{d:%Y-%m} {v:,.1f}" for d, v in out.head(4).items()) if len(out) else ""
            gaps = len(c) - c.index.to_series().diff().dt.days.le(35).sum() - 1
            print(f"         {IDX[t]:<7} {len(c):>3} months  {c.index[0]:%Y-%m} to {c.index[-1]:%Y-%m}  last {c.iloc[-1]:,.2f}"
                  f"{'  (' + str(gaps) + ' gaps)' if gaps > 0 else ''}{note}")
    print()


def load():
    D = {}
    for key, (fname, _, _) in FIELDS.items():
        f = RAW / f"{fname}.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True).loc[START:]
            empty = [c for c in df.columns if df[c].dropna().empty]
            if empty:
                print(f"  WARNING {key}: no data for {', '.join(IDX[c] for c in empty)} - left out")
                df = df.drop(columns=empty)
            D[key] = df
        elif key != "PX":
            raise FileNotFoundError(f"{f} missing: run  python -m src.valuation data/raw/bloomberg_valuation.xlsx")
    need = ["SPY"] + DEFENSIVE + GROWTH
    for key in ["PE", "EPS"]:
        miss = [IDX[t] for t in need if t not in D[key].columns]
        if miss:
            raise ValueError(f"{key} sheet has no data for {', '.join(miss)}: these are required (legs and SPX). "
                             f"Re-pull them on the terminal and rerun the import.")
    return D


# ------------------------------------------------------------------ signals
def leg_pe(PE, members):
    """Equal-weight leg P/E = 1 / average earnings yield (harmonic mean; handles very high P/Es)."""
    ey = (1 / PE[members]).mean(axis=1, skipna=False)
    return 1 / ey


def leg_rev(EPS, members, n=REV):
    """Average n-month log change in forward EPS (negative or zero EPS dropped)."""
    e = EPS[members].where(EPS[members] > 0)
    return np.log(e).diff(n).mean(axis=1, skipna=False)


def dev_from_median(x, min_hist=MIN_HIST, window=None):
    """Deviation from the median of history known at each date (expanding, or rolling `window`)."""
    med = x.expanding(min_hist).median() if window is None else x.rolling(window, min_periods=min_hist).median()
    return x - med


def build_signals(D, grw=GROWTH, dfn=DEFENSIVE, rev_n=REV, window=None):
    PE, EPS = D["PE"], D["EPS"]
    pe_g, pe_d = leg_pe(PE, grw), leg_pe(PE, dfn)
    gap = np.log(pe_g / pe_d).rename("gap")
    rel_rev = (leg_rev(EPS, dfn, rev_n) - leg_rev(EPS, grw, rev_n)).rename("rel_rev")
    real = level("DFII10")
    erp = (100 / PE["SPY"] - real.reindex(PE.index, method="ffill")).rename("erp")
    return pd.DataFrame({"pe_grw": pe_g, "pe_def": pe_d, "gap": gap,
                         "gap_dev": dev_from_median(gap, window=window), "rel_rev": rel_rev,
                         "erp": erp, "ey_spx": 100 / PE["SPY"], "real10y": real.reindex(PE.index, method="ffill")})


def priced_in(S):
    """How much earnings outperformance today's growth premium requires."""
    g = S["gap"].dropna()
    med, now = g.median(), g.iloc[-1]
    return dict(premium_now=np.exp(now) - 1, premium_median=np.exp(med) - 1,
                pct=(g < now).mean(), outgrowth=np.exp(now - med) - 1, reversion=np.exp(med - now) - 1,
                lo20=np.exp(g.quantile(.2)) - 1, hi80=np.exp(g.quantile(.8)) - 1, asof=g.index[-1], start=g.index[0])


def sector_table(D):
    PE, EPS, DY = D["PE"], D["EPS"], D.get("DY")
    rows = []
    for t in [t for t in IDX if t in PE.columns]:
        pe = PE[t].dropna()
        if pe.empty:
            continue
        rel = (PE[t] / PE["SPY"]).dropna()
        e = EPS[t].where(EPS[t] > 0) if t in EPS else None
        rows.append(dict(
            ticker=t, sector=NAME[t], fwd_pe=pe.iloc[-1], pe_pct=(pe < pe.iloc[-1]).mean() * 100,
            rel_pe=rel.iloc[-1] if t != "SPY" else np.nan,
            rel_pct=(rel < rel.iloc[-1]).mean() * 100 if t != "SPY" else np.nan,
            rev3=(np.exp(np.log(e).diff(3).iloc[-1]) - 1) * 100 if e is not None else np.nan,
            rev12=(np.exp(np.log(e).diff(12).iloc[-1]) - 1) * 100 if e is not None else np.nan,
            dy=DY[t].dropna().iloc[-1] if DY is not None and t in DY else np.nan,
            leg="Defensive" if t in DEFENSIVE else "Growth" if t in GROWTH else "Other"))
    return pd.DataFrame(rows).set_index("ticker")


# ------------------------------------------------------------------ tests
def returns_ex1():
    """Next-month sector returns in excess of SPY, all 11 sectors (XLC from 2018). Row t = month t+1."""
    r = complete_months(sector_returns())
    cols = ["SPY"] + [t for t in IDX if t != "SPY" and t in r.columns]
    return forward_excess(r[cols].loc["1999-01-31":])


def spread_panel(ex1, long, short, h):
    s = ex1[long].mean(axis=1, skipna=False) - ex1[short].mean(axis=1, skipna=False)
    return s.rolling(h).mean().shift(-(h - 1))      # row t = average of months t+1..t+h


def predictive(x, y, hac=HAC, drop=None):
    """y (annualized pp) on the demeaned signal x; returns expected spread today and the signal's tilt."""
    d = pd.concat([y * 1200, x], axis=1, keys=["y", "x"], sort=True).dropna()
    if drop is not None:
        d = d.drop(d.index[(d.index >= drop[0]) & (d.index <= drop[1])])
    mu = d["x"].mean()
    X = np.column_stack([np.ones(len(d)), d["x"].values - mu])
    res = sm.OLS(d["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    b = np.asarray(res.params)
    V = np.asarray(res.cov_params()) if hasattr(res, "cov_params") else np.diag(np.asarray(res.bse) ** 2)
    xnow = x.dropna()
    x0 = np.array([1.0, xnow.iloc[-1] - mu])
    fit, se = float(x0 @ b), float(np.sqrt(x0 @ V @ x0))
    tilt, tse = float(b[1] * x0[1]), float(abs(x0[1]) * np.sqrt(V[1, 1]))
    return dict(slope=float(b[1]), slope_t=float(np.asarray(res.tvalues)[1]), avg=float(b[0]),
                fit=fit, t=fit / se, tilt=tilt, tilt_t=tilt / tse if tse > 0 else np.nan,
                tilt_lo=tilt - 1.96 * tse, tilt_hi=tilt + 1.96 * tse, x_now=float(xnow.iloc[-1]),
                n=len(d), start=d.index[0], end=d.index[-1], asof=xnow.index[-1])


def verdict(r):
    return thesis_verdict({"raw_t": r["tilt_t"]})


def run_tests(S, ex1, grw=GROWTH, dfn=DEFENSIVE, h=H, drop=None):
    y = spread_panel(ex1, dfn, grw, h)
    hac = max(HAC, h + 3)
    return {"gap": predictive(S["gap_dev"], y, hac, drop), "rel_rev": predictive(S["rel_rev"], y, hac, drop),
            "erp": predictive(S["erp"], y, hac, drop)}


# ------------------------------------------------------------------ plots
def plot_gap(S, P, path):
    g = np.exp(S["gap"].dropna())
    med = np.exp(S["gap"].expanding(MIN_HIST).median())
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(g.index, g, color=COL_GRW, lw=1.8, label="Growth P/E ÷ defensive P/E (forward)")
    ax[0].plot(med.index, med, color="k", lw=1, ls="--", label="Median of history known at the time")
    ax[0].axhspan(np.exp(S["gap"].quantile(.2)), np.exp(S["gap"].quantile(.8)), color="0.9", label="20th-80th percentile, full sample")
    ax[0].scatter([g.index[-1]], [g.iloc[-1]], color=COL_GRW, zorder=5, s=40)
    ax[0].annotate(f"{g.iloc[-1]:.2f}x  ({P['pct']:.0%} percentile)", (g.index[-1], g.iloc[-1]),
                   xytext=(-10, 14), textcoords="offset points", ha="right", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7"))
    ax[0].set_ylabel("Relative forward P/E"); ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)
    ax[0].set_title("Is growth expensive relative to defensives?")
    pe = S[["pe_grw", "pe_def"]].dropna()
    ax[1].plot(pe.index, pe["pe_grw"], color=COL_GRW, lw=1.3, label="Growth leg (XLK, XLY)")
    ax[1].plot(pe.index, pe["pe_def"], color=COL_DEF, lw=1.3, label="Defensive leg (XLP, XLU, XLV)")
    ax[1].set_ylabel("Forward P/E"); ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sectors(T, path):
    T = T.drop("SPY", errors="ignore").dropna(subset=["pe_pct", "rev3"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    col = T["leg"].map({"Defensive": COL_DEF, "Growth": COL_GRW, "Other": COL_OTH})
    ax.scatter(T["pe_pct"], T["rev3"], s=90, c=col, edgecolor="k", lw=.5, zorder=3)
    for t, r in T.iterrows():
        ax.annotate(t, (r["pe_pct"], r["rev3"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axvline(50, color="k", lw=.8); ax.axhline(0, color="k", lw=.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Forward P/E percentile vs own history since 2006 (right = expensive)")
    ax.set_ylabel("3-month change in forward EPS, % (up = estimates rising)")
    ax.set_title("Sector valuation vs earnings momentum")
    for (x, y, txt) in [(.99, .98, "Expensive, estimates rising"), (.01, .98, "Cheap, estimates rising"),
                        (.99, .02, "Expensive, estimates falling"), (.01, .02, "Cheap, estimates falling")]:
        ax.text(x, y, txt, transform=ax.transAxes, ha="right" if x > .5 else "left",
                va="top" if y > .5 else "bottom", fontsize=8, color="0.45")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_revisions_erp(S, path):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    r = S["rel_rev"].dropna() * 100
    ax[0].fill_between(r.index, r, 0, where=r >= 0, color=COL_DEF, alpha=.6, label="Defensive estimates rising faster")
    ax[0].fill_between(r.index, r, 0, where=r < 0, color=COL_GRW, alpha=.6, label="Growth estimates rising faster")
    ax[0].set_ylabel("Pp, 3-month log change"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=.3)
    ax[0].set_title("Relative earnings revisions: defensives minus growth")
    e = S[["erp", "ey_spx", "real10y"]].dropna()
    ax[1].plot(e.index, e["ey_spx"], color="#1565c0", lw=1.2, label="S&P 500 forward earnings yield")
    ax[1].plot(e.index, e["real10y"], color="#6a1b9a", lw=1.2, label="10y real yield (TIPS)")
    ax[1].plot(e.index, e["erp"], color="k", lw=1.8, label="Equity risk premium (difference)")
    ax[1].axhline(e["erp"].median(), color="k", ls="--", lw=.8)
    ax[1].set_ylabel("%"); ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=.3)
    ax[1].set_title("Are equities priced for the real-yield regime?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_thesis(R3, R12, path):
    lab = {"gap": "Valuation gap", "rel_rev": "Relative revisions", "erp": "Equity risk premium"}
    keys = list(lab)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(keys)); w = .36
    for i, (R, name, c) in enumerate([(R3, "Next 3 months (headline)", "#37474f"), (R12, "Next 12 months (context)", "#90a4ae")]):
        v = [R[k]["tilt"] for k in keys]
        lo = [R[k]["tilt"] - R[k]["tilt_lo"] for k in keys]; hi = [R[k]["tilt_hi"] - R[k]["tilt"] for k in keys]
        ax.bar(x + (i - .5) * w, v, w, color=c, label=name)
        ax.errorbar(x + (i - .5) * w, v, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor="k", capsize=3, lw=1)
        for xi, k in zip(x, keys):
            ax.annotate(f"t {R[k]['tilt_t']:+.2f}", (xi + (i - .5) * w, max(R[k]["tilt_hi"], 0)), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in keys])
    ax.set_ylabel("Tilt toward defensives, pp/yr")
    ax.set_title("What today's pricing adds to the defensives-minus-growth spread (95% bands)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def _line(name, r):
    return (f"  {name:<22} signal now {r['x_now']:+.3f}   slope {r['slope']:+.1f} (t {r['slope_t']:+.2f})   "
            f"tilt {r['tilt']:+.1f} pp/yr (t {r['tilt_t']:+.2f})   expected {r['fit']:+.1f} (t {r['t']:+.2f})   "
            f"n={r['n']} {r['start']:%Y-%m}..{r['end']:%Y-%m}")


def main():
    D = load()
    S = build_signals(D)
    ex1 = returns_ex1()
    P = priced_in(S)
    T = sector_table(D)
    R3 = run_tests(S, ex1, h=3)
    R12 = run_tests(S, ex1, h=12)

    print("\n" + "=" * 100)
    print(f"MODULE 7 - WHAT'S PRICED IN   (valuations as of {S['gap'].dropna().index[-1]:%Y-%m-%d})")
    print("=" * 100)
    print("\nSECTOR VALUATION AND EARNINGS MOMENTUM")
    show = T.copy()
    show.index = [f"{t} {IDX[t]}" for t in show.index]
    print(show[["sector", "fwd_pe", "pe_pct", "rel_pe", "rel_pct", "rev3", "rev12", "dy", "leg"]]
          .rename(columns={"fwd_pe": "fwd P/E", "pe_pct": "P/E pctile", "rel_pe": "vs SPX", "rel_pct": "rel pctile",
                           "rev3": "EPS rev 3m %", "rev12": "EPS rev 12m %", "dy": "div yld"})
          .to_string(float_format=lambda v: f"{v:,.1f}"))

    print("\nTHE TWO LEGS")
    now = S.dropna(subset=["gap"]).iloc[-1]
    print(f"  Growth leg forward P/E {now['pe_grw']:.1f} | defensive leg {now['pe_def']:.1f} | "
          f"growth premium {P['premium_now']:+.0%} (median since {P['start']:%Y} {P['premium_median']:+.0%}, "
          f"20-80th pct {P['lo20']:+.0%} to {P['hi80']:+.0%}) -> {P['pct']:.0%} percentile")
    if P["outgrowth"] >= 0:
        print(f"  WHAT'S PRICED IN: if growth's premium returns to its median, growth must out-earn defensives by "
              f"{P['outgrowth']:.0%} (cumulative) just to hold its price vs defensives; with equal earnings growth,"
              f" reversion costs growth {-P['reversion']:.0%} vs defensives.")
    else:
        print(f"  WHAT'S PRICED IN: growth trades BELOW its median premium; with equal earnings growth, reversion to "
              f"the median would add {P['reversion']:+.0%} to growth vs defensives (a headwind for the thesis).")
    rr = S["rel_rev"].dropna()
    print(f"  Relative revisions (3m, defensives minus growth): {rr.iloc[-1] * 100:+.1f} pp "
          f"({(rr < rr.iloc[-1]).mean():.0%} percentile)")
    e = S["erp"].dropna()
    print(f"  Equity risk premium: forward earnings yield {now['ey_spx']:.2f}% - real 10y {now['real10y']:.2f}% = "
          f"{e.iloc[-1]:.2f}% ({(e < e.iloc[-1]).mean():.0%} percentile; median {e.median():.2f}%)")

    print("\nPREDICTIVE TESTS - next-3-month DEF-GRW spread, annualized pp, HAC errors")
    print(_line("Valuation gap (HEAD)", R3["gap"]) + f"   -> {verdict(R3['gap'])}")
    print(_line("Relative revisions", R3["rel_rev"]) + f"   -> {verdict(R3['rel_rev'])}")
    print(_line("Equity risk premium", R3["erp"]) + "   -> context only")
    print("\n  12-month horizon (context; overlapping, HAC(15))")
    for k, n in [("gap", "Valuation gap"), ("rel_rev", "Relative revisions"), ("erp", "Equity risk premium")]:
        print(_line(n, R12[k]))

    print("\nEVIDENCE SUMMARY")
    v = verdict(R3["gap"])
    print(f"  Headline (valuation gap, 3 months): {v}  - tilt {R3['gap']['tilt']:+.1f} pp/yr (t {R3['gap']['tilt_t']:+.2f})")
    print(f"  Second   (relative revisions, 3 months): {verdict(R3['rel_rev'])}  - tilt {R3['rel_rev']['tilt']:+.1f} "
          f"pp/yr (t {R3['rel_rev']['tilt_t']:+.2f})")
    print(f"  Growth trades at {P['premium_now']:+.0%} to defensives ({P['pct']:.0%} percentile); "
          f"pricing requires {P['outgrowth']:+.0%} relative earnings outperformance to sustain.")

    plot_gap(S, P, OUT / "valuation_gap.png")
    plot_sectors(T, OUT / "valuation_sectors.png")
    plot_revisions_erp(S, OUT / "valuation_revisions_erp.png")
    plot_thesis(R3, R12, OUT / "valuation_thesis.png")
    S.to_csv(OUT / "valuation_signals.csv"); T.to_csv(OUT / "valuation_sectors.csv")
    print("\nSaved: outputs/valuation_gap.png, valuation_sectors.png, valuation_revisions_erp.png, "
          "valuation_thesis.png, valuation_signals.csv, valuation_sectors.csv\n")


def robustness():
    D = load()
    ex1 = returns_ex1()
    base = build_signals(D)
    settings = [
        ("Baseline", dict(), dict()),
        ("Horizon 1 month", dict(), dict(h=1)),
        ("Horizon 6 months", dict(), dict(h=6)),
        ("Horizon 12 months", dict(), dict(h=12)),
        ("Revisions over 1 month", dict(rev_n=1), dict()),
        ("Revisions over 6 months", dict(rev_n=6), dict()),
        ("Median over rolling 5 years", dict(window=60), dict()),
        ("Exclude Mar-2020..Dec-2021", dict(), dict(drop=(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")))),
        ("Exclude 2022", dict(), dict(drop=(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")))),
        ("Defensives ex-utilities", dict(dfn=["XLP", "XLV"]), dict(dfn=["XLP", "XLV"])),
        ("Growth incl. comm. services", dict(grw=GROWTH + ["XLC"]), dict(grw=GROWTH + ["XLC"])),
    ]
    rows = []
    for name, sk, tk in settings:
        try:
            S = build_signals(D, **sk) if sk else base
            if "XLC" in tk.get("grw", []) and "XLC" not in ex1.columns:
                raise KeyError("XLC returns not in the panel")
            R = run_tests(S, ex1, **tk)
            rows.append(dict(setting=name, gap_tilt=R["gap"]["tilt"], gap_t=R["gap"]["tilt_t"], gap=verdict(R["gap"]),
                             rev_tilt=R["rel_rev"]["tilt"], rev_t=R["rel_rev"]["tilt_t"], rev=verdict(R["rel_rev"]),
                             n=R["gap"]["n"]))
        except Exception as ex:
            rows.append(dict(setting=name, gap=f"skipped: {ex}"))
    T = pd.DataFrame(rows).set_index("setting")
    print("\nMODULE 7 ROBUSTNESS - tilt toward defensives, pp/yr (t), one-sided 5% rule")
    print(T.to_string(float_format=lambda v: f"{v:+.2f}"))
    for c, n in [("gap", "Valuation gap"), ("rev", "Relative revisions")]:
        v = T[c].astype(str)
        k = len(v[~v.str.startswith("skipped")])
        print(f"  {n}: SUPPORTS {(v == 'SUPPORTS').sum()}/{k}, CONTRADICTS {(v == 'CONTRADICTS').sum()}/{k}, "
              f"tilt positive in {(T[c + '_tilt'] > 0).sum()}/{k}")
    T.to_csv(OUT / "valuation_robustness.csv")
    print("\nSaved: outputs/valuation_robustness.csv\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        import_workbook(args[0])
    if "--robust" in sys.argv:
        robustness()
    else:
        main()