"""Module 5 - Market-based confirmation (v1).

Run:
    python -m src.market            # main run: signal scorecard, composite, headline test, plots
    python -m src.market --robust   # specification robustness table

QUESTION
  Modules 1-3 use macro and rates data. Module 5 asks the market directly: are prices already
  behaving the way a "defensives over growth / risk-off" view expects? It is independent evidence
  (prices, not economic releases) and the natural cross-check on the thesis.

SIGNALS (each oriented so that + = CONFIRMS the defensive thesis; thresholds fixed in advance)
  Sector rotation  trade_3m      DEF-GRW total return, last 3 months          (+-1.5%)
                   trade_6m      DEF-GRW total return, last 6 months          (+-3.0%)
                   staples_disc  XLP minus XLY, last 3 months                 (+-1.5%)
                   def_cyc       Defensives minus Cyclicals, last 3 months    (+-1.5%)
  Cross-asset      large_small   SPY minus IWM, last 3 months                 (+-1.5%)
                   gold_copper   gold/copper ratio change, last 3 months      (+-3.0%)
  Risk             credit        3-month change in the HY spread (bp)         (+-25bp)
                   vix_term      VIX / VIX3M; >1.00 confirms, <0.90 contradicts (scorecard only, 2007+)
  Trend/breadth    spy_trend     SPY vs its 10-month average; below confirms  (+-2%)
                   breadth       share of sectors above their 10-month average; <40% confirms, >70% contradicts
  Each signal votes +1 / 0 / -1. Net score = (confirm - contradict) / signals, from -1 to +1.

PRE-REGISTERED HEADLINE TEST
  Continuous composite = real-time z-scores of the oriented raw signals (all except vix_term, which
  starts in 2007), averaged within each of the four groups and then across groups (equal group weights,
  so the four correlated sector-rotation signals count once). Regress next-3-month DEF-GRW on it (HAC), report expected spread and tilt
  with the same verdict rule as Modules 1-3. This tests whether market confirmation has historically
  MATTERED for the next 3 months, not just whether the market agrees today.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import RAW, fred, fred_optional, monthly, optional_csv, sector_returns, complete_months, current_month_end
from src.utils import zscore, conditional_stats, episode_table
from src.regime import thesis_verdict, SECTORS, DEFENSIVE, GROWTH, SAMPLE_START
from src.cycle import returns_panels, CYCLICAL
from src.policy import predictive_test

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

EXTRA = ["SPY", "IWM", "HG=F", "GC=F"]      # small caps, copper and gold futures (yfinance)
SMA = 10                                    # months for trend / breadth
SIGNALS = {   # name: (group, threshold, label)
    "trade_3m":     ("Sector rotation", 1.5, "Defensives minus Growth, 3m return (%)"),
    "trade_6m":     ("Sector rotation", 3.0, "Defensives minus Growth, 6m return (%)"),
    "staples_disc": ("Sector rotation", 1.5, "Staples minus Discretionary, 3m (%)"),
    "def_cyc":      ("Sector rotation", 1.5, "Defensives minus Cyclicals, 3m (%)"),
    "large_small":  ("Cross-asset", 1.5, "Large minus small caps (SPY-IWM), 3m (%)"),
    "gold_copper":  ("Cross-asset", 3.0, "Gold/copper ratio change, 3m (%)"),
    "credit":       ("Risk", 25.0, "HY spread change, 3m (bp)"),
    "vix_term":     ("Risk", None, "VIX / VIX3M (>1.00 stress, <0.90 calm)"),
    "spy_trend":    ("Trend & breadth", 2.0, "SPY trend: % below 10-month average (negative = above)"),
    "breadth":      ("Trend & breadth", None, "Sectors above 10-month average (%; <40 / >70)"),
}
COMPOSITE = [s for s in SIGNALS if s != "vix_term"]
BUCKETS = ["Market contradicts", "Mixed", "Market confirms"]
BUCKET_CUT = 0.3
HEADLINE = "DEF-GRW"


# ------------------------------------------------------------------ data
def extra_prices(refresh=False):
    """Monthly closes for SPY, IWM, copper and gold futures, cached in data/raw/market_prices.csv."""
    f = RAW / "market_prices.csv"
    if f.exists() and not refresh:
        p = pd.read_csv(f, index_col=0, parse_dates=True)
        if set(EXTRA) <= set(p.columns):
            return p
    import yfinance as yf
    px = yf.download(EXTRA, start="1998-12-01", auto_adjust=True, progress=False)["Close"]
    p = px.resample("ME").last()
    p.to_csv(f)
    return p


def total_return_index(r):
    return (1 + r.fillna(0)).cumprod().where(r.notna().cumsum() > 0)


def build_signals(h=3, h_long=6, thresholds=None):
    """Raw oriented signal values at each month-end (+ = confirms the defensive thesis)."""
    thr = {k: v[1] for k, v in SIGNALS.items()}
    thr.update(thresholds or {})
    r = complete_months(sector_returns())
    idx = total_return_index(r[["SPY"] + SECTORS])
    px = extra_prices().reindex(idx.index)

    def ret(cols, n):
        return (idx[cols].pct_change(n, fill_method=None) * 100)

    def basket(cols, n):                          # equal-weight basket, monthly rebalanced
        b = total_return_index(r[cols].mean(axis=1, skipna=False))
        return b.pct_change(n, fill_method=None) * 100

    hy = optional_csv("hy_oas")
    hy = monthly(hy) if hy is not None else monthly(fred("BAMLH0A0HYM2")) * 100
    vix, vix3m = fred_optional("VIXCLS"), fred_optional("VXVCLS")
    sma = idx[SECTORS].rolling(SMA).mean()
    above = (idx[SECTORS] > sma).astype(float).where(sma.notna())
    breadth = above.mean(axis=1) * 100
    breadth = breadth.where(above.notna().sum(axis=1) >= 8)

    X = pd.DataFrame({
        "trade_3m":     basket(DEFENSIVE, h) - basket(GROWTH, h),
        "trade_6m":     basket(DEFENSIVE, h_long) - basket(GROWTH, h_long),
        "staples_disc": ret("XLP", h) - ret("XLY", h),
        "def_cyc":      basket(DEFENSIVE, h) - basket(CYCLICAL, h),
        "large_small":  (px["SPY"].pct_change(h, fill_method=None) - px["IWM"].pct_change(h, fill_method=None)) * 100,
        "gold_copper":  (px["GC=F"] / px["HG=F"]).pct_change(h, fill_method=None) * 100,
        "credit":       hy.reindex(idx.index).diff(h),
        "vix_term":     (monthly(vix) / monthly(vix3m)).reindex(idx.index) if vix is not None and vix3m is not None else np.nan,
        "spy_trend":    -(idx["SPY"] / idx["SPY"].rolling(SMA).mean() - 1) * 100,
        "breadth":      breadth,
    }, index=idx.index).astype(float)
    return X, thr


def votes(X, thr):
    V = pd.DataFrame(index=X.index)
    for k in X.columns:
        x = X[k]
        if k == "vix_term":
            v = np.where(x > 1.0, 1, np.where(x < 0.9, -1, 0))
        elif k == "breadth":
            v = np.where(x < 40, 1, np.where(x > 70, -1, 0))
        else:
            v = np.where(x > thr[k], 1, np.where(x < -thr[k], -1, 0))
        V[k] = pd.Series(v, index=X.index).where(x.notna())
    return V


def net_score(V, cols=None):
    cols = cols or list(V.columns)
    return V[cols].mean(axis=1).where(V[cols].notna().sum(axis=1) >= len(cols) - 1).rename("net_score")


def composite(X, cols=None, min_periods=36):
    cols = cols or COMPOSITE
    Xc = X[cols].copy()
    if "breadth" in Xc:
        Xc["breadth"] = -Xc["breadth"]                                 # low breadth = confirms
    Z = zscore(Xc, window=None, min_periods=min_periods)
    # equal weight per GROUP (the four sector-rotation signals are highly correlated and would otherwise
    # dominate): average within each group, then across groups
    groups = {}
    for c in cols:
        groups.setdefault(SIGNALS[c][0], []).append(c)
    G = pd.DataFrame({g: Z[c].mean(axis=1).where(Z[c].notna().all(axis=1)) for g, c in groups.items()})
    return G.mean(axis=1).where(G.notna().all(axis=1)).rename("market_composite"), Z


def bucket(score, cut=BUCKET_CUT):
    lab = np.where(score >= cut, BUCKETS[2], np.where(score <= -cut, BUCKETS[0], BUCKETS[1]))
    return pd.Series(lab, index=score.index, dtype=object).where(score.notna()).rename("market_view")


# ------------------------------------------------------------------ plots
C_CONF, C_CONTRA, C_NEU = "#2e7d32", "#c62828", "#9e9e9e"


def plot_scorecard(X, V, thr, path):
    last = X.index[X[COMPOSITE].notna().all(axis=1)][-1]
    names = list(SIGNALS)
    vals, cols, txt = [], [], []
    for k in names:
        x, v = X.loc[last, k], V.loc[last, k]
        if pd.isna(x):
            vals.append(0); cols.append("white"); txt.append("n/a"); continue
        if k == "vix_term":
            norm = (x - 0.95) / 0.05
        elif k == "breadth":
            norm = (55 - x) / 15
        else:
            norm = x / thr[k]
        vals.append(float(np.clip(norm, -3, 3)))
        cols.append(C_CONF if v > 0 else C_CONTRA if v < 0 else C_NEU)
        txt.append(f"{x:+.1f}" if k not in ("vix_term", "breadth") else f"{x:.2f}" if k == "vix_term" else f"{x:.0f}%")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.axvline(0, c="k", lw=1); ax.axvspan(-1, 1, color="grey", alpha=0.12)
    for yy, v, t in zip(y, vals, txt):
        ax.text(v + (0.08 if v >= 0 else -0.08), yy, t, va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{SIGNALS[k][2]}  [{SIGNALS[k][0]}]" for k in names], fontsize=8.5)
    ax.set_xlim(-3.6, 3.6)
    ax.set_xlabel("← contradicts the defensive thesis      reading ÷ threshold      confirms →")
    n_c, n_x = int((V.loc[last] > 0).sum()), int((V.loc[last] < 0).sum())
    ax.set_title(f"Market confirmation scorecard, {last:%b %Y}: {n_c} confirm · {n_x} contradict · "
                 f"{int(V.loc[last].notna().sum()) - n_c - n_x} neutral (grey band = inside threshold)", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_ratios(path, years=5):
    r = complete_months(sector_returns())
    px = extra_prices()
    d = total_return_index(r[DEFENSIVE].mean(axis=1, skipna=False)) / total_return_index(r[GROWTH].mean(axis=1, skipna=False))
    idx = total_return_index(r[["SPY", "XLP", "XLY"]])
    series = {"Defensives / Growth": d, "Staples / Discretionary (XLP/XLY)": idx["XLP"] / idx["XLY"],
              "Large / small caps (SPY/IWM)": px["SPY"] / px["IWM"], "Gold / copper": px["GC=F"] / px["HG=F"]}
    hy = optional_csv("hy_oas")
    series["HY spread (bp)"] = monthly(hy) if hy is not None else monthly(fred("BAMLH0A0HYM2")) * 100
    series["SPY total return index"] = idx["SPY"]
    start = r.index[-1] - pd.DateOffset(years=years)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (name, s) in zip(axes.ravel(), series.items()):
        s = s.dropna()
        m = s.rolling(SMA).mean()
        ax.plot(s.loc[start:].index, s.loc[start:], lw=1.5, label=name)
        ax.plot(m.loc[start:].index, m.loc[start:], lw=1, ls="--", c="grey", label=f"{SMA}-month average")
        ax.set_title(name + "  (rising = defensive/risk-off)" if "SPY total" not in name else name + "  (falling = risk-off)",
                     fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"What the market is doing: key ratios, last {years} years (dashed = {SMA}-month average)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_history(score, comp, path, start="2003-01-01"):
    r = complete_months(sector_returns())
    rel = total_return_index(r[DEFENSIVE].mean(axis=1, skipna=False)) / total_return_index(r[GROWTH].mean(axis=1, skipna=False))
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 3]})
    ax[0].plot(rel.loc[start:].index, rel.loc[start:], c="#37474f", lw=1.5)
    ax[0].set_title("Defensives / Growth total-return ratio (rising = the trade is working)", fontsize=10)
    ax[0].set_yscale("log")
    s = score.loc[start:]
    ax[1].fill_between(s.index, 0, s, where=s > 0, color=C_CONF, alpha=0.5, step="mid", label="Net vote (confirm)")
    ax[1].fill_between(s.index, 0, s, where=s < 0, color=C_CONTRA, alpha=0.5, step="mid", label="Net vote (contradict)")
    ax[1].plot(comp.loc[start:].index, comp.loc[start:], c="k", lw=1, label="Continuous composite (z)")
    for c in (BUCKET_CUT, -BUCKET_CUT):
        ax[1].axhline(c, c="grey", ls=":", lw=1)
    ax[1].axhline(0, c="k", lw=0.8); ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title("Market confirmation over time (dotted = ±0.3 bucket cut-offs)", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_thesis(bs, current, p, path):
    s = bs.set_index("state").reindex(BUCKETS).dropna(subset=["raw"])
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {BUCKETS[0]: C_CONTRA, BUCKETS[1]: C_NEU, BUCKETS[2]: C_CONF}
    bars = ax.bar(x, s["raw"], color=[colors[k] for k in s.index], alpha=0.85, linewidth=0)
    for b, k in zip(bars, s.index):
        if k == current:
            b.set_edgecolor("k"); b.set_linewidth(2.5)
    ax.errorbar(x, s["raw"], yerr=[s["raw"] - s["raw_lo"], s["raw_hi"] - s["raw"]], fmt="none", ecolor="k", capsize=5)
    ax.axhline(0, c="k", lw=1)
    ax.axhline(s["base"].iloc[0], c="k", ls="--", lw=1, label=f"All-months average ({s['base'].iloc[0]:+.1f})")
    for k, (_, r) in enumerate(s.iterrows()):
        ax.text(k, max(r["raw_hi"], 0) + 0.8, f"t={r['raw_t']:.1f}\n{int(r['months'])} mo · {int(r['episodes'])} ep",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(s.index)
    lo, hi = min(s["raw_lo"].min(), 0) * 1.15 - 1, max(s["raw_hi"].max(), 0) * 1.15 + 6
    ax.set_ylim(lo, hi)
    ax.set_ylabel("DEF-GRW, average of next 3 months, annualized pp")
    ax.set_title(f"Does market confirmation matter? DEF-GRW over the next 3 months by net vote (black edge = today)\n"
                 f"Headline composite test: expected {p['fit']:+.1f} pp/yr (t={p['t']:.2f}) = average {p['avg']:+.1f} "
                 f"+ market tilt {p['tilt']:+.1f} (t={p['tilt_t']:.2f})", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("=" * 90, "\nMODULE 5 - MARKET-BASED CONFIRMATION (v1)")
    X, thr = build_signals()
    V = votes(X, thr)
    score = net_score(V)
    comp, Z = composite(X)
    last = X.index[X[COMPOSITE].notna().all(axis=1)][-1]
    print(f"  latest complete month: {last:%b %Y}; composite available from {comp.first_valid_index():%b %Y}")
    print("  first available: " + ", ".join(f"{k} {X[k].first_valid_index():%Y-%m}" for k in X.columns
                                             if X[k].first_valid_index() is not None))

    # ---- scorecard
    card = pd.DataFrame({"group": [SIGNALS[k][0] for k in SIGNALS], "signal": [SIGNALS[k][2] for k in SIGNALS],
                         "reading": X.loc[last].round(2).values,
                         "threshold": [thr[k] if thr[k] is not None else "rule" for k in SIGNALS],
                         "vote": V.loc[last].map({1: "CONFIRMS", 0: "neutral", -1: "CONTRADICTS"}).values,
                         "vote_3m_ago": V.shift(3).loc[last].map({1: "CONFIRMS", 0: "neutral", -1: "CONTRADICTS"}).values},
                        index=list(SIGNALS))
    card.to_csv(OUT / "market_scorecard.csv")
    print(f"\nScorecard ({last:%b %Y}):\n", card.drop(columns="signal").to_string())
    pd.concat([X.add_prefix("raw_"), V.add_prefix("vote_"), score, comp], axis=1).loc["2000":].round(3).to_csv(
        OUT / "market_signals.csv")
    print(f"\nNet vote score: {score.loc[last]:+.2f} (3 months ago {score.shift(3).loc[last]:+.2f}, "
          f"6 months ago {score.shift(6).loc[last]:+.2f}); continuous composite {comp.loc[last]:+.2f} z")
    print("Signal correlations since 2003 (z, oriented):\n", Z.loc["2003":].corr().round(2))

    # ---- does it matter? (predictive tests)
    ex1, ex3 = returns_panels()
    P = {k: predictive_test(comp, ex3, k) for k in ["DEF-GRW", "CYC-DEF"]}
    for k, p in P.items():
        print(f"\nComposite test {k}: next-3-month return on market composite, {p['start']:%Y-%m} to {p['end']:%Y-%m}, "
              f"n={p['n']}\n  slope {p['slope']:+.2f} pp/yr per 1 z (t={p['slope_t']:.2f}); composite now {p['x_now']:+.2f}"
              f"\n  expected {p['fit']:+.1f} pp/yr (95% CI {p['lo']:+.1f} to {p['hi']:+.1f}), t={p['t']:.2f}"
              f" = average {p['avg']:+.1f} + market tilt {p['tilt']:+.1f} (t={p['tilt_t']:.2f})")
    lab = bucket(score)
    bs = conditional_stats(lab.reindex(ex3.index), ex3[[HEADLINE]], order=BUCKETS, family={HEADLINE: "thesis"})
    bs.to_csv(OUT / "market_buckets.csv", index=False)
    print("\nDEF-GRW next 3 months by net vote bucket (cut-offs ±0.3):\n",
          bs.set_index("state")[["months", "episodes", "raw", "raw_lo", "raw_hi", "raw_t", "hit"]].round(2))
    per = []
    for k in COMPOSITE:
        q = predictive_test(Z[k], ex3, HEADLINE)
        per.append({"signal": k, "slope": round(q["slope"], 2), "t": round(q["slope_t"], 2), "n": q["n"]})
    per = pd.DataFrame(per).set_index("signal")
    per.to_csv(OUT / "market_signal_slopes.csv")
    print("\nIndividual signals: slope of next-3m DEF-GRW on each oriented z (descriptive; 9 tests, expect ~1 false |t|>2):\n", per)

    # ---- plots
    plot_scorecard(X, V, thr, OUT / "market_scorecard.png")
    plot_ratios(OUT / "market_ratios.png")
    plot_history(score, comp, OUT / "market_history.png")
    cur_bucket = lab.loc[last]
    plot_thesis(bs, cur_bucket, P[HEADLINE], OUT / "market_thesis.png")

    # ---- evidence summary
    p = P[HEADLINE]
    n_c, n_x = int((V.loc[last] > 0).sum()), int((V.loc[last] < 0).sum())
    br = bs.set_index("state").reindex([cur_bucket]).iloc[0]
    conf = [k for k in SIGNALS if V.loc[last, k] == 1]
    contra = [k for k in SIGNALS if V.loc[last, k] == -1]
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Market view ({last:%b %Y}): {n_c} of {int(V.loc[last].notna().sum())} signals confirm, {n_x} contradict; "
          f"net score {score.loc[last]:+.2f} -> {cur_bucket}")
    print(f"    confirming: {', '.join(conf) or 'none'}")
    print(f"    contradicting: {', '.join(contra) or 'none'}")
    print(f"  Trend: net score 3 months ago {score.shift(3).loc[last]:+.2f}, now {score.loc[last]:+.2f}")
    print(f"  Is the trade already working? DEF-GRW last 3m {X.loc[last, 'trade_3m']:+.1f}%, last 6m {X.loc[last, 'trade_6m']:+.1f}%")
    if pd.notna(br.get("raw")):
        print(f"  Descriptive: DEF-GRW next 3m when '{cur_bucket}': {br['raw']:+.1f} pp/yr (t={br['raw_t']:.2f}, "
              f"{int(br['episodes'])} episodes)")
    print(f"  HEADLINE composite test: composite now {p['x_now']:+.2f} z; expected DEF-GRW next 3m {p['fit']:+.1f} pp/yr "
          f"(95% CI {p['lo']:+.1f} to {p['hi']:+.1f}), t={p['t']:.2f}")
    print(f"    = sample average {p['avg']:+.1f} + market tilt {p['tilt']:+.1f} pp/yr (t={p['tilt_t']:.2f}; "
          f"slope {p['slope']:+.1f} per z, t={p['slope_t']:.2f})")
    print(f"  Verdict (pre-registered, one-sided 5%, on the expected spread): {thesis_verdict({'raw_t': p['t']})}")
    print(f"  Market-tilt verdict (same rule, on the tilt alone): {thesis_verdict({'raw_t': p['tilt_t']})}")
    print("=" * 90)


def robustness():
    ex1, ex3 = returns_panels()
    X0, thr0 = build_signals()
    base_score = net_score(votes(X0, thr0))
    last = X0.index[X0[COMPOSITE].notna().all(axis=1)][-1]
    rows = []

    def add(name, X, thr, cols=None, start=None):
        V = votes(X, thr)
        sc = net_score(V, list(cols) if cols else list(SIGNALS))
        comp = composite(X, cols)[0]
        if start:
            comp = comp.loc[start:]
        p = predictive_test(comp, ex3)
        rows.append({"spec": name, "net score now": round(sc.loc[last], 2) if pd.notna(sc.loc[last]) else None,
                     "composite now": round(p["x_now"], 2), "n": p["n"], "slope t": round(p["slope_t"], 2),
                     "expected": round(p["fit"], 1), "t": round(p["t"], 2), "verdict": thesis_verdict({"raw_t": p["t"]}),
                     "tilt": round(p["tilt"], 1), "tilt t": round(p["tilt_t"], 2),
                     "tilt verdict": thesis_verdict({"raw_t": p["tilt_t"]})})

    add("baseline", X0, thr0)
    add("thresholds x0.5", X0, {k: (v * 0.5 if v else v) for k, v in thr0.items()})
    add("thresholds x2", X0, {k: (v * 2 if v else v) for k, v in thr0.items()})
    X6, t6 = build_signals(h=6, h_long=12)
    add("6m horizon (12m for trade_6m)", X6, t6)
    X1, t1 = build_signals(h=1, h_long=3)
    add("1m horizon (3m for trade_6m)", X1, t1)
    groups = sorted({SIGNALS[k][0] for k in COMPOSITE})
    for g in groups:
        cols = [k for k in COMPOSITE if SIGNALS[k][0] != g]
        add(f"drop group: {g}", X0, thr0, cols)
    add("market-only (drop the trade's own momentum)", X0, thr0, [k for k in COMPOSITE if k not in ("trade_3m", "trade_6m")])
    add("2010+ sample", X0, thr0, start="2010-01-01")
    Xv = X0.copy()
    add("with VIX term structure (2008+)", Xv, thr0, COMPOSITE + ["vix_term"])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "market_robustness.csv", index=False)
    print("\n", out.astype(object).where(out.notna(), "").to_string(index=False))


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()