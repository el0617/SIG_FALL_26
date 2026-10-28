"""Module 11 - Sector tear sheets: one page per holding (v1).

Run (after Modules 1 and 7 have been run if their sections should appear):
    python -m src.tearsheets           # outputs/tearsheets.pdf (summary page + one page per holding)
                                       # and outputs/tearsheets_summary.csv

EACH PAGE
  Position     current (drifted) weight, IVV benchmark weight, active weight, after the recommended trade
  Risk         share of active risk (Euler split on active weights, 60 months; as Module 6), beta and
               volatility (60 months), worst 3-month return vs SPY since 2003
  Performance  return vs SPY over 1, 3, 6 and 12 months and since the 05/29/26 snapshot
  Rates        return vs SPY per +100bp in the real 10y and the 10y breakeven (monthly, 2003+, HAC t)
  Scenarios    average 3-month return vs SPY in each Module 8 scenario
  Regime       Module 1's edge for this sector in today's regime (if outputs/regime_stats_long.csv exists)
  Valuation    Module 7's forward P/E, percentiles and revisions (if the Bloomberg files exist)
  Flags        generated from the numbers by fixed rules (below), never written by hand

FLAG RULES (fixed before the first run)
  active risk share >= 15%          "Large share of active risk"
  |breakeven beta t| >= 2           "Sensitive to inflation expectations (+/-)"
  |real-yield beta t| >= 2          "Sensitive to real yields (+/-)"
  forward P/E percentile >= 80      "Expensive vs its own history";  <= 20 "Cheap vs its own history"
  3-month EPS revision <= 0         "Earnings estimates falling"
  |return vs SPY, 3 months| >= 8%   "Large recent move vs SPY"
Holdings use their own history; before launch VGT uses XLK and SCHH uses IYR (as Module 6). XLC starts mid-2018.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from pathlib import Path

from src.portfolio import load_portfolio, etf_returns, drift, apply_trades, CANDIDATES, RECOMMENDED, HOLDINGS, WINDOW, GROUPS, ASOF
from src.policy import build_state

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
START = "2003-02-28"
NAME = {"VGT": "Information technology", "XLY": "Consumer discretionary", "XLC": "Communication services",
        "XLF": "Financials", "XLI": "Industrials", "XLB": "Materials", "XLE": "Energy", "XLP": "Consumer staples",
        "XLV": "Health care", "XLU": "Utilities", "SCHH": "Real estate"}
PROXY = {"VGT": "XLK", "SCHH": "IYR"}        # names used by Modules 1 and 7


# ------------------------------------------------------------------ numbers
def compute():
    R = etf_returns()
    S = build_state()
    p = load_portfolio()
    w0, months = drift(p["weight"], R)
    bench = p["benchmark"]
    w_rec = apply_trades(w0, CANDIDATES[RECOMMENDED])
    last = R.index[-WINDOW:]
    C = R[HOLDINGS].loc[last].cov().values

    def euler(w):
        x = ((w - bench) / 100).reindex(HOLDINGS).values
        return pd.Series(x * (C @ x) / (x @ C @ x) * 100, index=HOLDINGS)

    rc, rc_rec = euler(w0), euler(w_rec)
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    cum = lambda x, n: ((1 + x.iloc[-n:] / 100).prod() - 1) * 100
    dX = S[["real10y", "breakeven10y"]].diff().reindex(R.index)
    from src.scenarios import windows, classify, BASE as SB, SCEN
    W = windows(R, S, SB["start"]); lab = classify(W, SB)
    g3 = lambda x: ((1 + x / 100).rolling(3).apply(np.prod, raw=True) - 1).shift(-3) * 100
    x3 = lambda x: ((1 + x / 100).rolling(3).apply(np.prod, raw=True) - 1) * 100
    rows, scen = {}, {}
    since = R.index[R.index > pd.Timestamp(ASOF) + pd.offsets.MonthEnd(0)]     # first full month after the snapshot (as Module 6)
    for t in HOLDINGS:
        d = pd.concat([R[t], R["SPY"]], axis=1, keys=["y", "m"]).loc[last].dropna()
        beta = float(np.polyfit(d["m"], d["y"], 1)[0])
        dd = pd.concat([rel[t], dX], axis=1).loc[START:].dropna()
        res = sm.OLS(dd[t].values, sm.add_constant(dd[["real10y", "breakeven10y"]].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        b, tt = np.asarray(res.params), np.asarray(res.tvalues)
        rr = (x3(R[t]) - x3(R["SPY"])).loc[START:]
        rows[t] = {
            "sector": NAME[t], "group": next((g for g, c in GROUPS.items() if t in c), ""),
            "weight": w0[t], "benchmark": bench[t], "active": w0[t] - bench[t], "active_after_trade": w_rec[t] - bench[t],
            "risk_share": rc[t], "risk_share_after_trade": rc_rec[t],
            "beta_60m": beta, "vol_60m": float(d["y"].std() * np.sqrt(12)),
            "rel_1m": cum(R[t], 1) - cum(R["SPY"], 1), "rel_3m": cum(R[t], 3) - cum(R["SPY"], 3),
            "rel_6m": cum(R[t], 6) - cum(R["SPY"], 6), "rel_12m": cum(R[t], 12) - cum(R["SPY"], 12),
            "rel_since_0529": (((1 + R.loc[since, t] / 100).prod()) - ((1 + R.loc[since, "SPY"] / 100).prod())) * 100,
            "worst_3m_rel": float(rr.min()), "worst_3m_date": rr.idxmin(),
            "real_beta": b[1], "real_t": tt[1], "be_beta": b[2], "be_t": tt[2], "n_rates": len(dd),
        }
        y = (g3(R[t]) - g3(R["SPY"])).reindex(W.index)
        scen[t] = y.groupby(lab).mean().reindex(SCEN)
    T = pd.DataFrame(rows).T
    return T, pd.DataFrame(scen), R, rel, months


def regime_context(T):
    f, s = OUT / "regime_stats_long.csv", OUT / "regime_state.csv"
    if not (f.exists() and s.exists()):
        return T, None
    st = pd.read_csv(f)
    reg = pd.read_csv(s, index_col=0, parse_dates=True)["regime"].dropna()
    now = reg.iloc[-1]
    for t in T.index:
        r = st[(st.series == PROXY.get(t, t)) & (st.state == now)]
        T.loc[t, "regime_edge"] = float(r["edge"].iloc[0]) if len(r) else np.nan
        T.loc[t, "regime_t"] = float(r["t"].iloc[0]) if len(r) else np.nan
    return T, now


def valuation_context(T):
    try:
        from src.valuation import load, sector_table
        V = sector_table(load())
    except Exception as e:
        print(f"  [tearsheets] valuation section skipped (Bloomberg files needed): {str(e)[:60]}")
        return T, False
    for t in T.index:
        k = PROXY.get(t, t)
        if k in V.index:
            for c in ["fwd_pe", "pe_pct", "rel_pct", "rev3", "rev12", "dy"]:
                T.loc[t, c] = V.loc[k, c]
    return T, True


def flags(r):
    f = []
    if r["risk_share"] >= 15:
        f.append(f"Large share of active risk ({r['risk_share']:.0f}%)")
    if abs(r["be_t"]) >= 2:
        f.append(f"Sensitive to inflation expectations ({r['be_beta']:+.1f}% per 100bp, t {r['be_t']:.1f})")
    if abs(r["real_t"]) >= 2:
        f.append(f"Sensitive to real yields ({r['real_beta']:+.1f}% per 100bp, t {r['real_t']:.1f})")
    if pd.notna(r.get("pe_pct")):
        if r["pe_pct"] >= 80:
            f.append(f"Expensive vs its own history ({r['pe_pct']:.0f}th percentile)")
        elif r["pe_pct"] <= 20:
            f.append(f"Cheap vs its own history ({r['pe_pct']:.0f}th percentile)")
    if pd.notna(r.get("rev3")) and r["rev3"] <= 0:
        f.append(f"Earnings estimates falling ({r['rev3']:+.1f}% in 3 months)")
    if abs(r["rel_3m"]) >= 8:
        f.append(f"Large recent move vs SPY ({r['rel_3m']:+.1f}% in 3 months)")
    return f


# ------------------------------------------------------------------ pages
def _kv(ax, items, title):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    y = 0.92
    for k, v in items:
        ax.text(0.0, y, k, fontsize=8.5, color="0.35", transform=ax.transAxes)
        ax.text(1.0, y, v, fontsize=8.5, ha="right", transform=ax.transAxes)
        y -= 0.085


def page(pdf, t, r, scen, rel, now_regime, have_val, asof):
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 1.25, 1.1], hspace=0.55, wspace=0.35)
    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.75, f"{t}  ·  {r['sector']}", fontsize=17, fontweight="bold")
    head.text(0, 0.30, f"{r['group']}  ·  active {r['active']:+.2f}pp ({r['weight']:.2f}% vs benchmark {r['benchmark']:.2f}%)  ·  "
              f"{r['risk_share']:.0f}% of active risk  ·  data through {asof:%b %Y}", fontsize=10, color="0.3")
    fl = flags(r)
    head.text(0, -0.15, ("Flags: " + "; ".join(fl)) if fl else "Flags: none", fontsize=9, color="#c62828" if fl else "0.4")

    a = fig.add_subplot(gs[1, :2])
    s = rel[t].loc[rel.index[-60]:].fillna(0)
    c = ((1 + s / 100).cumprod() - 1) * 100
    a.plot(c.index, c, color="#1565c0", lw=1.6)
    a.axhline(0, c="k", lw=0.6)
    a.axvline(pd.Timestamp(ASOF), color="0.5", ls="--", lw=0.8)
    a.set_title("Cumulative monthly return minus SPY, last 5 years (%); dashed = 05/29/26 snapshot", fontsize=9)
    a.grid(alpha=0.3)

    b = fig.add_subplot(gs[1, 2])
    v = scen[t]
    b.barh(range(len(v))[::-1], v.values, color=["#2e7d32" if x > 0 else "#c62828" for x in v.values])
    b.set_yticks(range(len(v))[::-1]); b.set_yticklabels([s.replace("Stagflationary", "Stagfl.") for s in v.index], fontsize=8)
    b.axvline(0, c="k", lw=0.6)
    b.set_title("Module 8 scenarios: avg 3m return vs SPY (%)", fontsize=9)

    _kv(fig.add_subplot(gs[2, 0]), [
        ("Weight after XLP→XLF", f"{r['weight'] + r['active_after_trade'] - r['active']:.2f}%"),
        ("Active after trade", f"{r['active_after_trade']:+.2f}pp"),
        ("Risk share after trade", f"{r['risk_share_after_trade']:.0f}%"),
        ("Beta (60m)", f"{r['beta_60m']:.2f}"), ("Volatility (60m)", f"{r['vol_60m']:.1f}%"),
        ("Worst 3m vs SPY", f"{r['worst_3m_rel']:+.1f}% ({pd.Timestamp(r['worst_3m_date']):%b %Y})")], "Position and risk")
    _kv(fig.add_subplot(gs[2, 1]), [
        ("1 month", f"{r['rel_1m']:+.1f}%"), ("3 months", f"{r['rel_3m']:+.1f}%"), ("6 months", f"{r['rel_6m']:+.1f}%"),
        ("12 months", f"{r['rel_12m']:+.1f}%"), ("Since 05/29/26", f"{r['rel_since_0529']:+.1f}%"),
        ("Real 10y, per +100bp", f"{r['real_beta']:+.1f}% (t {r['real_t']:.1f})"),
        ("Breakeven, per +100bp", f"{r['be_beta']:+.1f}% (t {r['be_t']:.1f})")], "Vs SPY: performance and rates")
    items = []
    if now_regime is not None and pd.notna(r.get("regime_edge")):
        items.append((f"Edge in {now_regime} (M1)", f"{r['regime_edge']:+.1f} pp/yr (t {r['regime_t']:.1f})"))
    if have_val and pd.notna(r.get("fwd_pe")):
        items += [("Forward P/E", f"{r['fwd_pe']:.1f}"), ("Percentile, own history", f"{r['pe_pct']:.0f}th"),
                  ("Percentile vs S&P 500", "—" if pd.isna(r.get("rel_pct")) else f"{r['rel_pct']:.0f}th"),
                  ("EPS revision 3m / 12m", f"{r['rev3']:+.1f}% / {r['rev12']:+.1f}%"), ("Dividend yield", f"{r['dy']:.2f}%")]
    if not items:
        items = [("Regime and valuation", "run Modules 1 and 7 first")]
    _kv(fig.add_subplot(gs[2, 2]), items, "Regime and valuation")
    fig.text(0.01, 0.01, "TOPDOWN2 sector tear sheet · Modules 1, 3, 6, 7, 8 · history, not a forecast; Bloomberg-derived "
             "figures are licensed, internal use only", fontsize=7, color="0.5")
    pdf.savefig(fig); plt.close(fig)


def summary_page(pdf, T, asof):
    fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
    ax.set_title(f"TOPDOWN2 sector tear sheets: summary (data through {asof:%b %Y})", loc="left", fontsize=14, fontweight="bold")
    cols = ["weight", "benchmark", "active", "risk_share", "rel_3m", "rel_since_0529", "be_beta", "real_beta"]
    head = ["Weight", "Bmrk", "Active", "Risk %", "3m vs SPY", "Since 5/29", "BE beta", "Real beta"]
    cell = [[f"{T.loc[t, c]:+.1f}" if c in ("active", "rel_3m", "rel_since_0529", "be_beta", "real_beta") else f"{T.loc[t, c]:.1f}"
             for c in cols] + [str(len(flags(T.loc[t])))] for t in T.index]
    cell = [[f"{t}  {T.loc[t, 'sector']}"] + row for t, row in zip(T.index, cell)]
    tb = ax.table(cellText=cell, colLabels=["Holding"] + head + ["Flags"], loc="upper left", bbox=[0.0, 0.25, 1.0, 0.65],
                  colWidths=[0.26] + [0.08] * len(head) + [0.06], cellLoc="right")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5)
    for (r_, c_), cl in tb.get_celld().items():
        if c_ == 0:
            cl.set_text_props(ha="left")
    ax.text(0, 0.15, "Risk % = share of active risk (Euler on active weights, 60 months). BE / Real beta = return vs SPY per +100bp in the\n"
            "10y breakeven / real 10y, monthly since 2003. Flags follow fixed rules listed in src/tearsheets.py.", fontsize=8.5, color="0.35")
    pdf.savefig(fig); plt.close(fig)


def main():
    print("=" * 90 + "\nMODULE 11 - SECTOR TEAR SHEETS\n" + "=" * 90)
    T, scen, R, rel, months = compute()
    T, now_regime = regime_context(T)
    T, have_val = valuation_context(T)
    asof = R.index[-1]
    order = T.sort_values("active").index
    with PdfPages(OUT / "tearsheets.pdf") as pdf:
        summary_page(pdf, T.loc[order], asof)
        for t in order:
            page(pdf, t, T.loc[t], scen, rel, now_regime, have_val, asof)
    T["flags"] = [" | ".join(flags(T.loc[t])) for t in T.index]
    T.to_csv(OUT / "tearsheets_summary.csv")
    scen.T.round(3).to_csv(OUT / "tearsheets_scenarios.csv")
    show = T.loc[order, ["weight", "benchmark", "active", "risk_share", "rel_3m", "rel_since_0529", "be_beta", "be_t", "real_beta", "real_t"]]
    print(show.astype(float).round(2).to_string())
    print("\nFlags:")
    for t in order:
        print(f"  {t:5s} " + (T.loc[t, "flags"] or "none"))
    print(f"\n  wrote outputs/tearsheets.pdf ({len(order) + 1} pages) and outputs/tearsheets_summary.csv")
    if have_val:
        print("  NOTE: the PDF contains Bloomberg-derived valuation figures: keep it out of git and inside the team.")


if __name__ == "__main__":
    main()
