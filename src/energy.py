"""Module 10 - Energy deep dive: what the XLE overweight is, what drives it, what it costs (v1).

Run:
    python -m src.energy            # drivers, diversification, stress, scenarios, sizing, plots
    python -m src.energy --robust   # driver betas and correlations on other windows

WHY
  Energy is TOPDOWN2's largest overweight (about +5.6pp vs IVV), its largest contributor since 05/29
  (Bloomberg attribution +1.11pp) and about 22% of active risk (Module 6; Bloomberg 12%). The thesis case
  for it is indirect: energy hedges the inflation channel that hurts the defensive leg (Modules 3-4).
  This module tests that case and states what the position would cost if oil breaks.

PARTS
  A. Drivers: XLE minus SPY (monthly %, 2003+) on oil (% change in WTI), 10y breakeven and real-yield
     changes (pp), SPY return and the dollar (%). HAC(3). Full sample and post-2020.
  B. Diversification: correlation of XLE's active return with the growth underweight, the defensive
     overweight and the rest of the portfolio's active return (60 months and full sample; SPY-down months).
  C. Stress: what the current XLE active weight would have cost in the oil crashes (2008, 2014-16, 2020, 2022 H2)
     and in the worst 3-month windows.
  D. Scenarios: XLE vs SPY in each Module 8 scenario (same rules), times today's active weight.
  E. Sizing: tracking error and energy's share of active risk for XLE active weights 0 to +8pp
     (funded from, or returned to, the other holdings in proportion to their weights).
  F. Valuation and revisions from Module 7 (if the Bloomberg files exist) and today's oil state.

PRE-REGISTERED CHECKLIST (written 2026-10-28, before any output was seen; do not change after seeing results)
  The overweight is JUSTIFIED BY THE THESIS if all four hold, PARTLY if two or three hold, NOT if fewer:
    1 Inflation hedge   XLE's breakeven beta > 0 with t >= 1.65 (full sample)
    2 Diversifier       correlation of XLE's active return with the growth underweight's <= 0.30 (60 months)
    3 Scenario payoff   XLE beats SPY on average in both the inflation-scare and stagflationary scenarios
    4 Tail cost         the current overweight would have cost <= 1.0pp of active return in the worst oil crash
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.data import fred_optional, monthly, RAW
from src.policy import build_state
from src.portfolio import load_portfolio, etf_returns, drift, HOLDINGS, WINDOW

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
START = "2003-02-28"
POST = "2020-01-01"
CRASHES = {"2008 crisis (Jul 2008 - Feb 2009)": ("2008-07-31", "2009-02-28"),
           "Oil glut (Jul 2014 - Jan 2016)": ("2014-07-31", "2016-01-31"),
           "Covid (Jan - Apr 2020)": ("2020-01-31", "2020-04-30"),
           "Post-peak (Jun - Dec 2022)": ("2022-06-30", "2022-12-31")}
TAIL_LIMIT = 1.0
CORR_LIMIT = 0.30
SIZES = np.arange(0.0, 8.01, 1.0)


# ------------------------------------------------------------------ data
def oil():
    s = fred_optional("DCOILWTICO")
    if s is None:
        return None
    s = s.where(s > 0)                  # April 2020 printed a negative price; month-end level is what we use
    return monthly(s)


def panel():
    R = etf_returns()
    S = build_state()
    o = oil()
    D = pd.DataFrame({
        "xle_rel": R["XLE"] - R["SPY"],
        "vgt_rel": R["VGT"] - R["SPY"],
        "grw_rel": R[["VGT", "XLY"]].mean(axis=1) - R["SPY"],
        "def_rel": R[["XLP", "XLV", "XLU"]].mean(axis=1) - R["SPY"],
        "spy": R["SPY"],
        "oil": o.reindex(R.index).pct_change(fill_method=None) * 100 if o is not None else np.nan,
        "breakeven": S["breakeven10y"].reindex(R.index).diff(),
        "real": S["real10y"].reindex(R.index).diff(),
        "dollar": S["dollar"].reindex(R.index).pct_change(fill_method=None) * 100 if "dollar" in S else np.nan,
    })
    return R, S, D, o


def active_weights(R):
    p = load_portfolio()
    w0, _ = drift(p["weight"], R)
    return w0, p["benchmark"], (w0 - p["benchmark"])


# ------------------------------------------------------------------ A. drivers
def drivers(D, start=START, end=None, cols=("oil", "breakeven", "real", "spy", "dollar")):
    cols = [c for c in cols if D[c].notna().sum() > 36]
    d = D[["xle_rel"] + cols].loc[start:end].dropna()
    res = sm.OLS(d["xle_rel"].values, sm.add_constant(d[cols].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    out = pd.DataFrame({"beta": np.asarray(res.params)[1:], "t": np.asarray(res.tvalues)[1:]}, index=cols)
    return out, dict(n=len(d), r2=float(res.rsquared) * 100, start=d.index[0], end=d.index[-1])


# ------------------------------------------------------------------ B. diversification
def diversification(R, D, a):
    """Correlations of XLE's active contribution with the other active bets (today's active weights on history)."""
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    rel["XLC"] = rel["XLC"].fillna(0.0)
    contrib = rel * (a / 100)
    legs = pd.DataFrame({
        "energy (XLE)": contrib["XLE"],
        "growth underweight (VGT, XLY)": contrib[["VGT", "XLY"]].sum(axis=1),
        "defensive overweight (XLP, XLV, XLU)": contrib[["XLP", "XLV", "XLU"]].sum(axis=1),
        "rest of portfolio (ex XLE)": contrib.drop(columns="XLE").sum(axis=1),
    }).dropna()
    rows = {}
    for lab, sl in [("last 60 months", legs.iloc[-WINDOW:]), ("since 2003", legs.loc[START:]),
                    ("SPY-down months since 2003", legs.loc[START:][R["SPY"].reindex(legs.loc[START:].index) < 0])]:
        rows[lab] = sl.corr()["energy (XLE)"].drop("energy (XLE)")
    return pd.DataFrame(rows), legs


# ------------------------------------------------------------------ C. stress
def stress(R, a_xle):
    rel = R["XLE"] - R["SPY"]
    rows = []
    for name, (s, e) in CRASHES.items():
        r = rel.loc[pd.Timestamp(s) + pd.offsets.MonthEnd(1):e]       # months after the start month-end
        cum = ((1 + R["XLE"].loc[r.index] / 100).prod() - (1 + R["SPY"].loc[r.index] / 100).prod()) * 100
        rows.append({"episode": name, "months": len(r), "XLE minus SPY (%)": cum,
                     "cost of today's overweight (pp)": cum * a_xle / 100})
    x3 = (((1 + R["XLE"] / 100).rolling(3).apply(np.prod, raw=True)) - ((1 + R["SPY"] / 100).rolling(3).apply(np.prod, raw=True))) * 100
    worst = x3.loc[START:].nsmallest(5)
    return pd.DataFrame(rows).set_index("episode"), worst, x3


# ------------------------------------------------------------------ D. scenarios (Module 8 rules)
def scenario_payoffs(R, S, a_xle):
    from src.scenarios import windows, classify, BASE as SBASE, SCEN
    W = windows(R, S, SBASE["start"])
    lab = classify(W, SBASE)
    g = lambda x: ((1 + x / 100).rolling(3).apply(np.prod, raw=True) - 1).shift(-3) * 100
    x = (g(R["XLE"]) - g(R["SPY"])).reindex(W.index)
    t = pd.DataFrame({"windows": lab.value_counts().reindex(SCEN), "XLE minus SPY, 3m (%)": x.groupby(lab).mean().reindex(SCEN),
                      "XLE beat SPY (% of windows)": (x > 0).groupby(lab).mean().reindex(SCEN) * 100})
    t["portfolio impact of overweight (pp)"] = t["XLE minus SPY, 3m (%)"] * a_xle / 100
    return t


# ------------------------------------------------------------------ E. sizing
def sizing(R, w0, bench, sizes=SIZES):
    """XLE active weight from 0 to +8pp; the difference is funded pro rata from the other holdings."""
    last = R.index[-WINDOW:]
    C = R[HOLDINGS].loc[last].cov().values
    rows = []
    for s in sizes:
        w = w0.copy()
        target = bench["XLE"] + s
        delta = target - w["XLE"]
        others = [h for h in HOLDINGS if h != "XLE"]
        w[others] = w[others] - delta * w[others] / w[others].sum()
        w["XLE"] = target
        x = ((w - bench) / 100).reindex(HOLDINGS).values
        var = x @ C @ x
        rc = x * (C @ x) / var
        rows.append({"XLE active (pp)": s, "tracking error (%)": np.sqrt(var * 12),
                     "energy share of active risk (%)": rc[HOLDINGS.index("XLE")] * 100})
    return pd.DataFrame(rows).set_index("XLE active (pp)")


# ------------------------------------------------------------------ F. valuation / oil state
def valuation_context():
    try:
        from src.valuation import load, sector_table
        T = sector_table(load())
        return T.loc[["XLE"], ["fwd_pe", "pe_pct", "rel_pct", "rev3", "rev12", "dy"]]
    except Exception as e:
        print(f"  [energy] Module 7 valuation unavailable (Bloomberg files needed): {str(e)[:60]}")
        return None


def oil_state(o):
    if o is None:
        return None
    o = o.dropna()
    now = o.iloc[-1]
    return dict(asof=o.index[-1], level=now, chg6=(now / o.iloc[-7] - 1) * 100, chg12=(now / o.iloc[-13] - 1) * 100,
                pct=(o.loc[START:] < now).mean() * 100)


# ------------------------------------------------------------------ plots
def plot_energy(D, legs, sz, x3, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    a = axes[0, 0]
    roll = legs["energy (XLE)"].rolling(36).corr(legs["growth underweight (VGT, XLY)"])
    a.plot(roll.index, roll, color="#e65100", lw=1.6)
    a.axhline(CORR_LIMIT, color="k", ls="--", lw=0.8); a.axhline(0, color="k", lw=0.6)
    a.set_title("Energy vs growth-underweight active returns: 36-month rolling correlation", fontsize=10)
    a = axes[0, 1]
    d = D[["xle_rel", "breakeven"]].loc[START:].dropna()
    a.scatter(d["breakeven"] * 100, d["xle_rel"], s=10, alpha=0.5, color="#e65100")
    a.axhline(0, c="k", lw=0.6); a.axvline(0, c="k", lw=0.6)
    a.set_xlabel("Monthly change in 10y breakeven (bp)"); a.set_ylabel("XLE minus SPY (%)")
    a.set_title("Energy and inflation expectations, monthly since 2003", fontsize=10)
    a = axes[1, 0]
    a.plot(sz.index, sz["tracking error (%)"], marker="o", color="#37474f", label="tracking error")
    a2 = a.twinx()
    a2.plot(sz.index, sz["energy share of active risk (%)"], marker="s", color="#e65100", label="energy share")
    a.set_xlabel("XLE active weight (pp)"); a.set_ylabel("tracking error (%)"); a2.set_ylabel("energy share of active risk (%)")
    a.set_title("Sizing: risk as the XLE overweight changes (others funded pro rata)", fontsize=10)
    a = axes[1, 1]
    s = x3.loc[START:]
    a.fill_between(s.index, s, 0, where=s < 0, color="#c62828", alpha=0.5)
    a.fill_between(s.index, s, 0, where=s >= 0, color="#2e7d32", alpha=0.5)
    for _, (st, en) in CRASHES.items():
        a.axvspan(pd.Timestamp(st), pd.Timestamp(en), color="0.85", zorder=0)
    a.set_title("XLE minus SPY, rolling 3 months (%); grey = oil-crash episodes", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ run
def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("=" * 90 + "\nMODULE 10 - ENERGY DEEP DIVE\n" + "=" * 90)
    R, S, D, o = panel()
    w0, bench, a = active_weights(R)
    a_xle = float(a["XLE"])
    print(f"Position: XLE {w0['XLE']:.2f}% vs benchmark {bench['XLE']:.2f}% -> active {a_xle:+.2f}pp "
          f"(drifted to {R.index[-1]:%b %Y})")
    if o is None:
        print("  WARNING: WTI oil price (FRED DCOILWTICO) unavailable; oil driver skipped")

    B, info = drivers(D)
    Bp, infop = drivers(D, start=POST)
    print(f"\nA. What moves XLE vs SPY (monthly %, HAC). Oil per +10%, rates per +100bp, SPY per +1%, dollar per +1%:")
    show = pd.DataFrame({"since 2003": B["beta"], "t": B["t"], "since 2020": Bp["beta"], "t ": Bp["t"]})
    if "oil" in show.index:
        show.loc["oil", ["since 2003", "since 2020"]] *= 10
    print(show.round(2).to_string())
    print(f"  R2 {info['r2']:.0f}% since 2003 (n={info['n']}), {infop['r2']:.0f}% since 2020 (n={infop['n']})")
    show.round(3).to_csv(OUT / "energy_drivers.csv")

    corr, legs = diversification(R, D, a)
    print("\nB. Correlation of energy's active return with the other bets (today's weights on history):\n", corr.round(2).to_string())
    corr.round(3).to_csv(OUT / "energy_diversification.csv")

    st, worst, x3 = stress(R, a_xle)
    print("\nC. Oil-crash stress (cumulative):\n", st.round(2).to_string())
    print("  Worst 3-month XLE minus SPY since 2003 (%): " + ", ".join(f"{d:%Y-%m} {v:+.1f}" for d, v in worst.items()))
    st.round(3).to_csv(OUT / "energy_stress.csv")

    sc = scenario_payoffs(R, S, a_xle)
    print("\nD. Module 8 scenarios:\n", sc.round(2).to_string())
    sc.round(3).to_csv(OUT / "energy_scenarios.csv")

    sz = sizing(R, w0, bench)
    print("\nE. Sizing (60-month covariance):\n", sz.round(2).to_string())
    sz.round(3).to_csv(OUT / "energy_sizing.csv")

    val, os_ = valuation_context(), oil_state(o)
    if val is not None:
        print("\nF. Valuation and revisions (Module 7):\n", val.round(1).to_string())
    if os_ is not None:
        print(f"  WTI {os_['level']:.1f} on {os_['asof']:%Y-%m}: {os_['chg6']:+.0f}% over 6 months, {os_['chg12']:+.0f}% over 12; "
              f"{os_['pct']:.0f}th percentile since 2003")
    plot_energy(D, legs, sz, x3, OUT / "energy.png")

    # ---- pre-registered checklist
    c1 = "breakeven" in B.index and B.loc["breakeven", "beta"] > 0 and B.loc["breakeven", "t"] >= 1.65
    c2 = corr.loc["growth underweight (VGT, XLY)", "last 60 months"] <= CORR_LIMIT
    c3 = bool((sc.loc[["Inflation scare", "Stagflationary sell-off"], "XLE minus SPY, 3m (%)"] > 0).all())
    worst_cost = float(st["cost of today's overweight (pp)"].min())
    c4 = worst_cost >= -TAIL_LIMIT
    n = sum([c1, c2, c3, c4])
    verdict = "JUSTIFIED BY THE THESIS" if n == 4 else "PARTLY JUSTIFIED" if n >= 2 else "NOT JUSTIFIED"
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Position: XLE active {a_xle:+.2f}pp; {sz.loc[sz.index[np.argmin(abs(sz.index - round(a_xle)))], 'energy share of active risk (%)']:.0f}% "
          f"of active risk at about that size")
    be = B.loc["breakeven"] if "breakeven" in B.index else None
    print(f"  1 Inflation hedge: breakeven beta {be['beta']:+.2f}% per 100bp (t={be['t']:.1f}) -> {'PASS' if c1 else 'FAIL'}")
    print(f"  2 Diversifier: corr with growth underweight {corr.loc['growth underweight (VGT, XLY)', 'last 60 months']:+.2f} "
          f"(60m; limit {CORR_LIMIT:+.2f}) -> {'PASS' if c2 else 'FAIL'}")
    print(f"  3 Scenario payoff: inflation scare {sc.loc['Inflation scare', 'XLE minus SPY, 3m (%)']:+.1f}%, stagflationary "
          f"sell-off {sc.loc['Stagflationary sell-off', 'XLE minus SPY, 3m (%)']:+.1f}% -> {'PASS' if c3 else 'FAIL'}")
    print(f"  4 Tail cost: worst oil crash would cost {worst_cost:+.2f}pp at today's overweight (limit -{TAIL_LIMIT:.1f}) -> "
          f"{'PASS' if c4 else 'FAIL'}")
    print(f"  Verdict (pre-registered, {n}/4): {verdict}")
    if "oil" in B.index:
        print(f"  Oil is the main driver: +10% oil -> XLE {B.loc['oil', 'beta'] * 10:+.1f}% vs SPY (t={B.loc['oil', 't']:.1f})")
    print("=" * 90)


def robustness():
    R, S, D, o = panel()
    w0, bench, a = active_weights(R)
    rows = []
    for name, s, e in [("since 2003", START, None), ("2003-2012", START, "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
                       ("since 2020", POST, None), ("ex 2020", START, None)]:
        DD = D.drop(D.loc["2020"].index) if name == "ex 2020" else D
        B, info = drivers(DD, start=s, end=e)
        row = {"window": name, "n": info["n"], "R2": round(info["r2"])}
        for c in B.index:
            row[f"{c} beta"] = round(B.loc[c, "beta"] * (10 if c == "oil" else 1), 2)
            row[f"{c} t"] = round(B.loc[c, "t"], 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "energy_robustness.csv", index=False)
    print("\nDriver betas by window (oil per +10%):\n", out.to_string(index=False))
    corr, legs = diversification(R, D, a)
    roll = legs["energy (XLE)"].rolling(36).corr(legs["growth underweight (VGT, XLY)"]).dropna()
    print(f"\n36-month correlation with the growth underweight: now {roll.iloc[-1]:+.2f}, range {roll.min():+.2f} to {roll.max():+.2f}, "
          f"above {CORR_LIMIT:+.2f} in {(roll > CORR_LIMIT).mean():.0%} of windows")


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()
