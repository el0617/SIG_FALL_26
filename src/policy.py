"""Module 3 - Monetary policy, rates and liquidity (v1).

Run:
    python -m src.policy            # main run: state, rate betas, scenarios, impulse test, plots
    python -m src.policy --robust   # specification robustness table

QUESTION
  Modules 1-2 found one robust macro fact (inflation pressure above the Fed-consistent anchor) and
  no cycle signal for the defensive tilt. Module 3 tests the mechanism that fact implies:
      inflation pressure -> higher real yields / tighter policy -> long-duration growth stocks lag.
  And it checks the trap: utilities and staples are also bond proxies, so rising real yields may
  hurt the DEFENSIVE leg too.

THREE PARTS
  A. Rate sensitivity (contemporaneous, monthly, 2003+): each sector's return vs SPY regressed on
     the change in the 10y REAL yield (TIPS) and the 10y BREAKEVEN (nominal = real + breakeven).
     Both legs of the trade (DEF = XLP/XLU/XLV, GRW = XLK/XLY) and the DEF-GRW spread.
     Stability: rolling 60-month betas and three sub-periods. Translation: scenario table.
  B. Policy and liquidity state: real 10y, real fed funds, 2y minus fed funds (market-implied path),
     Chicago Fed NFCI, net liquidity (Fed balance sheet - Treasury General Account - reverse repo).
  C. PRE-REGISTERED HEADLINE (predictive): "tightening impulse" = average real-time z-score of
     6-month changes in the real 10y yield, the NFCI, and minus net liquidity. Regress next-3-month
     DEF-GRW on it (HAC). Expected spread and cycle-free tilt at today's reading, same verdict rule
     as Modules 1-2 (t >= 1.65 SUPPORTS, <= -1.65 CONTRADICTS, else NEUTRAL).
     A defensive thesis needs: positive slope AND a positive impulse today.
  Plus a descriptive Fed-stance table (hiking / on hold / cutting).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.data import fred, fred_optional, level, monthly, sector_returns, complete_months, current_month_end
from src.utils import chg, lagged, zscore, conditional_stats, episode_table
from src.regime import finalize, EDGE_LOG, thesis_verdict, SECTORS, ROW_NAME, DEFENSIVE, GROWTH, SAMPLE_START
from src.cycle import returns_panels

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ specification (frozen)
BETA_START = "2003-02-28"     # TIPS real yield + breakeven history on FRED
ROLL = 60                     # rolling-beta window (months)
SUBPERIODS = [("2003-2012", "2003-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
              ("2020-2026", "2020-01-01", "2026-12-31")]
IMPULSE_CHG = 6               # months for impulse changes
IMPULSE_COLS = ["real10y", "nfci", "neg_liquidity"]
SHOCKS_BP = [-50, -25, 25, 50]  # 3-month real-yield scenarios
STANCE_BAND = 0.25            # 6m change in fed funds (pp) separating hiking / hold / cutting
STANCES = ["Cutting", "On hold", "Hiking"]
HEADLINE = "DEF-GRW"


# ------------------------------------------------------------------ data
def _billions(s, name):
    """FRED reports some balance-sheet series in millions, others in billions: normalize."""
    s = s.dropna()
    if s.abs().median() > 50_000:
        print(f"  [units] {name}: millions -> billions")
        return s / 1000
    return s


def net_liquidity():
    walcl = _billions(monthly(fred("WALCL")), "WALCL")
    tga = _billions(monthly(fred("WTREGEN")), "WTREGEN")
    rrp_raw = fred_optional("RRPONTSYD")
    idx = walcl.index
    rrp = (_billions(monthly(rrp_raw), "RRPONTSYD").reindex(idx) if rrp_raw is not None
           else pd.Series(0.0, index=idx))
    # No reverse-repo operation in a month = zero outstanding (before 2013 operations were rare tests),
    # so missing months are ZERO, not unknown. Only months after the last observation stay missing.
    last = rrp.last_valid_index()
    rrp = rrp.fillna(0.0).where(rrp.index <= last) if last is not None else rrp.fillna(0.0)
    return (walcl - tga.reindex(idx) - rrp).rename("net_liquidity_bn")


def build_state():
    """Month-end levels of rates, policy and liquidity."""
    real10 = monthly(fred("DFII10"))
    be10 = monthly(fred("T10YIE"))
    nom10 = monthly(fred("DGS10"))
    two = monthly(fred("DGS2"))
    ff = monthly(fred("DFF"), "mean")
    core = lagged(chg(level("PCEPILFE"), 12), 1)            # core PCE y/y, available with a lag
    nfci = monthly(fred("NFCI"))
    usd = fred_optional("DTWEXBGS")
    cols = {
        "real10y": real10, "breakeven10y": be10, "nominal10y": nom10, "two_year": two,
        "fed_funds": ff, "core_pce_yoy": core, "real_fed_funds": ff - core,
        "path_2y_minus_ff": two - ff, "nfci": nfci, "net_liquidity_bn": net_liquidity(),
        "dollar": monthly(usd) if usd is not None else None,
    }
    S = pd.concat({k: v for k, v in cols.items() if v is not None}, axis=1, sort=True)
    return finalize(S, "policy")


def relative_returns(start=BETA_START):
    """Contemporaneous monthly returns (%): each sector minus SPY, the two legs, and the spread."""
    r = complete_months(sector_returns())[["SPY"] + SECTORS] * 100
    out = r[SECTORS].sub(r["SPY"], axis=0)
    out["DEF-SPY"] = r[DEFENSIVE].mean(axis=1, skipna=False) - r["SPY"]
    out["GRW-SPY"] = r[GROWTH].mean(axis=1, skipna=False) - r["SPY"]
    out["DEF-GRW"] = r[DEFENSIVE].mean(axis=1, skipna=False) - r[GROWTH].mean(axis=1, skipna=False)
    return out.loc[start:]


# ------------------------------------------------------------------ A. rate sensitivity
def rate_betas(R, S, cols=("real10y", "breakeven10y"), start=None, end=None):
    """Return (% vs SPY) on monthly changes in yields (pp). Beta = % per +100bp; HAC(3)."""
    dX = S[list(cols)].diff()
    rows = []
    for name in R.columns:
        d = pd.concat([R[name], dX], axis=1, sort=True).loc[start:end].dropna()
        if len(d) < 36:
            continue
        res = sm.OLS(d[name].values, sm.add_constant(d[list(cols)].values)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        b, t, se = np.asarray(res.params), np.asarray(res.tvalues), np.asarray(res.bse)
        row = {"series": name, "n": len(d)}
        for k, c in enumerate(cols, start=1):
            row[f"b_{c}"], row[f"t_{c}"], row[f"se_{c}"] = b[k], t[k], se[k]
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def rolling_beta(R, S, series, col="real10y", window=ROLL):
    d = pd.concat([R[series], S["real10y"].diff(), S["breakeven10y"].diff()], axis=1,
                  keys=["y", "real10y", "breakeven10y"], sort=True).dropna()
    out = {}
    for k in range(window, len(d) + 1):
        w = d.iloc[k - window:k]
        X = np.column_stack([np.ones(window), w["real10y"].values, w["breakeven10y"].values])
        b = np.linalg.lstsq(X, w["y"].values, rcond=None)[0]
        out[d.index[k - 1]] = b[1 if col == "real10y" else 2]
    return pd.Series(out, name=series)


def scenarios(B, S, shocks=SHOCKS_BP):
    """Implied 3-month relative return (%) for a cumulative real-yield move, breakevens unchanged.
    Uses the monthly beta on the cumulative change (assumes no lagged response)."""
    rows = []
    for bp in shocks:
        for s in ["DEF-GRW", "DEF-SPY", "GRW-SPY"]:
            b, se = B.loc[s, "b_real10y"], B.loc[s, "se_real10y"]
            a, c = (b - 1.96 * se) * bp / 100, (b + 1.96 * se) * bp / 100
            rows.append({"shock_bp": bp, "series": s, "implied_3m_pct": b * bp / 100,
                         "lo": min(a, c), "hi": max(a, c)})
    tab = pd.DataFrame(rows)
    d3 = S["real10y"].diff(3).loc[BETA_START:].dropna() * 100
    odds = {bp: float((d3 >= bp).mean() if bp > 0 else (d3 <= bp).mean()) for bp in shocks}
    return tab, odds, d3


# ------------------------------------------------------------------ C. predictive impulse test
def impulse(S, chg_n=IMPULSE_CHG, cols=None, min_periods=36):
    cols = cols or IMPULSE_COLS
    X = pd.DataFrame({
        "real10y": S["real10y"].diff(chg_n),                                  # + = tighter
        "nfci": S["nfci"].diff(chg_n),                                        # + = tighter
        "neg_liquidity": -chg(S["net_liquidity_bn"], chg_n),                  # + = draining
    })[cols]
    Z = zscore(X, window=None, min_periods=min_periods)                       # expanding, real time
    return Z.mean(axis=1).where(Z.notna().all(axis=1)).rename("tightening_impulse"), Z


def predictive_test(x, ex3, series=HEADLINE):
    """Next-3-month spread (annualized pp) on the DEMEANED impulse; HAC(6).
    fit = expected spread today; tilt = fit minus sample average (the policy contribution)."""
    d = pd.concat([ex3[series] * 1200, x], axis=1, keys=["y", "x"], sort=True).dropna()
    mu = d["x"].mean()
    X = np.column_stack([np.ones(len(d)), d["x"].values - mu])
    res = sm.OLS(d["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    b = np.asarray(res.params)
    V = np.asarray(res.cov_params()) if hasattr(res, "cov_params") else np.diag(np.asarray(res.bse) ** 2)
    xnow = x.dropna()
    x0 = np.array([1.0, xnow.iloc[-1] - mu])
    fit, se = float(x0 @ b), float(np.sqrt(x0 @ V @ x0))
    tilt, tse = float(b[1] * x0[1]), float(abs(x0[1]) * np.sqrt(V[1, 1]))
    return dict(slope=float(b[1]), slope_t=float(np.asarray(res.tvalues)[1]), avg=float(b[0]),
                fit=fit, t=fit / se, lo=fit - 1.96 * se, hi=fit + 1.96 * se,
                tilt=tilt, tilt_t=tilt / tse if tse > 0 else np.nan, x_now=float(xnow.iloc[-1]),
                asof=xnow.index[-1], n=len(d), start=d.index[0], end=d.index[-1], series=series)


def fed_stance(S, band=STANCE_BAND):
    ff = level("FEDFUNDS")
    d6 = ff.diff(6)
    lab = np.where(d6 > band, "Hiking", np.where(d6 < -band, "Cutting", "On hold"))
    return pd.Series(lab, index=ff.index, dtype=object).where(d6.notna()).rename("fed_stance")


# ------------------------------------------------------------------ plots
COL_DEF, COL_GRW, COL_SPR = "#2e7d32", "#c62828", "#37474f"


def plot_betas(B, path, per_bp=50):
    order = [s for s in DEFENSIVE + GROWTH if s in B.index] + \
            [s for s in SECTORS if s not in DEFENSIVE + GROWTH and s in B.index] + ["DEF-SPY", "GRW-SPY", "DEF-GRW"]
    b = B.loc[order, "b_real10y"] * per_bp / 100
    ci = 1.96 * B.loc[order, "se_real10y"] * per_bp / 100
    col = [COL_DEF if s in DEFENSIVE or s == "DEF-SPY" else COL_GRW if s in GROWTH or s == "GRW-SPY"
           else COL_SPR if s == "DEF-GRW" else "#90a4ae" for s in order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(order))
    ax.bar(x, b, color=col, alpha=0.85)
    ax.errorbar(x, b, yerr=ci, fmt="none", ecolor="k", capsize=4, lw=1)
    ax.axhline(0, c="k", lw=1)
    for k in range(len(order)):
        ax.text(x[k], (b.iloc[k] + ci.iloc[k] if b.iloc[k] >= 0 else b.iloc[k] - ci.iloc[k]) + (0.08 if b.iloc[k] >= 0 else -0.08),
                f"t={B.loc[order[k], 't_real10y']:.1f}", ha="center", va="bottom" if b.iloc[k] >= 0 else "top", fontsize=8)
    ax.axvline(len(order) - 3.5, c="grey", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([ROW_NAME.get(s, s) for s in order], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Return vs SPY in a month when real 10y yield rises {per_bp}bp (%)")
    ax.set_title(f"Real-yield sensitivity, {B['n'].max()} months since 2003 (breakevens held constant; 95% CI, HAC)\n"
                 "green = defensive leg · red = growth leg · right of dotted line = baskets and the trade spread",
                 fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_rolling(rb, path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for s, c in [("DEF-SPY", COL_DEF), ("GRW-SPY", COL_GRW), ("DEF-GRW", COL_SPR)]:
        ax.plot(rb[s].index, rb[s] / 2, color=c, lw=1.6, label=s)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("% vs benchmark per +50bp real yield")
    ax.set_title(f"Rolling {ROLL}-month real-yield betas: is the relationship stable? "
                 "(DEF-GRW above 0 = rising real yields favour the trade)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_scenarios(tab, odds, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    shocks = sorted(tab.shock_bp.unique())
    w = 0.26
    for k, (s, c) in enumerate([("DEF-SPY", COL_DEF), ("GRW-SPY", COL_GRW), ("DEF-GRW", COL_SPR)]):
        t = tab[tab.series == s].set_index("shock_bp").loc[shocks]
        x = np.arange(len(shocks)) + (k - 1) * w
        ax.bar(x, t["implied_3m_pct"], w, color=c, alpha=0.85, label=s)
        ax.errorbar(x, t["implied_3m_pct"], yerr=[t["implied_3m_pct"] - t["lo"], t["hi"] - t["implied_3m_pct"]],
                    fmt="none", ecolor="k", capsize=3, lw=0.8)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(np.arange(len(shocks)))
    ax.set_xticklabels([f"{bp:+d}bp\n(a move this big in {odds[bp]:.0%}\nof 3-month windows)" for bp in shocks], fontsize=8.5)
    ax.set_ylabel("Implied 3-month return vs benchmark (%)")
    ax.set_title("Scenario translation: 3-month move in the real 10y yield, breakevens unchanged (95% CI)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_dashboard(S, imp, path, start="2003-01-01"):
    s = S.loc[start:]
    fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(s.index, s["real10y"], label="Real 10y (TIPS)"); ax[0].plot(s.index, s["breakeven10y"], label="10y breakeven")
    ax[1].plot(s.index, s["fed_funds"], label="Fed funds"); ax[1].plot(s.index, s["real_fed_funds"], label="Real fed funds (minus core PCE y/y)")
    ax[2].plot(s.index, s["path_2y_minus_ff"], color="#6a1b9a", label="2y minus fed funds (<0 = cuts priced)")
    ax[3].plot(s.index, s["nfci"], color="#ef6c00", label="Chicago Fed NFCI (>0 = tighter than average)")
    ax[4].plot(s.index, s["net_liquidity_bn"] / 1000, color="#00838f", label="Net liquidity, $tn (Fed assets - TGA - RRP)")
    ax4b = ax[4].twinx()
    ax4b.plot(imp.loc[start:].index, imp.loc[start:], color="k", lw=1, alpha=0.6, label="Tightening impulse (z, right)")
    ax4b.axhline(0, c="k", lw=0.6, ls=":")
    for a in ax[:4]:
        a.axhline(0, c="k", lw=0.6); a.legend(fontsize=8, loc="upper left")
    h1, l1 = ax[4].get_legend_handles_labels(); h2, l2 = ax4b.get_legend_handles_labels()
    ax[4].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[0].set_title("Module 3: rates, policy and liquidity state", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("=" * 90, "\nMODULE 3 - MONETARY POLICY, RATES AND LIQUIDITY (v1)")
    S = build_state()
    carried = {k: v for k, v in EDGE_LOG.items() if k.startswith("policy")}
    print(f"  ragged-edge carries: {carried or 'none'}")
    S.loc["1990":].to_csv(OUT / "policy_state.csv")
    last = S.dropna(subset=["real10y"]).index[-1]
    print("\nCurrent state (last 6 months):\n",
          S[["real10y", "breakeven10y", "fed_funds", "real_fed_funds", "path_2y_minus_ff", "nfci",
             "net_liquidity_bn"]].tail(6).round(2))

    # ---- A. rate sensitivity
    R = relative_returns()
    B = rate_betas(R, S)
    B.round(3).to_csv(OUT / "policy_rate_betas.csv")
    print(f"\nRate betas, % return vs SPY per +100bp (monthly, {BETA_START[:7]} on, HAC):\n",
          B[["n", "b_real10y", "t_real10y", "b_breakeven10y", "t_breakeven10y"]].round(2))
    if "dollar" in S:
        Bd = rate_betas(R, S, cols=("real10y", "breakeven10y", "dollar"))
        print("\nWith the trade-weighted dollar as a control (real-yield beta should barely move):\n",
              Bd.loc[["DEF-SPY", "GRW-SPY", "DEF-GRW"], ["n", "b_real10y", "t_real10y", "b_dollar", "t_dollar"]].round(2))
    sub = pd.concat({name: rate_betas(R[["DEF-SPY", "GRW-SPY", "DEF-GRW"]], S, start=a, end=b)[
        ["b_real10y", "t_real10y", "b_breakeven10y", "t_breakeven10y", "n"]] for name, a, b in SUBPERIODS}, axis=1)
    sub.round(2).to_csv(OUT / "policy_betas_subperiods.csv")
    print("\nStability: real-yield and breakeven betas by sub-period (% per +100bp):\n", sub.round(2))
    rb = pd.concat([rolling_beta(R, S, s) for s in ["DEF-SPY", "GRW-SPY", "DEF-GRW"]], axis=1, sort=True)
    rb.round(3).to_csv(OUT / "policy_rolling_betas.csv")
    tab, odds, d3 = scenarios(B, S)
    tab.round(2).to_csv(OUT / "policy_scenarios.csv", index=False)

    # ---- C. headline predictive test
    imp, Zimp = impulse(S)
    print("\nImpulse inputs first available (z): " +
          ", ".join(f"{c} {Zimp[c].first_valid_index():%Y-%m}" for c in Zimp.columns))
    ex1, ex3 = returns_panels()
    P = {k: predictive_test(imp, ex3, k) for k in ["DEF-GRW", "CYC-DEF"]}
    for k, p in P.items():
        print(f"\nImpulse test {k}: next-3-month return on tightening impulse, {p['start']:%Y-%m} to {p['end']:%Y-%m}, "
              f"n={p['n']}\n  slope {p['slope']:+.2f} pp/yr per 1 z (t={p['slope_t']:.2f}); impulse now {p['x_now']:+.2f} "
              f"({p['asof']:%b %Y})\n  expected {p['fit']:+.1f} pp/yr (95% CI {p['lo']:+.1f} to {p['hi']:+.1f}), t={p['t']:.2f}"
              f" = average {p['avg']:+.1f} + policy tilt {p['tilt']:+.1f} (t={p['tilt_t']:.2f})")
    print("\nImpulse components now (z):", Zimp.dropna().iloc[-1].round(2).to_dict())

    # ---- Fed stance (descriptive)
    st = fed_stance(S)
    lab = st.reindex(ex3.index)
    fs = conditional_stats(lab, ex3[["DEF-GRW"]], order=STANCES, family={"DEF-GRW": "thesis"})
    fs.to_csv(OUT / "policy_fed_stance.csv", index=False)
    print("\nDEF-GRW next 3 months by Fed stance (6m change in fed funds, ±25bp):\n",
          fs.set_index("state")[["months", "episodes", "raw", "raw_lo", "raw_hi", "raw_t", "hit"]].round(2))
    cur_stance = st.dropna().iloc[-1]

    # ---- plots
    plot_betas(B, OUT / "policy_rate_betas.png")
    plot_rolling(rb, OUT / "policy_rolling_betas.png")
    plot_scenarios(tab, odds, OUT / "policy_scenarios.png")
    plot_dashboard(S, imp, OUT / "policy_dashboard.png")

    # ---- evidence summary
    p = P[HEADLINE]
    bD, bG, bS = (B.loc[s] for s in ["DEF-SPY", "GRW-SPY", "DEF-GRW"])
    signs = sub.xs("b_real10y", axis=1, level=1).loc["DEF-GRW"]
    be_signs = sub.xs("b_breakeven10y", axis=1, level=1).loc["DEF-GRW"]
    s50 = tab[(tab.shock_bp == 50) & (tab.series == "DEF-GRW")].iloc[0]
    fsr = fs.set_index("state").reindex([cur_stance])
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  State ({last:%b %Y}): real 10y {S.loc[last, 'real10y']:.2f}% (6m chg {S['real10y'].diff(6).loc[last]*100:+.0f}bp), "
          f"breakeven {S.loc[last, 'breakeven10y']:.2f}%, real fed funds {S['real_fed_funds'].dropna().iloc[-1]:+.2f}%, "
          f"2y-FF {S.loc[last, 'path_2y_minus_ff']*100:+.0f}bp, NFCI {S['nfci'].dropna().iloc[-1]:+.2f}, "
          f"net liquidity 6m {chg(S['net_liquidity_bn'], 6).dropna().iloc[-1]:+.1f}%")
    print(f"  Fed stance: {cur_stance}")
    print(f"  Mechanism, per +50bp real yield in a month: GRW vs SPY {bG['b_real10y']/2:+.2f}% (t={bG['t_real10y']:.1f}), "
          f"DEF vs SPY {bD['b_real10y']/2:+.2f}% (t={bD['t_real10y']:.1f}), DEF-GRW {bS['b_real10y']/2:+.2f}% (t={bS['t_real10y']:.1f})")
    print(f"  Mechanism stable? DEF-GRW real-yield beta (per 100bp) by sub-period: "
          + ", ".join(f"{k} {v:+.2f}" for k, v in signs.items()))
    print(f"  Inflation-expectations channel, per +25bp 10y breakeven in a month: DEF-GRW "
          f"{bS['b_breakeven10y']/4:+.2f}% (t={bS['t_breakeven10y']:.1f}); by sub-period (per 100bp): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in be_signs.items()))
    print(f"  Scenario: real 10y +50bp over 3 months -> DEF-GRW {s50['implied_3m_pct']:+.1f}% "
          f"(95% CI {s50['lo']:+.1f} to {s50['hi']:+.1f}); a rise that large happened in {odds[50]:.0%} of 3-month windows since 2003")
    print(f"  HEADLINE impulse test: tightening impulse now {p['x_now']:+.2f} z; expected DEF-GRW next 3m {p['fit']:+.1f} pp/yr "
          f"(95% CI {p['lo']:+.1f} to {p['hi']:+.1f}), t={p['t']:.2f}")
    print(f"    = sample average {p['avg']:+.1f} + policy tilt {p['tilt']:+.1f} pp/yr (t={p['tilt_t']:.2f}; slope "
          f"{p['slope']:+.1f} per z, t={p['slope_t']:.2f})")
    print(f"  Verdict (pre-registered, one-sided 5%, on the expected spread): {thesis_verdict({'raw_t': p['t']})}")
    print(f"  Policy-tilt verdict (same rule, on the tilt alone): {thesis_verdict({'raw_t': p['tilt_t']})}")
    if len(fsr.dropna(subset=['raw'])):
        r = fsr.iloc[0]
        print(f"  Descriptive: DEF-GRW next 3m when Fed is '{cur_stance}': {r['raw']:+.1f} pp/yr, t={r['raw_t']:.2f}, "
              f"{int(r['episodes'])} episodes")
    print("=" * 90)


def robustness():
    S = build_state()
    R = relative_returns()
    ex1, ex3 = returns_panels()
    rows = []

    def add(name, B=None, p=None):
        row = {"spec": name}
        if B is not None:
            if "b_breakeven10y" in B:
                row["DEF-GRW be beta/25bp"] = round(B.loc["DEF-GRW", "b_breakeven10y"] / 4, 2)
            row.update({"DEF-GRW beta/50bp": round(B.loc["DEF-GRW", "b_real10y"] / 2, 2),
                        "t beta": round(B.loc["DEF-GRW", "t_real10y"], 2),
                        "GRW beta/50bp": round(B.loc["GRW-SPY", "b_real10y"] / 2, 2),
                        "DEF beta/50bp": round(B.loc["DEF-SPY", "b_real10y"] / 2, 2)})
        if p is not None:
            row.update({"impulse now": round(p["x_now"], 2), "expected": round(p["fit"], 1), "t": round(p["t"], 2),
                        "verdict": thesis_verdict({"raw_t": p["t"]}), "tilt": round(p["tilt"], 1),
                        "tilt t": round(p["tilt_t"], 2), "tilt verdict": thesis_verdict({"raw_t": p["tilt_t"]})})
        rows.append(row)

    base_imp = impulse(S)[0]
    add("baseline", rate_betas(R, S), predictive_test(base_imp, ex3))
    add("nominal 10y instead of real+breakeven", rate_betas(R, S, cols=("nominal10y",)).rename(
        columns={"b_nominal10y": "b_real10y", "t_nominal10y": "t_real10y"}), None)
    if "dollar" in S:
        add("+ dollar control (2006+)", rate_betas(R, S, cols=("real10y", "breakeven10y", "dollar")), None)
    add("betas excl. 2020", rate_betas(R.drop(R.loc["2020"].index), S), None)
    add("betas 2013+ only", rate_betas(R, S, start="2013-01-01"), None)
    add("betas 2020+ only", rate_betas(R, S, start="2020-01-01"), None)
    add("impulse: 3m changes", None, predictive_test(impulse(S, chg_n=3)[0], ex3))
    add("impulse: 12m changes", None, predictive_test(impulse(S, chg_n=12)[0], ex3))
    for c in IMPULSE_COLS:
        print(f"  running: drop {c}")
        add(f"impulse: drop {c}", None, predictive_test(impulse(S, cols=[x for x in IMPULSE_COLS if x != c])[0], ex3))
    add("impulse: real 10y only", None, predictive_test(impulse(S, cols=["real10y"])[0], ex3))
    p16 = predictive_test(base_imp.loc["2016-07-01":], ex3)
    add("impulse: 2016+ sample only (pre-fix window)", None, p16)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "policy_robustness.csv", index=False)
    print("\n", out.astype(object).where(out.notna(), "").to_string(index=False))


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()