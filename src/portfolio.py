"""Module 6 - Portfolio risk, scenarios and candidate weightings (v2).

Run:
    python -m src.portfolio            # all candidates: risk, scenarios, insurance, plots
    python -m src.portfolio --robust   # sizing curve for the recommended trades + window checks

INPUT  data/portfolio.csv (created on first run from the Bloomberg PORT screen of 05/29/26):
       ticker, weight (%, current holdings), benchmark (% S&P 500 GICS sector weight).
       Benchmark = IVV GICS sector weights from Bloomberg PORT (% End Wgt Bmrk, 10/21/26); the 0.26%
       "not classified" sleeve is dropped and the rest rescaled to 100.

v2 (code review): risk contributions are now an Euler split of ACTIVE risk on active weights
(portfolio - benchmark) and the covariance of total ETF returns, the way Bloomberg PORT reports CTEV.
v1 split tracking error vs SPY on HOLDING weights and SPY-relative returns, which inflated the
overweights (XLE showed 56%) and hid the growth underweight. Check vs PORT (05/29 weights, 60 months):
VGT 54% vs Bloomberg IT 53%; XLE 19% vs 12% (history includes the 2021-22 energy swings).
Tracking error vs SPY, beta, VaR/CVaR and scenarios are unchanged.

WHAT IT COMPUTES for each candidate portfolio (benchmark = SPY total return)
  - weights drifted from 05/29/26 to the latest complete month with actual ETF returns
  - beta, volatility, tracking error (60-month and EWMA), relative VaR/CVaR (95%, 1 and 3 months)
  - each position's share of tracking error
  - scenario returns vs SPY built from Module 3/4 sensitivities (post-2020, post-2020 ex-2022,
    full-sample breakeven), and market beta
  - insurance math vs SPY (carry, protection, breakeven drawdown probability), as in Module 4
CANDIDATES are defined as TRADES (percentage-point changes) applied to the drifted current weights,
each trade tied to a module finding. See CANDIDATES below.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.data import RAW, sector_returns, complete_months
from src.policy import build_state

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
PFILE = Path("data/portfolio.csv")
ASOF = "2026-05-29"          # PORT snapshot date
WINDOW = 60                  # months for covariance, beta, TE
EWMA_LAMBDA = 0.97
DRAWDOWN = -5.0
ERP = 5.0
HOLDINGS = ["VGT", "XLY", "XLC", "XLF", "XLI", "XLB", "XLE", "XLP", "XLV", "XLU", "SCHH"]
PROXY = {"VGT": "XLK", "SCHH": "IYR"}          # used before the ETF existed
RISK_LEGS = {"Growth underweight (VGT, XLY)": ["VGT", "XLY"], "Communication services (XLC)": ["XLC"],
             "Energy and materials (XLE, XLB)": ["XLE", "XLB"], "Defensive overweight (XLP, XLV, XLU)": ["XLP", "XLV", "XLU"],
             "Other (XLF, XLI, SCHH)": ["XLF", "XLI", "SCHH"]}     # same grouping as the scorecard's Bloomberg legs
GROUPS = {"Growth (VGT, XLY)": ["VGT", "XLY"], "Defensive (XLP, XLV, XLU)": ["XLP", "XLV", "XLU"],
          "Inflation hedge (XLE, XLB)": ["XLE", "XLB"], "Real-yield beneficiary (XLF)": ["XLF"],
          "Other (XLC, XLI, SCHH)": ["XLC", "XLI", "SCHH"]}
DEFAULT = pd.DataFrame({
    "ticker":    HOLDINGS,
    # Bloomberg PORT 'TOPDOWN2', 05/29/26
    "weight":    [28.43, 6.39, 7.73, 10.56, 5.59, 6.39, 8.61, 10.60, 11.28, 2.45, 1.97],
    # IVV GICS sector weights, Bloomberg PORT 10/21/26 (% End Wgt Bmrk); 0.26% not classified dropped
    "benchmark": [38.27, 9.01, 9.40, 11.80, 8.01, 1.67, 3.72, 4.74, 9.69, 1.97, 1.71],
})

# Trades in percentage points, applied to drifted current weights. Each must net to zero.
CANDIDATES = {
    "Current (drifted)": {},
    # the thesis as first framed: more defensives, less growth
    "Team direction: add defensives": {"VGT": -4, "XLY": -1, "XLP": +2, "XLV": +2, "XLU": +1},
    # Module 3/4: staples carry the largest breakeven risk (t=-4.8) and highest beta cost; financials have the
    # strongest positive real-yield beta (t=3.7) and the portfolio is underweight them
    "Recommended core: fix the defensive leg": {"XLP": -3, "XLF": +3},
    # conditional step: only if real yields keep rising with breakevens stable (see triggers)
    "Core + extend growth underweight": {"XLP": -3, "XLF": +3, "VGT": -2, "XLV": +1, "XLI": +1},
}
RECOMMENDED = "Recommended core: fix the defensive leg"
SCENARIOS = {   # name: (real-yield bp, breakeven bp, equity %, which real-yield beta)
    "Real yields +50bp (post-2020 betas)":         (50, 0, 0, "post"),
    "Real yields +50bp (post-2020 ex-2022)":       (50, 0, 0, "ex22"),
    "Breakevens +25bp (inflation scare)":          (0, 25, 0, "full"),
    "Stagflation: breakevens +25bp, real +25bp":   (25, 25, 0, "post"),
    "Equity -10% (risk-off)":                      (0, 0, -10, "post"),
    "Growth scare: equity -10%, real yields -50bp": (-50, 0, -10, "post"),
}


# ------------------------------------------------------------------ data
def load_portfolio():
    if not PFILE.exists():
        PFILE.parent.mkdir(exist_ok=True)
        DEFAULT.to_csv(PFILE, index=False)
        print(f"  created {PFILE} from the 05/29/26 PORT screen and 10/21/26 IVV sector weights")
    p = pd.read_csv(PFILE).set_index("ticker")
    for c in ("weight", "benchmark"):
        p[c] = p[c] / p[c].sum() * 100
    return p


def etf_returns(refresh=False):
    """Monthly total returns (%) for the 11 holdings and SPY; VGT/SCHH spliced with proxies pre-inception."""
    f = RAW / "holdings_prices.csv"
    if f.exists() and not refresh:
        px = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        px = yf.download(["VGT", "SCHH"], start="2003-12-01", auto_adjust=True, progress=False)["Close"]
        px = px.resample("ME").last()
        px.to_csv(f)
    base = complete_months(sector_returns()) * 100
    own = (px / px.shift(1) - 1) * 100
    R = pd.DataFrame(index=base.index)
    for t in HOLDINGS + ["SPY"]:
        if t in PROXY:
            s = own[t].reindex(base.index) if t in own else pd.Series(np.nan, index=base.index)
            R[t] = s.combine_first(base[PROXY[t]])
        else:
            R[t] = base[t]
    return R


def drift(w, R, start=ASOF):
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)    # first full month after snapshot
    Rm = R.loc[first:, HOLDINGS]
    d = w * (1 + Rm / 100).prod()
    return d / d.sum() * 100, Rm.index


def apply_trades(w, trades):
    assert abs(sum(trades.values())) < 1e-9, "trades must net to zero"
    n = w.copy()
    for t, dw in trades.items():
        n[t] = n[t] + dw
    if (n < 0).any():
        raise ValueError(f"negative weight after trades: {n[n < 0].to_dict()}")
    return n


# ------------------------------------------------------------------ sensitivities
def etf_sensitivities(R, S):
    """Per-ETF: market beta (60m), and sensitivities of (ETF - SPY) to real yield and breakeven changes."""
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    dX = S[["real10y", "breakeven10y"]].diff()
    out = {}
    last = R.index[-WINDOW:]
    for t in HOLDINGS:
        row = {}
        d = pd.concat([R[t], R["SPY"]], axis=1, keys=["y", "m"]).loc[last].dropna()
        row["beta"] = float(np.polyfit(d["m"], d["y"], 1)[0])
        for tag, start, drop22 in [("full", "2003-02-28", False), ("post", "2020-01-01", False), ("ex22", "2020-01-01", True)]:
            dd = pd.concat([rel[t], dX], axis=1, sort=True).loc[start:].dropna()
            if drop22:
                dd = dd.drop(dd.loc["2022"].index)
            if len(dd) < 36:
                row[f"real_{tag}"] = row[f"be_{tag}"] = np.nan
                continue
            res = sm.OLS(dd[t].values, sm.add_constant(dd[["real10y", "breakeven10y"]].values)).fit()
            row[f"real_{tag}"], row[f"be_{tag}"] = res.params[1], res.params[2]     # % per +100bp
        out[t] = row
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ portfolio analytics
def analytics(name, w, R, sens, bench):
    a = w / 100
    x = (w - bench.reindex(w.index)) / 100          # active weights; sum to zero
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    port_rel_hist = (rel.fillna(0) * a).sum(axis=1).where(rel.notna().all(axis=1))
    last = R.index[-WINDOW:]
    C = rel.loc[last].cov()
    te = float(np.sqrt(a @ C.values @ a * 12))
    ew = rel.loc[last].copy()
    wts = EWMA_LAMBDA ** np.arange(len(ew))[::-1]
    Cw = np.cov(ew.values.T, aweights=wts)
    te_ewma = float(np.sqrt(a @ Cw @ a * 12))
    # Euler split of active risk on active weights, on the covariance of TOTAL ETF returns (as Bloomberg PORT).
    # Because x sums to zero, x'Cx is the same with total or SPY-relative returns, but the split is not:
    # on SPY-relative returns each share shifts by x_i * cov(SPY, active return), which inflated overweights
    # (XLE, XLP, XLB) and shrank the VGT underweight in a low-beta active position.
    xv = x.values
    Ct = R[HOLDINGS].loc[last].cov().values
    te_bench = float(np.sqrt(xv @ Ct @ xv * 12))
    rc = xv * (Ct @ xv) / (xv @ Ct @ xv)
    port_abs = (R[HOLDINGS].loc[last] * a).sum(axis=1)
    beta = float(np.polyfit(R["SPY"].loc[last], port_abs, 1)[0])
    vol = float(port_abs.std() * np.sqrt(12))
    h = port_rel_hist.dropna()
    h3 = h.rolling(3).sum().dropna()
    var1, var3 = -np.percentile(h, 5), -np.percentile(h3, 5)
    cvar1, cvar3 = -h[h <= -var1].mean(), -h3[h3 <= -var3].mean()
    # insurance math vs SPY (3-month windows, full history)
    spy3 = ((1 + R["SPY"] / 100).rolling(3).apply(np.prod, raw=True) - 1) * 100
    d = pd.concat([h3, spy3], axis=1, keys=["x", "s"], sort=True).dropna()
    dd = d["s"] <= DRAWDOWN
    carry, prot = d.loc[~dd, "x"].mean(), d.loc[dd, "x"].mean()
    pstar = 0.0 if carry >= 0 and prot >= 0 else (np.nan if prot <= carry else -carry / (prot - carry))
    scen = {}
    for s, (real, be, eq, tag) in SCENARIOS.items():
        x = (a * (sens[f"real_{tag}"] * real / 100 + sens["be_full" if tag == "full" or be else f"be_{tag}"] * be / 100
                  + (sens["beta"] - 1) * eq)).sum()
        scen[s] = float(x)
    m = dict(portfolio=name, beta=beta, vol=vol, te=te, te_bench=te_bench, te_ewma=te_ewma, var1=var1, cvar1=cvar1, var3=var3,
             cvar3=cvar3, beta_cost=(beta - 1) * ERP, carry_3m=carry, protection_3m=prot, p_star=pstar,
             history_from=h.index.min())
    m.update(scen)
    return m, pd.Series(rc * 100, index=HOLDINGS, name=name)


def leg_shares(rc):
    """Active-risk shares (%) summed by thesis leg."""
    return pd.Series({g: rc[c].sum() for g, c in RISK_LEGS.items()})


def group_weights(w):
    return pd.Series({g: w[c].sum() for g, c in GROUPS.items()})


# ------------------------------------------------------------------ plots
def plot_weights(W, bench, path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = list(W.columns)
    x = np.arange(len(HOLDINGS)); bw = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(x + (k - (len(names) - 1) / 2) * bw, W[n] - bench, bw, label=n)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(HOLDINGS)
    ax.set_ylabel("Active weight vs S&P 500 (pp)")
    ax.set_title("Active weights by candidate (benchmark: IVV GICS sector weights, Bloomberg 10/21/26)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_scenarios(M, path):
    S = M[list(SCENARIOS)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(SCENARIOS)); bw = 0.8 / len(S)
    for k, (n, row) in enumerate(S.iterrows()):
        ax.bar(x + (k - (len(S) - 1) / 2) * bw, row.values, bw, label=n)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(": ", ":\n").replace(" (", "\n(") for s in SCENARIOS], fontsize=8)
    ax.set_ylabel("Return vs SPY (%)")
    ax.set_title("Scenario returns relative to SPY (sensitivities from Modules 3-4; one-month shock)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_risk(RC, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(HOLDINGS)); bw = 0.8 / RC.shape[1]
    for k, n in enumerate(RC.columns):
        ax.bar(x + (k - (RC.shape[1] - 1) / 2) * bw, RC[n], bw, label=n)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(HOLDINGS)
    ax.set_ylabel("Share of active risk (%)")
    ax.set_title(f"Where the active risk comes from (Euler split on active weights, {WINDOW}-month covariance)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_corr(R, path):
    last = R.index[-WINDOW:]
    C = R.loc[last, HOLDINGS + ["SPY"]].corr()
    stress = R.loc[R["SPY"] <= -3, HOLDINGS + ["SPY"]].corr()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, M, ttl in [(axes[0], C, f"Last {WINDOW} months"),
                       (axes[1], stress, f"Stress months (SPY ≤ -3%, n={int((R['SPY'] <= -3).sum())})")]:
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8)
        for i in range(len(M)):
            for j in range(len(M)):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Correlation of monthly returns", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def build_all(R, sens, p, candidates=CANDIDATES):
    w0, months = drift(p["weight"], R)
    W, M, RC = {}, [], []
    for name, trades in candidates.items():
        w = apply_trades(w0, trades)
        W[name] = w
        m, rc = analytics(name, w, R, sens, p["benchmark"])
        M.append(m); RC.append(rc)
    return pd.DataFrame(W), pd.DataFrame(M).set_index("portfolio"), pd.concat(RC, axis=1), w0, months


def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=" * 90, "\nMODULE 6 - PORTFOLIO RISK, SCENARIOS AND CANDIDATE WEIGHTINGS (v1)")
    p = load_portfolio()
    R = etf_returns()
    S = build_state()
    sens = etf_sensitivities(R, S)
    sens.round(2).to_csv(OUT / "portfolio_etf_sensitivities.csv")
    print("\nETF sensitivities (beta: 60m vs SPY; real/be: % vs SPY per +100bp):\n", sens.round(2))

    W, M, RC, w0, months = build_all(R, sens, p)
    print(f"\nWeights drifted from {ASOF} through {months[-1]:%b %Y} ({len(months)} months of returns):")
    print(pd.concat([p["weight"].rename("05/29/26"), w0.rename("drifted"), p["benchmark"].rename("benchmark (IVV)"),
                     (w0 - p["benchmark"]).rename("active")], axis=1).round(2))
    W.round(2).to_csv(OUT / "portfolio_candidate_weights.csv")
    print("\nCandidate weights (%):\n", W.round(2))
    G = pd.DataFrame({n: group_weights(W[n]) for n in W}).assign(benchmark=group_weights(p["benchmark"]))
    print("\nGroup weights (%):\n", G.round(1))

    M.round(3).to_csv(OUT / "portfolio_metrics.csv")
    risk_cols = ["beta", "vol", "te", "te_bench", "te_ewma", "var1", "cvar1", "var3", "cvar3", "beta_cost", "carry_3m",
                 "protection_3m", "p_star"]
    print("\nRisk vs SPY (annualized %, VaR/CVaR = 95% relative loss vs SPY over 1 and 3 months; "
          "te_bench = tracking error vs the sector-weighted benchmark):\n", M[risk_cols].round(2))
    print("\nScenario returns vs SPY (%):\n", M[list(SCENARIOS)].T.round(2))
    RC.round(1).to_csv(OUT / "portfolio_risk_contributions.csv")
    print("\nShare of active risk by position (%; Euler split on active weights, sums to 100):\n", RC.round(1))
    L = pd.DataFrame({n: leg_shares(RC[n]) for n in RC})
    L.round(1).to_csv(OUT / "portfolio_risk_legs.csv")
    print("\nShare of active risk by thesis leg (%):\n", L.round(1))

    plot_weights(W, p["benchmark"], OUT / "portfolio_active_weights.png")
    plot_scenarios(M, OUT / "portfolio_scenarios.png")
    plot_risk(RC, OUT / "portfolio_risk_contributions.png")
    plot_corr(R, OUT / "portfolio_correlations.png")

    c, r = M.loc["Current (drifted)"], M.loc[RECOMMENDED]
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (current vs recommended)")
    cur_rc = RC["Current (drifted)"].sort_values(key=abs, ascending=False)
    print("  Largest active-risk positions (current): " + ", ".join(f"{t} {v:.0f}%" for t, v in cur_rc.head(4).items()))
    for k, lab, fmt in [("te", "Tracking error vs SPY", "{:.2f}%"), ("te_bench", "Tracking error vs benchmark", "{:.2f}%"),
                        ("beta", "Beta vs SPY", "{:.2f}"),
                        ("beta_cost", f"Beta cost at {ERP:.0f}% ERP", "{:+.2f} pp/yr"),
                        ("cvar3", "3-month CVaR vs SPY", "{:.2f}%"), ("p_star", "Breakeven drawdown probability", "{:.0%}")]:
        print(f"  {lab:32s} current {fmt.format(c[k]) if pd.notna(c[k]) else 'n/a':>12s}   recommended "
              f"{fmt.format(r[k]) if pd.notna(r[k]) else 'n/a':>12s}")
    for s in SCENARIOS:
        print(f"  {s:46s} current {c[s]:+.2f}%   recommended {r[s]:+.2f}%")
    print("=" * 90)


def robustness():
    p = load_portfolio()
    R = etf_returns()
    S = build_state()
    sens = etf_sensitivities(R, S)
    base = CANDIDATES[RECOMMENDED]
    cands = {"Current (drifted)": {}}
    for k in [0.5, 1.0, 1.5, 2.0]:
        cands[f"Recommended x{k}"] = {t: v * k for t, v in base.items()}
    W, M, RC, w0, _ = build_all(R, sens, p, cands)
    print("\nSizing curve for the recommended trades (how risk and exposures scale):\n",
          M[["te", "te_bench", "beta", "beta_cost", "cvar3"] + list(SCENARIOS)].round(2).to_string())
    M.round(3).to_csv(OUT / "portfolio_sizing.csv")


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()