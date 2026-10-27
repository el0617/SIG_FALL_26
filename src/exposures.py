"""Module 6b - Factor model of the portfolio's active risk (extension of Module 6).

Run:
    python -m src.exposures            # 60-month window: exposures, risk decomposition, plots
    python -m src.exposures --robust   # same on other windows (36m, since 2007)

QUESTION
  Strip away the sector labels: what is TOPDOWN2 actually betting on, relative to SPY?
  Are the active bets the ones the thesis argues for (intended), or something else (unintended)?

HOW
  Active return of each candidate portfolio = sum of weights x (ETF return - SPY return), using
  TODAY's weights over the past window (a historical simulation of the current positioning).
  It is regressed on two separate sets of factors (reported separately, because market beta and
  real yields move together and would trade off in one regression):
    Style lens  Ken French: market, size, value, profitability, investment, momentum
    Macro lens  SPY return (beta control), change in real 10y yield, change in 10y breakeven,
                % change in the trade-weighted dollar
  Tracking error is split into each factor's contribution (Euler decomposition: exposure x
  covariance) plus a sector-specific remainder the factors do not explain.

  This is a DESCRIPTIVE risk tool. It produces no verdict and does not forecast factor returns.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.portfolio import (load_portfolio, etf_returns, drift, apply_trades, CANDIDATES, HOLDINGS,
                           RECOMMENDED, WINDOW)
from src.factors import french_factors
from src.policy import build_state

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

STYLE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
MACRO = ["Market", "Real yield", "Breakeven", "Dollar"]
LABEL = {"Mkt-RF": "Market", "SMB": "Size (small)", "HML": "Value", "RMW": "Profitability",
         "CMA": "Investment (conservative)", "Mom": "Momentum",
         "Market": "Market (SPY)", "Real yield": "Real 10y, per +100bp", "Breakeven": "Breakeven, per +100bp",
         "Dollar": "Dollar, per +1%"}
# Which exposures the thesis argues for. Everything else is a side effect to watch.
INTENDED = {
    "Mkt-RF": "intended: lower beta is part of the defensive tilt",
    "SMB": "unintended", "HML": "unintended: watch (value lagged after real-yield jumps)",
    "RMW": "consistent with the defensive leg (quality)", "CMA": "consistent with the defensive leg",
    "Mom": "unintended",
    "Market": "intended: lower beta", "Real yield": "intended: growth underweight should gain when real yields rise",
    "Breakeven": "intended: energy overweight should gain when breakevens rise", "Dollar": "unintended",
}
ORDER = ["Current (drifted)", "Team direction: add defensives", RECOMMENDED, "Core + extend growth underweight"]


# ------------------------------------------------------------------ data
def active_returns(R, W):
    """Monthly active return (%) of each candidate vs SPY, today's weights applied to history.
    XLC (launched June 2018) is treated as matching SPY before launch; this only matters for windows
    reaching back before mid-2018 and is reported."""
    rel = R[HOLDINGS].sub(R["SPY"], axis=0)
    xlc_filled = rel["XLC"].isna() & rel.drop(columns="XLC").notna().all(axis=1)
    rel = rel.fillna({"XLC": 0.0})
    ok = rel.notna().all(axis=1) & R["SPY"].notna()
    A = pd.DataFrame({n: (rel * (W[n] / 100)).sum(axis=1).where(ok) for n in W.columns})
    return A, xlc_filled


def macro_factors(R, S):
    usd = S["dollar"] if "dollar" in S else pd.Series(np.nan, index=S.index)
    return pd.DataFrame({
        "Market": R["SPY"],
        "Real yield": S["real10y"].diff(),
        "Breakeven": S["breakeven10y"].diff(),
        "Dollar": usd.pct_change(fill_method=None) * 100,
    })


# ------------------------------------------------------------------ model
def decompose(a, X, cols, window=WINDOW, start=None):
    """Regress active returns on factors; split tracking error into factor contributions + specific."""
    d = pd.concat([a.rename("a"), X[cols]], axis=1, sort=True).dropna()
    d = d.loc[start:] if start else d.iloc[-window:]
    y, F = d["a"].values, d[cols].values
    res = sm.OLS(y, sm.add_constant(F)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    b = np.asarray(res.params)[1:]
    se, t = np.asarray(res.bse)[1:], np.asarray(res.tvalues)[1:]
    C = np.cov(F.T, ddof=1)
    resid = y - np.asarray(res.params)[0] - F @ b
    var_tot = float(np.var(y, ddof=1))
    contrib = b * (C @ b)                                  # Euler contributions to variance
    var_spec = float(np.var(resid, ddof=1))
    out = pd.DataFrame({"exposure": b, "se": se, "t": t,
                        "risk_share": contrib / var_tot * 100}, index=cols)
    summary = dict(n=len(d), start=d.index[0], end=d.index[-1], te=np.sqrt(var_tot * 12),
                   te_factor=np.sqrt(max(float(b @ C @ b), 0) * 12), te_specific=np.sqrt(var_spec * 12),
                   factor_share=float(b @ C @ b) / var_tot * 100, specific_share=var_spec / var_tot * 100,
                   r2=float(res.rsquared) * 100, alpha_ann=float(np.asarray(res.params)[0]) * 12)
    return out, summary


def run(A, X_style, X_macro, window=WINDOW, start=None):
    rows, sums = [], []
    for name in A.columns:
        for lens, X, cols in [("style", X_style, STYLE), ("macro", X_macro, MACRO)]:
            if X is None:
                continue
            ex, s = decompose(A[name], X, cols, window, start)
            ex = ex.assign(portfolio=name, lens=lens, factor=ex.index)
            rows.append(ex)
            sums.append(dict(portfolio=name, lens=lens, **s))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sums)


# ------------------------------------------------------------------ plots
C_INT, C_UNI, C_MUT = "#2e7d32", "#c62828", "#90a4ae"


def plot_exposures(E, path, names=("Current (drifted)", RECOMMENDED)):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, lens, cols in [(axes[0], "style", STYLE), (axes[1], "macro", MACRO)]:
        sub = E[E.lens == lens]
        if sub.empty:
            ax.set_visible(False); continue
        y = np.arange(len(cols))[::-1]
        h = 0.36
        for k, n in enumerate(names):
            e = sub[sub.portfolio == n].set_index("factor").reindex(cols)
            colors = [(C_UNI if INTENDED[c].startswith("unintended") else C_INT) for c in cols]
            ax.barh(y + (h / 2 if k == 0 else -h / 2), e["exposure"], h,
                    color=colors, alpha=0.45 if k == 0 else 0.95, edgecolor="k" if k == 1 else "none",
                    label=n.replace("Recommended core: fix the defensive leg", "Recommended (XLP → XLF)"))
            ax.errorbar(e["exposure"], y + (h / 2 if k == 0 else -h / 2), xerr=1.96 * e["se"],
                        fmt="none", ecolor="k", capsize=2, lw=0.8)
        ax.axvline(0, c="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in cols], fontsize=9)
        ax.set_title(("Style lens (Ken French factors)" if lens == "style" else
                      "Macro lens (market, rates, dollar)"), fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Active factor exposures vs SPY (light = current, dark = recommended; 95% CI)\n"
                 "green = intended by the thesis or consistent with it · red = unintended", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_risk(E, Sm, path, lens="style"):
    cols = STYLE if lens == "style" else MACRO
    names = [n for n in ORDER if n in set(Sm.portfolio)]
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    y = np.arange(len(names))[::-1]
    for i, n in enumerate(names):
        s = Sm[(Sm.portfolio == n) & (Sm.lens == lens)].iloc[0]
        e = E[(E.portfolio == n) & (E.lens == lens)].set_index("factor").reindex(cols)
        pos, neg = 0.0, 0.0
        for k, c in enumerate(cols):
            v = e.loc[c, "risk_share"]
            if v >= 0:
                ax.barh(y[i], v, left=pos, color=cmap(k), label=LABEL[c] if i == 0 else None); pos += v
            else:
                ax.barh(y[i], v, left=neg, color=cmap(k), label=LABEL[c] if i == 0 else None); neg += v
        ax.barh(y[i], s["specific_share"], left=pos, color="#cfd8dc", hatch="//", edgecolor="#90a4ae",
                label="Sector-specific (not explained)" if i == 0 else None)
        ax.text(pos + s["specific_share"] + 1, y[i], f"TE {s['te']:.2f}%", va="center", fontsize=8.5)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("Recommended core: fix the defensive leg", "Recommended (XLP → XLF)")
                                          for n in names], fontsize=9)
    ax.set_xlabel("Share of tracking-error variance (%); negative = the factor offsets other bets")
    ax.set_title(f"Where active risk comes from ({lens} lens): factor bets vs sector-specific risk", fontsize=10)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def setup():
    p = load_portfolio()
    R = etf_returns()
    S = build_state()
    w0, _ = drift(p["weight"], R)
    W = pd.DataFrame({n: apply_trades(w0, t) for n, t in CANDIDATES.items()})
    A, xlc_filled = active_returns(R, W)
    ff = french_factors()
    Xs = ff[STYLE] if ff is not None else None
    Xm = macro_factors(R, S)
    return A, Xs, Xm, xlc_filled


def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=" * 90, "\nMODULE 6b - FACTOR MODEL OF ACTIVE RISK")
    A, Xs, Xm, xlc_filled = setup()
    if Xs is None:
        print("  Ken French factors unavailable: style lens skipped (see Module 4 for the manual download)")
    E, Sm = run(A, Xs, Xm)
    E.to_csv(OUT / "exposures_long.csv", index=False)
    Sm.to_csv(OUT / "exposures_summary.csv", index=False)
    s0 = Sm.iloc[0]
    print(f"  window: {WINDOW} months ending {s0['end']:%b %Y} (style lens ends when the French data ends)")

    for lens, cols in [("style", STYLE), ("macro", MACRO)]:
        sub = E[E.lens == lens]
        if sub.empty:
            continue
        print(f"\n{lens.upper()} LENS: active exposures (t-stat in brackets)")
        tab = pd.DataFrame({n: [f"{r.exposure:+.3f} ({r.t:+.1f})" for r in sub[sub.portfolio == n].set_index("factor").reindex(cols).itertuples()]
                            for n in ORDER}, index=[LABEL[c] for c in cols])
        print(tab.to_string())
        rs = pd.DataFrame({n: sub[sub.portfolio == n].set_index("factor").reindex(cols)["risk_share"].values for n in ORDER},
                          index=[LABEL[c] for c in cols])
        sm = Sm[Sm.lens == lens].set_index("portfolio").reindex(ORDER)
        rs.loc["Sector-specific (not explained)"] = sm["specific_share"].values
        print(f"\n{lens.upper()} LENS: share of tracking-error variance (%)\n", rs.round(1).to_string())
        print(f"  tracking error: " + ", ".join(f"{n.split(':')[0]} {sm.loc[n, 'te']:.2f}% "
                                                  f"(factors {sm.loc[n, 'factor_share']:.0f}%)" for n in ORDER))

    plot_exposures(E, OUT / "exposures_factors.png")
    for lens in ("style", "macro"):
        if not E[E.lens == lens].empty:
            plot_risk(E, Sm, OUT / f"exposures_risk_{lens}.png", lens)

    # ---- evidence summary
    cur, rec = "Current (drifted)", RECOMMENDED
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (what TOPDOWN2 is betting on)")
    for lens in ("style", "macro"):
        sub = E[(E.lens == lens) & (E.portfolio == cur)].set_index("factor")
        if sub.empty:
            continue
        sm = Sm[(Sm.lens == lens) & (Sm.portfolio == cur)].iloc[0]
        top = sub["risk_share"].abs().sort_values(ascending=False).index[:3]
        print(f"  {lens} lens: factors explain {sm['factor_share']:.0f}% of active risk; largest: " +
              ", ".join(f"{LABEL[f]} {sub.loc[f, 'risk_share']:+.0f}%" for f in top))
        flags = [f for f in sub.index if INTENDED[f].startswith("unintended") and abs(sub.loc[f, "t"]) >= 2]
        print(f"    unintended exposures with |t| >= 2: " +
              (", ".join(f"{LABEL[f]} {sub.loc[f, 'exposure']:+.3f} (t={sub.loc[f, 't']:.1f})" for f in flags) or "none"))
        for f in [f for f in ("Real yield", "Breakeven") if f in sub.index]:
            e = sub.loc[f]
            print(f"    {LABEL[f]}: {e['exposure']:+.2f}% (t={e['t']:.1f}) -> "
                  f"{'as intended' if e['exposure'] > 0 else 'opposite to the thesis'}"
                  f"{'' if abs(e['t']) >= 1.65 else ' (not significant)'}")
    for lens, f in [("style", "HML"), ("macro", "Real yield"), ("macro", "Breakeven"), ("style", "Mkt-RF")]:
        a = E[(E.lens == lens) & (E.portfolio == cur) & (E.factor == f)]
        b = E[(E.lens == lens) & (E.portfolio == rec) & (E.factor == f)]
        if len(a) and len(b):
            print(f"  Recommended trade changes {LABEL[f]} exposure {a.exposure.iloc[0]:+.3f} -> {b.exposure.iloc[0]:+.3f}")
    print("=" * 90)


def robustness():
    A, Xs, Xm, xlc_filled = setup()
    rows = []
    for name, kw in [("60 months", {"window": 60}), ("36 months", {"window": 36}), ("since 2007", {"start": "2007-01-01"})]:
        E, Sm = run(A, Xs, Xm, **kw)
        for n in ("Current (drifted)", RECOMMENDED):
            for lens in ("style", "macro"):
                e = E[(E.portfolio == n) & (E.lens == lens)].set_index("factor")
                if e.empty:
                    continue
                s = Sm[(Sm.portfolio == n) & (Sm.lens == lens)].iloc[0]
                row = {"window": name, "portfolio": n.split(":")[0], "lens": lens, "months": s["n"],
                       "factor share %": round(s["factor_share"]), "TE": round(s["te"], 2)}
                for f in (["Mkt-RF", "HML", "Mom"] if lens == "style" else ["Market", "Real yield", "Breakeven"]):
                    row[LABEL[f].split(",")[0]] = f"{e.loc[f, 'exposure']:+.2f} ({e.loc[f, 't']:+.1f})"
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "exposures_robustness.csv", index=False)
    print(f"\n(XLC treated as matching SPY before June 2018: affects the 'since 2007' rows only)\n")
    print(out.astype(object).where(out.notna(), "").to_string(index=False))


if __name__ == "__main__":
    robustness() if "--robust" in sys.argv else main()