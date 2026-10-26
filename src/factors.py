"""Module 4 - Opportunity cost, insurance value and where to put the money (v1).

Run:
    python -m src.factors            # main run
    python -m src.factors --robust   # robustness table

QUESTIONS (sharpened by Modules 1-3 and 5)
  1. What does the defensive leg COST?  Defensives have a lower market beta than growth, so in a
     rising market they give up the equity premium. Cost ~ beta gap x equity risk premium, plus
     any persistent drift (alpha) measured over 1999-2026.
  2. What does it BUY?  Insurance math: average 3-month return of the trade in normal quarters
     (carry) vs in market drawdowns (SPY 3-month return <= -5%: protection). The breakeven
     drawdown probability p* is how likely a drawdown must be for the trade to pay for itself.
     Compared with the historical base rate, and with the base rate when Module 2's real-time
     recession probability was at least as high as today.
  3. WHERE should the money taken out of growth go?  Seven pre-defined funding candidates, each
     measured as "candidate minus growth (XLK, XLY)": market beta, carry, protection, p*, and the
     two rate channels from Module 3 (real-yield beta post-2020, breakeven beta). Pre-set screen:
     a candidate "hedges the inflation channel" if its breakeven beta is at most half of the
     current defensive basket's AND its post-2020 real-yield beta is positive.
  4. Factor lens (Ken French factors, downloaded automatically): which factor bets the trade
     carries, and how those factors behaved after large real-yield rises.
"""
import io
import sys
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from src.data import RAW, sector_returns, complete_months
from src.utils import conditional_stats
from src.regime import SECTORS, ROW_NAME, DEFENSIVE, GROWTH, SAMPLE_START
from src.policy import build_state, rate_betas, BETA_START

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ specification (frozen)
DRAWDOWN = -5.0          # SPY 3-month total return (%) at or below which a quarter counts as a drawdown
ERP = 5.0                # assumed equity risk premium (%/yr) for the beta-gap cost; 3 and 7 in --robust
POST = "2020-01-01"      # start of the post-2020 rate regime found in Module 3
CANDIDATES = {           # name: {ticker: weight}; each is measured as candidate minus GROWTH
    "Defensives (current)":        {"XLP": 1, "XLU": 1, "XLV": 1},
    "Defensives ex-utilities":     {"XLP": 1, "XLV": 1},
    "Health care":                 {"XLV": 1},
    "Staples":                     {"XLP": 1},
    "Defensives + inflation hedge": {"XLP": 1, "XLU": 1, "XLV": 1, "XLE": 0.75, "XLB": 0.75},
    "Financials":                  {"XLF": 1},
    "Broad market (SPY)":          {"SPY": 1},
}
BASE = "Defensives (current)"
FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
FNAME = {"Mkt-RF": "Market", "SMB": "Size (small)", "HML": "Value", "RMW": "Profitability",
         "CMA": "Investment\n(conservative)", "Mom": "Momentum"}


# ------------------------------------------------------------------ data
def monthly_panel():
    r = complete_months(sector_returns())[["SPY"] + SECTORS] * 100
    grw = r[GROWTH].mean(axis=1, skipna=False)
    out = {}
    for name, w in CANDIDATES.items():
        w = pd.Series(w, dtype=float); w /= w.sum()
        out[name] = (r[w.index] * w).sum(axis=1, min_count=len(w)) - grw
    return pd.DataFrame(out).loc[SAMPLE_START:], r.loc[SAMPLE_START:]


def compound(m, n=3):
    """n-month compounded return (%) ending at each month, from monthly % returns."""
    return ((1 + m / 100).rolling(n).apply(np.prod, raw=True) - 1) * 100


def _parse_french(text):
    rows, header = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None and len(parts) > 1 and parts[0] == "" and any(p in ("Mkt-RF", "Mom") for p in parts):
            header = parts[1:]
            continue
        if header is not None:
            if "Annual" in line:
                break
            if parts and len(parts[0]) == 6 and parts[0].isdigit():
                rows.append([parts[0]] + parts[1:len(header) + 1])
    df = pd.DataFrame(rows, columns=["ym"] + header)
    df.index = pd.to_datetime(df.pop("ym"), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.apply(pd.to_numeric, errors="coerce").replace(-99.99, np.nan)


def french_factors(refresh=False):
    """Ken French 5 factors + momentum, monthly %, cached in data/raw/ff_factors.csv. None if unavailable."""
    f = RAW / "ff_factors.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True)
    try:
        parts = []
        for url in FF_URLS.values():
            raw = urllib.request.urlopen(url, timeout=30).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            parts.append(_parse_french(z.read(z.namelist()[0]).decode("latin-1")))
        ff = pd.concat(parts, axis=1, sort=True)
        ff.columns = [c.strip() for c in ff.columns]
        ff.to_csv(f)
        return ff
    except Exception as e:
        print(f"  [factors] Ken French download failed ({str(e)[:80]}). Factor section skipped.\n"
              f"  To fix: download the two CSV zips from the Ken French data library (5 factors 2x3, "
              f"momentum), unzip both into data/raw/, then run: python -m src.factors --load-french")
        return None


def load_french_local():
    parts = []
    for pat in ["F-F_Research_Data_5_Factors_2x3*.csv", "F-F_Momentum_Factor*.csv"]:
        hits = sorted(RAW.glob(pat)) + sorted(RAW.glob(pat.replace(".csv", ".CSV")))
        if not hits:
            raise FileNotFoundError(f"data/raw/{pat} not found")
        parts.append(_parse_french(hits[0].read_text(encoding="latin-1")))
    ff = pd.concat(parts, axis=1, sort=True)
    ff.columns = [c.strip() for c in ff.columns]
    ff.to_csv(RAW / "ff_factors.csv")
    print(f"  wrote data/raw/ff_factors.csv ({ff.index.min():%Y-%m} to {ff.index.max():%Y-%m})")


# ------------------------------------------------------------------ analysis
def insurance(P, r, drawdown=DRAWDOWN, start=None):
    """Carry vs protection for each column of P (monthly % spreads), using 3-month windows."""
    spy3 = compound(r["SPY"])
    rows = []
    for c in P.columns:
        # 3-month return of the spread = compounded candidate minus compounded growth;
        # approximated by the sum of monthly spreads (difference < 0.1pp at these magnitudes)
        x3 = P[c].rolling(3).sum()
        d = pd.concat([x3, spy3], axis=1, keys=["x", "spy"], sort=True).loc[start:].dropna()
        dd = d["spy"] <= drawdown
        carry, prot = d.loc[~dd, "x"].mean(), d.loc[dd, "x"].mean()
        if carry >= 0 and prot >= 0:
            pstar, note = 0.0, "pays in both"
        elif prot <= carry:
            pstar, note = np.nan, "no protection"
        else:
            pstar, note = -carry / (prot - carry), ""
        rows.append({"candidate": c, "carry_3m": carry, "protection_3m": prot, "p_star": pstar, "note": note,
                     "drawdown_share": dd.mean(), "n": len(d)})
    return pd.DataFrame(rows).set_index("candidate")


def market_beta(P, r, start=None):
    rows = {}
    for c in P.columns:
        d = pd.concat([P[c], r["SPY"]], axis=1, keys=["y", "m"], sort=True).loc[start:].dropna()
        res = sm.OLS(d["y"].values, sm.add_constant(d["m"].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows[c] = {"beta": float(res.params[1]), "beta_t": float(res.tvalues[1]),
                   "alpha_ann": float(res.params[0]) * 12, "alpha_t": float(res.tvalues[0])}
    return pd.DataFrame(rows).T


def candidate_table(P, r, S, drawdown=DRAWDOWN, erp=ERP):
    ins = insurance(P, r, drawdown)
    mb = market_beta(P, r)
    Bf = rate_betas(P.loc[BETA_START:], S)
    Bp = rate_betas(P.loc[POST:], S)
    T = pd.DataFrame({
        "mean_pp_yr": P.mean() * 12, "vol_pp_yr": P.std() * np.sqrt(12),
        "market_beta": mb["beta"], "beta_cost_pp_yr": mb["beta"] * erp, "alpha_pp_yr": mb["alpha_ann"],
        "alpha_t": mb["alpha_t"],
        "carry_3m": ins["carry_3m"], "protection_3m": ins["protection_3m"], "p_star": ins["p_star"],
        "real_beta_50bp_post2020": Bp["b_real10y"] / 2, "real_t_post2020": Bp["t_real10y"],
        "be_beta_25bp": Bf["b_breakeven10y"] / 4, "be_t": Bf["t_breakeven10y"],
        "be_beta_25bp_post2020": Bp["b_breakeven10y"] / 4,
    })
    T["sharpe"] = T["mean_pp_yr"] / T["vol_pp_yr"]
    base_be = T.loc[BASE, "be_beta_25bp"]
    T["hedges_inflation_channel"] = (T["be_beta_25bp"].abs() <= 0.5 * abs(base_be)) & (T["real_beta_50bp_post2020"] > 0)
    return T, ins


def drawdown_odds(r, drawdown=DRAWDOWN):
    """Base rate of a drawdown quarter, and the rate when Module 2's real-time recession probability
    was at least today's value (next 3 months)."""
    from src.cycle import recession_probit
    prob = recession_probit()[0]
    fwd = compound(r["SPY"]).shift(-3)                 # SPY return over months t+1..t+3
    d = pd.concat([fwd, prob], axis=1, keys=["f", "p"], sort=True).dropna()
    now = float(prob.dropna().iloc[-1])
    base = float((d["f"] <= drawdown).mean())
    hi = d[d["p"] >= now]
    cond = float((hi["f"] <= drawdown).mean()) if len(hi) >= 24 else np.nan
    return dict(base=base, cond=cond, n_cond=len(hi), p_now=now, n=len(d))


def factor_loadings(P, ff, cols=FACTORS):
    rows = {}
    for c in P.columns:
        d = pd.concat([P[c], ff[cols]], axis=1, sort=True).dropna()
        res = sm.OLS(d[c].values, sm.add_constant(d[cols].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        row = {"alpha_pp_yr": res.params[0] * 12, "alpha_t": res.tvalues[0], "r2": res.rsquared, "n": len(d)}
        for k, f in enumerate(cols, start=1):
            row[f] = res.params[k]; row[f + "_t"] = res.tvalues[k]; row[f + "_se"] = res.bse[k]
        rows[c] = row
    return pd.DataFrame(rows).T


def factors_after_real_yield_rises(ff, S, threshold=0.50):
    """Next-3-month factor returns when the real 10y rose by >= threshold (pp) over the prior 6 months."""
    d6 = S["real10y"].diff(6)
    lab = pd.Series(np.where(d6 >= threshold, "Real yields up 50bp+ (6m)", "Other months"), index=d6.index,
                    dtype=object).where(d6.notna())
    f3 = ff[[c for c in FACTORS if c != "Mkt-RF"]].rolling(3).mean().shift(-3) / 100   # monthly decimal -> stats x1200
    st = conditional_stats(lab.reindex(f3.index), f3, order=["Real yields up 50bp+ (6m)", "Other months"],
                           family={c: "factor" for c in f3.columns}, loeo=True)
    return st, lab


# ------------------------------------------------------------------ plots
def plot_insurance(T, odds, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, row in T.iterrows():
        c = "#2e7d32" if row["hedges_inflation_channel"] else ("#37474f" if name == BASE else "#90a4ae")
        ax.scatter(row["carry_3m"], row["protection_3m"], s=140, color=c, edgecolor="k", zorder=3)
        ps = "pays in both" if row["p_star"] == 0 else ("no protection" if pd.isna(row["p_star"]) else f"p*={row['p_star']:.0%}")
        ax.annotate(f"{name}\n({ps})", (row["carry_3m"], row["protection_3m"]), xytext=(7, 4),
                    textcoords="offset points", fontsize=8.5)
    ax.axhline(0, c="k", lw=0.8); ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Carry: average 3-month return vs growth in NORMAL quarters (%)")
    ax.set_ylabel(f"Protection: average 3-month return vs growth when SPY fell ≥{abs(DRAWDOWN):.0f}% (%)")
    sub = (f"drawdown base rate {odds['base']:.0%} of quarters"
           + (f"; {odds['cond']:.0%} when Module 2's recession probability was ≥ today's {odds['p_now']:.0%}"
              if not np.isnan(odds['cond']) else ""))
    ax.set_title("What the money taken out of growth buys: carry vs protection, 1999-2026\n"
                 f"p* = drawdown probability needed to break even ({sub})\n"
                 "green = passes the inflation-channel screen", fontsize=9.5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_table(T, path):
    cols = [("market_beta", "Market beta\nvs SPY (trade-off)", "{:+.2f}", 0),
            ("beta_cost_pp_yr", f"Beta cost at\n{ERP:.0f}% ERP (pp/yr)", "{:+.1f}", 1),
            ("alpha_pp_yr", "Drift ex-beta\n(pp/yr)", "{:+.1f}", 1),
            ("carry_3m", "Carry\n3m (%)", "{:+.1f}", 1),
            ("protection_3m", "Protection\n3m (%)", "{:+.1f}", 1),
            ("real_beta_50bp_post2020", "Real yield +50bp\npost-2020 (%)", "{:+.2f}", 1),
            ("be_beta_25bp", "Breakeven +25bp\n2003-26 (%)", "{:+.2f}", 1),
            ("be_beta_25bp_post2020", "Breakeven +25bp\npost-2020 (%)", "{:+.2f}", 1)]
    M = np.array([[T.loc[n, c] for c, *_ in cols] for n in T.index], dtype=float)
    Zc = np.zeros_like(M)
    for j, (_, _, _, sign) in enumerate(cols):
        if sign == 0:
            continue                                   # trade-off column: no good/bad colouring
        col = M[:, j] * sign
        s = np.nanmax(np.abs(col - np.nanmedian(col))) or 1
        Zc[:, j] = (col - np.nanmedian(col)) / s
    fig, ax = plt.subplots(figsize=(14, 0.62 * len(T) + 2))
    ax.imshow(Zc, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j, (_, _, fmt, _) in enumerate(cols):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else fmt.format(M[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([("✓ " if T.loc[n, "hedges_inflation_channel"] else "   ") + n for n in T.index], fontsize=9)
    ax.set_title("Funding candidates, each measured as candidate minus growth (XLK, XLY). "
                 "Green = better for the trade in that column; ✓ = passes the inflation-channel screen", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_loadings(L, path, name=BASE):
    row = L.loc[name]
    b = np.array([row[f] for f in FACTORS], dtype=float)
    ci = 1.96 * np.array([row[f + "_se"] for f in FACTORS], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(FACTORS))
    ax.bar(x, b, color=["#37474f" if abs(row[f + "_t"]) >= 2 else "#b0bec5" for f in FACTORS])
    ax.errorbar(x, b, yerr=ci, fmt="none", ecolor="k", capsize=4)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([FNAME[f] for f in FACTORS], fontsize=9)
    ax.set_ylabel("Loading (monthly regression)")
    ax.set_title(f"Factor bets inside '{name} minus growth' (Ken French factors, {int(row['n'])} months, HAC 95% CI)\n"
                 f"R² {row['r2']:.0%}; return not explained by factors {row['alpha_pp_yr']:+.1f} pp/yr "
                 f"(t={row['alpha_t']:.2f}); dark = |t| ≥ 2", fontsize=9.5)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ runs
def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=" * 90, "\nMODULE 4 - OPPORTUNITY COST, INSURANCE VALUE AND FUNDING CANDIDATES (v1)")
    P, r = monthly_panel()
    S = build_state()
    print(f"  sample {P.index.min():%Y-%m} to {P.index.max():%Y-%m}; drawdown = SPY 3m <= {DRAWDOWN:.0f}%; ERP {ERP:.0f}%")

    T, ins = candidate_table(P, r, S)
    T.round(3).to_csv(OUT / "factors_candidates.csv")
    show = ["market_beta", "beta_cost_pp_yr", "alpha_pp_yr", "alpha_t", "mean_pp_yr", "sharpe", "carry_3m",
            "protection_3m", "p_star", "real_beta_50bp_post2020", "real_t_post2020", "be_beta_25bp", "be_t",
            "be_beta_25bp_post2020", "hedges_inflation_channel"]
    print("\nFunding candidates (each = candidate minus growth):\n", T[show].round(2).to_string())

    odds = drawdown_odds(r)
    print(f"\nDrawdown odds (SPY next 3m <= {DRAWDOWN:.0f}%): base rate {odds['base']:.0%} of {odds['n']} months; "
          f"when real-time recession probability >= today's {odds['p_now']:.0%}: "
          + (f"{odds['cond']:.0%} ({odds['n_cond']} months)" if not np.isnan(odds['cond']) else "too few months"))

    ff = french_factors()
    L = None
    if ff is not None:
        L = factor_loadings(P, ff)
        L.round(3).to_csv(OUT / "factors_loadings.csv")
        cols = ["r2", "alpha_pp_yr", "alpha_t"] + [c for f in FACTORS for c in (f, f + "_t")]
        print(f"\nFactor loadings (Ken French, {ff.index.min():%Y-%m} to {ff.index.max():%Y-%m}):\n", L[cols].astype(float).round(2).to_string())
        fst, lab = factors_after_real_yield_rises(ff, S)
        fst.to_csv(OUT / "factors_after_real_yield_rises.csv", index=False)
        print("\nFactor returns over the NEXT 3 months after real yields rose 50bp+ in 6 months (annualized pp):\n",
              fst.set_index(["series", "state"])[["months", "episodes", "raw", "raw_t", "edge", "t", "q", "sign_stable"]].round(2))
        print(f"  today: real 10y 6m change {S['real10y'].diff(6).dropna().iloc[-1]*100:+.0f}bp -> '{lab.dropna().iloc[-1]}'")
        plot_loadings(L, OUT / "factors_loadings.png")

    plot_insurance(T, odds, OUT / "factors_insurance.png")
    plot_table(T, OUT / "factors_candidates.png")

    # ---- evidence summary
    b = T.loc[BASE]
    passed = [n for n in T.index if T.loc[n, "hedges_inflation_channel"]]
    print("\n" + "=" * 90 + "\nEVIDENCE SUMMARY (paste into the scorecard)")
    print(f"  Defensive leg (DEF minus GRW): market beta {b['market_beta']:+.2f} vs SPY -> forgoes about "
          f"{-b['beta_cost_pp_yr']:.1f} pp/yr of equity premium at a {ERP:.0f}% ERP")
    print(f"    drift beyond beta 1999-2026: {b['alpha_pp_yr']:+.1f} pp/yr (t={b['alpha_t']:.2f})")
    ps = b["p_star"]
    pstxt = "pays in both" if ps == 0 else ("no protection" if pd.isna(ps) else f"{ps:.0%}")
    print(f"  Insurance: carry {b['carry_3m']:+.1f}% per normal quarter, protection {b['protection_3m']:+.1f}% per "
          f"drawdown quarter -> breakeven drawdown probability p* = {pstxt}")
    print(f"    vs base rate {odds['base']:.0%}"
          + (f", and {odds['cond']:.0%} when the recession probability was >= today's {odds['p_now']:.0%}" if not np.isnan(odds['cond']) else ""))
    if not pd.isna(ps):
        ref = odds["cond"] if not np.isnan(odds["cond"]) else odds["base"]
        print(f"    -> on these odds the defensive leg {'PAYS FOR ITSELF' if ref >= ps else 'DOES NOT pay for itself'} "
              f"over a typical quarter ({ref:.0%} vs {ps:.0%} needed)")
    print(f"  Rate channels: post-2020 real-yield +50bp {b['real_beta_50bp_post2020']:+.2f}%, breakeven +25bp "
          f"{b['be_beta_25bp']:+.2f}% (2003-26) / {b['be_beta_25bp_post2020']:+.2f}% (post-2020)")
    print(f"  Candidates passing the inflation-channel screen: {', '.join(passed) or 'none'}")
    if L is not None:
        l = L.loc[BASE]
        big = sorted(FACTORS, key=lambda f: -abs(l[f + "_t"]))[:3]
        print("  Largest factor bets in the defensive trade: " +
              ", ".join(f"{FNAME[f].replace(chr(10), ' ')} {l[f]:+.2f} (t={l[f + '_t']:.1f})" for f in big) + f"; R² {l['r2']:.0%}")
    print("=" * 90)


def robustness():
    P, r = monthly_panel()
    S = build_state()
    rows = []
    for name, kw in [("baseline", {}), ("drawdown -3%", {"drawdown": -3.0}), ("drawdown -8%", {"drawdown": -8.0}),
                     ("ERP 3%", {"erp": 3.0}), ("ERP 7%", {"erp": 7.0})]:
        T, _ = candidate_table(P, r, S, **kw)
        for c in [BASE, "Defensives ex-utilities", "Defensives + inflation hedge", "Broad market (SPY)"]:
            rows.append({"spec": name, "candidate": c, "beta cost": round(T.loc[c, "beta_cost_pp_yr"], 1),
                         "carry": round(T.loc[c, "carry_3m"], 2), "protection": round(T.loc[c, "protection_3m"], 2),
                         "p*": None if pd.isna(T.loc[c, "p_star"]) else round(T.loc[c, "p_star"], 2),
                         "passes screen": bool(T.loc[c, "hedges_inflation_channel"])})
    for start in ["2010-01-01", POST]:
        ins = insurance(P, r, start=start)
        for c in [BASE, "Defensives + inflation hedge"]:
            rows.append({"spec": f"insurance from {start[:4]}", "candidate": c,
                         "carry": round(ins.loc[c, "carry_3m"], 2), "protection": round(ins.loc[c, "protection_3m"], 2),
                         "p*": None if pd.isna(ins.loc[c, "p_star"]) else round(ins.loc[c, "p_star"], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "factors_robustness.csv", index=False)
    print("\n", out.astype(object).where(out.notna(), "").to_string(index=False))


if __name__ == "__main__":
    if "--load-french" in sys.argv:
        load_french_local()
    elif "--robust" in sys.argv:
        robustness()
    else:
        main()