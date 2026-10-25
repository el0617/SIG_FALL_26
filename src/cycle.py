import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.data import fred, monthly, optional_csv, sector_returns
from src.utils import zexp, composite, conditional_returns

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
LAG = 1

def yield_spread():
    """10y minus 3m, monthly avg. GS10/TB3MS go back to 1953 -> more recessions for the probit."""
    return (monthly(fred("GS10"), "mean") - monthly(fred("TB3MS"), "mean")).rename("spread")

def build_leading():
    """All components oriented so that HIGHER = BETTER."""
    c = pd.DataFrame({
        "curve":     yield_spread(),
        "claims":    -monthly(fred("ICSA"), "mean").pct_change(12) * 100,
        "permits":   (monthly(fred("PERMIT")).pct_change(12) * 100).shift(LAG),
        "sentiment": monthly(fred("UMCSENT")).diff(6),
    })
    hy = optional_csv("hy_oas")
    if hy is None:
        hy = fred("BAMLH0A0HYM2")
    c["hy_oas_inv"] = -monthly(hy, "mean")
    no, inv = optional_csv("ism_new_orders"), optional_csv("ism_inventories")
    if no is not None and inv is not None:
        c["ism_no_minus_inv"] = (monthly(no) - monthly(inv)).shift(LAG)
    return c

def diffusion(z, n=3):
    """% of components whose z-score improved over n months."""
    return (np.sign(z.diff(n)).clip(lower=0).mean(axis=1) * 100).rename("diffusion")

def cycle_phase(lei, n=3):
    lvl = lei.rolling(3).mean()
    mom = lei.diff(n)
    conds = [(lvl <= 0) & (mom > 0), (lvl > 0) & (mom > 0), (lvl > 0) & (mom <= 0), (lvl <= 0) & (mom <= 0)]
    names = ["Early", "Mid", "Late", "Contraction"]
    return pd.Series(np.select(conds, names, default=""), index=lei.index).replace("", np.nan).rename("phase")

def recession_probit():
    usrec = monthly(fred("USREC"))
    # target = 1 if any NBER recession month falls in t+1 ... t+12
    y = pd.concat([usrec.shift(-k) for k in range(1, 13)], axis=1).max(axis=1, skipna=False).rename("y")
    sp = yield_spread()
    d = pd.concat([y, sp], axis=1).dropna()
    X = sm.add_constant(d["spread"])
    res = sm.Probit(d["y"], X).fit(disp=False, cov_type="HAC", cov_kwds={"maxlags": 12})
    prob = pd.Series(res.predict(sm.add_constant(sp.dropna())), index=sp.dropna().index, name="p_recession_12m")
    auc = roc_auc_score(d["y"], res.predict(X))
    return prob, res, auc

def main():
    comps = build_leading()
    z = zexp(comps)
    lei = composite(z).rename("lei_z")
    div = diffusion(z)
    phase = cycle_phase(lei)

    prob, res, auc = recession_probit()
    print(res.summary().tables[1])
    print(f"\nIn-sample AUC: {auc:.2f}   Latest P(recession in next 12m): {prob.iloc[-1]:.1%}")

    out = pd.concat([lei, lei.diff(3).rename("lei_mom"), div, phase, prob], axis=1)
    out.to_csv(OUT / "cycle_state.csv")
    print("\nCurrent state (last 6 months):\n", out.dropna(subset=["lei_z"]).tail(6).round(2))

    mean, hit, n = conditional_returns(phase, sector_returns())
    print("\nAnnualized excess return vs SPY by cycle phase (%):\n", mean)
    print("\nHit rate (%):\n", hit); print("\nMonths:\n", n)
    mean.to_csv(OUT / "cycle_excess_returns.csv"); hit.to_csv(OUT / "cycle_hit_rates.csv")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    out["lei_z"].dropna().plot(ax=ax[0], title="Leading composite (z)"); ax[0].axhline(0, c="k", lw=0.8)
    out["p_recession_12m"].dropna().plot(ax=ax[1], title="Probit: P(recession within 12m)")
    usrec = monthly(fred("USREC"))
    for a in ax:
        a.fill_between(usrec.index, *a.get_ylim(), where=usrec > 0, color="grey", alpha=0.25)
    fig.savefig(OUT / "cycle_dashboard.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()