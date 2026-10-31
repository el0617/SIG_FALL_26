"""Consistency checks behind the traceability map (Task 3).

Run:
    python -m src.consistency

1. Module 8's own decision rule (12 settings) applied to the weighting-engine versions.
2. Module 10's oil-crash tail cost at each version's XLE active weight (limit -1.0pp).
3. Sign of the portfolio's real-yield exposure with and without a market (SPY) control.
Needs outputs/weighting_weights.csv (python -m src.weighting).
"""
import pandas as pd
import statsmodels.api as sm

import src.weighting as wg
from src.portfolio import load_portfolio, etf_returns, etf_sensitivities, apply_trades, HOLDINGS, CANDIDATES
from src.policy import build_state
from src.scenarios import build, payoffs, summarize, verdict, CURRENT
from src.energy import stress

M8_SPECS = [{}, {"dd": -4.0}, {"dd": -7.0}, {"be": 15.0}, {"be": 25.0}, {"real": 20.0}, {"real": 35.0},
            {"K": 6}, {"K": 24}, {"start": "2010-01-31"}, {"model_tag": "post"}, {"path_band": 50.0}]


def main():
    p = load_portfolio(); b = p["benchmark"].reindex(HOLDINGS)
    R = etf_returns(); S = build_state(); sens = etf_sensitivities(R, S)
    W = pd.read_csv("outputs/weighting_weights.csv", index_col=0)
    wg.HEDGE = 1.0
    full = wg.construct(b, sens, wg.cov(R, wg.WINDOW), ("B1", "B2"), 2.5)[0]     # post-hoc variant
    w0 = W["Current"]
    cands = {CURRENT: w0,
             "Add defensives": apply_trades(w0, CANDIDATES["Team direction: add defensives"]).reindex(HOLDINGS),
             "Proposal": W["Proposal"], "Base (half hedge)": W["Base"], "Base (full hedge, post hoc)": full}

    print("1. Module 8 decision rule vs Current, 12 settings")
    res = {c: [] for c in cands if c != CURRENT}
    for prm in M8_SPECS:
        bb = build(prm)
        E, M, NEG, _, _ = payoffs(R, bb["W"], bb["lab"], cands, sens, bb["p"]["model_tag"])
        T = summarize(bb["P"], E, M, NEG)
        for c in res:
            res[c].append(verdict(T, c)[0])
    for c, v in res.items():
        print(f"   {c:30s} {pd.Series(v).value_counts().to_dict()}")

    print("2. Oil-crash tail cost at each XLE active weight (Module 10 limit -1.0pp)")
    for c, w in cands.items():
        st, _, _ = stress(R, float((w - b)["XLE"]))
        print(f"   {c:30s} XLE {float((w - b)['XLE']):+.1f}pp -> worst {st['cost of today' + chr(39) + 's overweight (pp)'].min():+.2f}pp")

    print("3. Real-yield exposure of active return (per +100bp), sign under different controls")
    rel = R[HOLDINGS].sub(R["SPY"], axis=0).fillna({"XLC": 0.0})
    X = pd.DataFrame({"re": S["real10y"].diff(), "be": S["breakeven10y"].diff(), "spy": R["SPY"]})
    for c in ["Current (drifted)", "Base (half hedge)"]:
        a = (rel * ((cands[c] - b) / 100)).sum(axis=1).rename("a")
        for name, start, cols in [("60m, with SPY", None, ["re", "be", "spy"]),
                                  ("post-2020, no SPY", "2020-01-01", ["re", "be"]),
                                  ("post-2020, with SPY", "2020-01-01", ["re", "be", "spy"])]:
            d = pd.concat([a, X], axis=1, sort=True).dropna()
            d = d.iloc[-60:] if start is None else d.loc[start:]
            r = sm.OLS(d["a"], sm.add_constant(d[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            print(f"   {c:20s} {name:22s} {r.params['re']:+.2f}% (t {r.tvalues['re']:+.1f})")


if __name__ == "__main__":
    main()
