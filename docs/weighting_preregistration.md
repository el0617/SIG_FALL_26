# Weighting engine — pre-registration

Written 2026-10-31, before `src/weighting.py` exists or any weight has been computed. Committed and pushed
before the first run. Not blind: written after the thesis tournament (Task 1) and all module outputs.
Any change after the first run is reported as a deviation.

## Principle
No return forecast survived out of sample, so this is not an optimizer. Start from the benchmark and spend a
fixed tracking-error (TE) budget only on bets with a stated reason, sized by the strength of their evidence.

## Inputs
- Benchmark `b`: IVV GICS sector weights (`data/portfolio.csv`, `benchmark` column), as in Module 6.
- Current: PORT weights of 05/29/26 drifted to the last complete month (`portfolio.drift`).
- Returns: `portfolio.etf_returns()` (monthly %, VGT spliced with XLK before 2004; XLC = SPY before 2018 where needed).
- TE = sqrt(12 · xᵀΣx), x = active weights, Σ = covariance of monthly total returns over the last 60 months
  (as Module 6). Stability windows: 36 and 96 months.

## Bets (unit active vectors; funding is always pro rata to benchmark weights of the holdings not in the bet)
| Bet | Long | Funded from | Evidence (Task 1 / Modules 3, 4) | Evidence units |
|---|---|---|---|---|
| B1 Defensive insurance | XLP, XLU, XLV, equal | all other holdings, pro rata | alpha beyond beta t 2.0; sell-off payoff t 4.9 | 2 (strong) |
| B2 Breakeven hedge | XLE, XLB, equal | all other holdings, pro rata | breakeven betas t 4.0 / 4.8; no alpha (tournament 0) | sized by hedge rule |
| B3 Growth underweight | − VGT, − XLY, pro rata to their benchmark weights | proceeds to all other holdings, pro rata | team thesis; no scored support (tournament) | 1 (discretionary) |

- **Hedge rule (B2):** B2's size is set so the portfolio's breakeven beta (full-sample, each ETF minus SPY,
  `etf_sensitivities` `be_full`) is HALF of what B1 alone would give. Half, not full, because the hedge carries no
  alpha in Task 1.
- **Evidence sizing:** in versions that include B3, B1 and B3 are sized so their standalone TEs are in the ratio 2 : 1.
- Everything not in a bet sits at benchmark weight plus/minus its share of the pro-rata funding. No other
  active positions (XLC, XLI, XLF, SCHH end near benchmark by construction).

## Versions (the whole active vector is scaled to hit the TE target exactly)
| Version | Bets | TE target vs benchmark |
|---|---|---|
| Conservative | B1 + B2 | 1.5% |
| Base | B1 + B2 | 2.5% |
| Conviction | B1 + B2 + B3 | 3.5% (≈ today's 3.46%: same risk, spent differently) |

Weights rounded to 0.1pp after scaling (rounding residual put on the largest holding); no weight below 0.

## Reported for each version, and for Current, the "Proposal" (Core + trim energy and growth underweight) and "XLP → XLF only"
Weights, active weights, trades from current (pp and $ on $100M); TE vs benchmark and vs SPY; beta vs SPY;
3-month CVaR 95% of active return (fixed weights, history since 1999, XLC = SPY before 2018); Euler share of
active risk by bet group; breakeven beta (per +25bp) and post-2020 real-yield beta (per +50bp); expected
3-month active return under Module 8 sets A/B/C and the four judgment theses (same weights as Task 1), and in
sell-off windows; TE and weights under 36/96-month covariances.

## Checks (pass / fail, fixed now) — Base version vs Current
1. TE vs benchmark lower than Current.
2. Sell-off payoff (scenarios 1 + 2, 3-month active) at least 75% of Current's.
3. Breakeven beta smaller in magnitude than Current's.
4. Expected active return under Module 8 set A not lower than Current's by more than 0.10pp a quarter.
5. Stable: under 36- and 96-month covariances, every Base active weight keeps its sign and stays within ±1.5pp of the 60-month value.

Base is presented as the recommended version only if it passes all five; otherwise the failures are reported
and no version is recommended by the engine.
