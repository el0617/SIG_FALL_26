# SIG_FALL_26: TOPDOWN2 thesis models

Student-managed top-down fund. Thesis: over the next 3 months, overweight defensives (XLP, XLU, XLV),
underweight growth (VGT proxied by XLK, XLY). Portfolio TOPDOWN2, $100M, 11 sector ETFs.

Modules (run with `python -m src.<name>`, add `--robust` for the robustness table):
- regime (1): growth/inflation regime from coincident data
- cycle (2): leading-indicator momentum, phases, real-time recession probit
- policy (3): real yields, breakevens, Fed path, liquidity impulse
- factors (4): cost of defensives as insurance, Ken French loadings
- market (5): 10 market signals, composite
- portfolio (6): portfolio risk, candidate portfolios, scenarios
- exposures (6b): style and macro factor model of active returns
- valuation (7): forward P/E gap, earnings revisions, equity risk premium (Bloomberg data)

Conventions:
- Target is the next-3-month DEF-GRW return (defensives minus growth), annualized pp (x1200).
- Row t of forward returns = months t+1..t+h; signals must use only data known at month-end t.
- Pre-registered verdict: one-sided 5%, t >= 1.65 SUPPORTS, t <= -1.65 CONTRADICTS, else NEUTRAL.
  Do not change pre-registered tests, thresholds or samples unless a line of code is wrong.
- Bloomberg files in data/raw are licensed: never commit them.