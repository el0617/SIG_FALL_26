# Thesis tournament — pre-registration

Written 2026-10-31, before `src/tournament.py` exists or any tournament number has been computed.
Committed and pushed before the first run; the commit time is the record.

**Disclosure (not blind):** written after all module outputs and the ETF-by-scenario averages
("scenario playbook", 2026-10-31) had been seen. The rules are kept standard and simple so they
cannot easily be tuned to that knowledge. Any change after the first run is reported as a deviation.

## Question
Which thesis about the equity sleeve has the strongest quantitative backing, scored on one common set of rules?

## The five theses (each = long basket minus short basket, equal weights, rebalanced monthly)

| Code | Thesis | Long | Short | Economic claim |
|---|---|---|---|---|
| A | Defensive tilt (the team thesis) | XLP, XLU, XLV | VGT, XLY | Defensives beat growth |
| B | Inflation-hedge tilt | XLE, XLB | SPY | Inflation-sensitive sectors beat the market |
| C | Barbell (TOPDOWN2's structure) | ½ (XLP, XLU, XLV) + ½ (XLE, XLB) | VGT, XLY | Recession + inflation hedges beat growth |
| D | Hard-landing insurance | XLP, XLU, XLV | SPY | Defensives beat the market |
| E | Growth drift (anti-thesis) | VGT, XLY | SPY | Growth keeps beating the market |

Returns: monthly total returns from `portfolio.etf_returns()` (VGT spliced with XLK before 2004), through the
last complete month. Baseline return sample 1999-01 onward; scenario criteria use Module 8 windows (2003+).

## Criteria (each scores +1 / 0 / −1; total from −6 to +6)

| # | Criterion | Statistic | +1 | −1 |
|---|---|---|---|---|
| C1 | Pays on average | mean monthly tilt return, HAC(3) t | t ≥ 1.65 | t ≤ −1.65 |
| C2 | Pays beyond market beta | intercept of tilt on SPY, HAC(3) t | t ≥ 1.65 | t ≤ −1.65 |
| C3 | Stable through time | sign of the mean in 1999–2012 and in 2013–latest | both > 0 | both < 0 |
| C4 | Pays in today's conditions | expected 3-month tilt return under Module 8 probability sets A, B, C | > 0 in all 3 | < 0 in all 3 |
| C5 | Robust across macro theses | expected 3-month tilt return under the four judgment theses below | > 0 in all 4 | < 0 in all 4 |
| C6 | Insurance value | mean 3-month tilt return in sell-off windows (scenarios 1 + 2), t with n_eff = windows / 3 | t ≥ 1.65 | t ≤ −1.65 |

Expected 3-month returns = Σ (scenario probability × mean 3-month tilt return in that scenario's windows),
scenarios and probability sets exactly as `src/scenarios.py` builds them.

Judgment theses (C5), scenario weights in % (stagflationary sell-off / growth scare / inflation scare /
real-yield squeeze / easing rally / steady state):
- Inflation persists 8 / 5 / 30 / 17 / 5 / 35
- Higher for longer 6 / 10 / 12 / 30 / 5 / 37
- Hard landing 8 / 25 / 7 / 8 / 20 / 32
- Soft landing 2 / 5 / 10 / 10 / 18 / 55

**Ranking:** by total score; ties broken by C4's expected return under set A, then by the information ratio.

## Context (reported, NOT scored)
Annualized mean, volatility, information ratio, market beta; expected 3-month return per scenario and per
thesis, also scaled to a 2.5% tracking-error budget (× 2.5 / annualized volatility); valuation of the long
vs short basket from Module 7 (harmonic forward P/E ratio percentile since 2006, 3-month relative EPS
revisions). Valuation is not scored because Module 7 found no predictive value (0/11 settings).

## Robustness (every setting fully rescored)
3 return-sample starts (1999 baseline, 2003, 2010; scenario windows start at the later of the two dates)
× 7 scenario-threshold specs (baseline; drawdown −4% / −7%; breakeven 15 / 25bp; real 20 / 35bp)
= 21 settings. Reported per thesis: score range, settings ranked first (ties count for each tied thesis).

## Decision language (fixed now)
- **Strongest backing:** first in the baseline AND first (ties included) in at least 14 of 21 settings.
- Otherwise: "no thesis has clearly the strongest backing"; the baseline ranking is reported as is.
- **Quantitatively contradicted:** baseline total ≤ −2.
- **Quantitatively supported:** baseline total ≥ +2.
