#!/usr/bin/env bash
# One-command data freeze for the final presentation.
#
# Run from the repo root in the Codespace (it needs FRED_API_KEY and the Bloomberg CSVs in data/raw):
#     git fetch origin && git checkout freeze-2026-11-04 && bash scripts/freeze.sh
#
# What it does, in order, stopping at the first problem:
#   1. checks the FRED key and that the Bloomberg caches exist (without them Modules 1, 2, 5, 7 fall back
#      to free series and the verdicts change for a reason unrelated to the data);
#   2. re-downloads every FRED / Yahoo / Ken French cache (python -m src.data --refresh);
#   3. runs every module in dependency order, then the robustness runs, saving each log in logs/;
#   4. checks whether the agreed cap (defensive overweight no higher than today's) binds on Base;
#   5. writes FREEZE.md, commits, tags freeze-YYYY-MM-DD and pushes the branch and tag.
# DRY_RUN=1 skips the refresh and the commit/push (for testing the steps on cached data).
# Licensed Bloomberg data never leaves the Codespace: those files and logs are in .gitignore, and the
# script aborts if any of them is staged.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TAG="freeze-$(date +%F)"
MAIN="${MAIN:-regime cycle policy factors market portfolio exposures valuation scenarios walkforward energy tournament weighting consistency}"
ROBUST="${ROBUST:-regime cycle policy factors market portfolio exposures valuation scenarios walkforward energy}"
BBG="hy_oas ism_new_orders ism_prices ism_inventories val_pe val_eps"
say() { printf "\n== %s\n" "$*"; }
die() { printf "\nSTOPPED: %s\n" "$*" >&2; exit 1; }

say "1/5 Checks"
[ -n "${FRED_API_KEY:-}" ] || die "FRED_API_KEY is not set (Codespaces secret, or: export FRED_API_KEY=...)"
python -c "import pandas, numpy, scipy, statsmodels, matplotlib, fredapi, yfinance, openpyxl" 2>/dev/null \
  || pip install -q pandas numpy scipy statsmodels matplotlib fredapi yfinance openpyxl
mkdir -p logs
missing=""
for f in $BBG; do [ -f "data/raw/$f.csv" ] || missing="$missing $f.csv"; done
if [ -n "$missing" ] && [ "${ALLOW_NO_BLOOMBERG:-0}" != "1" ]; then
  die "Bloomberg caches missing in data/raw:$missing. Import them first (src.import_bloomberg / src.valuation),
         or rerun with ALLOW_NO_BLOOMBERG=1 to accept the free fallbacks (verdicts may change)."
fi
python - > logs/bloomberg_status.txt <<'EOF'
import pandas as pd
from pathlib import Path
for f in ["hy_oas", "ism_new_orders", "ism_prices", "ism_inventories", "val_pe", "val_eps"]:
    p = Path("data/raw") / f"{f}.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        print(f"{f:16s} last obs {d.dropna(how='all').index.max():%Y-%m-%d}")
    else:
        print(f"{f:16s} MISSING")
EOF
cat logs/bloomberg_status.txt

if [ "${DRY_RUN:-0}" = "1" ]; then say "DRY_RUN: skipping the refresh"; else
say "2/5 Refreshing FRED, Yahoo and Ken French caches"
python -m src.data --refresh 2>&1 | tee logs/data_refresh.txt || true
grep -q "all downloads succeeded" logs/data_refresh.txt || die "some downloads failed: see logs/data_refresh.txt"
fi

say "3/5 Running the modules"
for m in $MAIN; do
  printf "  %-12s" "$m"; python -m src.$m > "logs/$m.txt" 2>&1 || die "src.$m failed: see logs/$m.txt"; echo ok
done
for m in $ROBUST; do
  printf "  %-12s" "$m --robust"; python -m src.$m --robust > "logs/${m}_robust.txt" 2>&1 || die "src.$m --robust failed: see logs/${m}_robust.txt"; echo ok
done
if [ "${DRY_RUN:-0}" != "1" ] && grep -h "WARNING .* ends" logs/*.txt | sort -u | grep . ; then
  die "a cache is stale (above). Re-run the refresh before freezing."
fi

say "4/5 Defensive cap check"
python - <<'EOF' | tee logs/cap_check.txt
import pandas as pd
W = pd.read_csv("outputs/weighting_weights.csv", index_col=0)
a = W[["Current", "Base"]].sub(W["benchmark"], axis=0)
d = a.loc[["XLP", "XLV", "XLU"]].sum()
print(f"Defensive active weight: Current {d['Current']:+.1f}pp, Base {d['Base']:+.1f}pp")
print("Cap BINDS: the recommendation is 'Base capped (post hoc)', a reported deviation" if d["Base"] > d["Current"] + 0.05
      else "Cap does not bind: Base is the recommendation as pre-registered")
M = pd.read_csv("outputs/weighting_metrics.csv", index_col=0)
cols = [c for c in ["Current", "Base", "Base capped (post hoc)"] if c in M.index]
print(M.loc[cols, ["te_bench", "beta", "cvar3", "selloff_3m", "be_25bp", "exp_A"]].round(2).to_string())
print("\nPre-registered checks, Base:\n" + open("outputs/weighting_checks.csv").read())
print("Same checks, Base capped (context):\n" + open("outputs/weighting_checks_capped.csv").read())
EOF

say "5/5 Freeze: FREEZE.md, commit, tag, push"
{
  echo "# Data freeze $(date +%F)"
  echo
  echo "- Run: $(date '+%F %H:%M %Z') on branch $(git rev-parse --abbrev-ref HEAD), code at $(git rev-parse --short HEAD)"
  echo "- Data: monthly statistics run through the last complete month before the run; releases after it"
  echo "  (e.g. October payrolls on 6 Nov, October CPI on 10 Nov) are in the narrative, not the models."
  echo "- Bloomberg caches (licensed, not committed):"
  sed 's/^/  - /' logs/bloomberg_status.txt
  echo "- Cap check:"
  head -2 logs/cap_check.txt | sed 's/^/  - /'
  echo "- Deviations from pre-registration: the post-hoc variant Base capped in src/weighting.py (team cap on the defensive overweight); it is the recommendation only if the cap check above says it binds."
  echo
  echo "## Download caches"
  echo '```'
  python -m src.data --status 2>/dev/null | head -60
  echo '```'
} > FREEZE.md

if [ "${DRY_RUN:-0}" = "1" ]; then say "DRY_RUN: FREEZE.md written; no commit, tag or push"; exit 0; fi
git add -A
if git diff --cached --name-only | grep -E "data/raw/(bloomberg|val_|ism_|hy_oas)|outputs/(valuation_signals|valuation_sectors|tournament_valuation)|logs/valuation"; then
  die "licensed Bloomberg files are staged (above). Unstage them; nothing was committed."
fi
git commit -q -m "Data freeze $(date +%F): refresh all caches, re-run every module and robustness check"
git tag -a "$TAG" -m "Final presentation data freeze"
git push -u origin HEAD
git push origin "$TAG"
say "Done. Tag $TAG pushed. Tell Claude: \"freeze pushed, tag $TAG\""
