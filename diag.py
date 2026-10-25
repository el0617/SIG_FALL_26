import platform, sys
import numpy as np, pandas as pd, statsmodels
from pathlib import Path
from src.data import fred, sector_returns
from src.regime import build_growth, build_inflation, build_regime
from src.utils import zrobust

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("versions: python", platform.python_version(), "pandas", pd.__version__, "statsmodels", statsmodels.__version__)
print("files in data/raw:", sorted(p.name for p in Path("data/raw").glob("*")))

G, I = build_growth(), build_inflation()

print("\n== 1. COMPONENT COVERAGE (first, last, count)")
for name, df in [("G", G), ("I", I)]:
    for c in df:
        s = df[c].dropna()
        print(name, c, s.index.min().date(), s.index.max().date(), len(s))

print("\n== 2. GAP WINDOW (which components are NaN)")
print(G.loc["2025-06":"2026-10"].round(2))
print(I.loc["2025-06":"2026-10"].round(2))

print("\n== 3. RAW LEVELS, RECENT")
for sid in ["PAYEMS", "INDPRO", "RRSFS", "CPILFESL", "PCEPILFE", "T5YIFR"]:
    s = fred(sid).dropna()
    print(sid, "last obs:", s.index.max().date()); print(s.loc["2025-06":].round(2).to_string())
print("ICSA last 6:"); print(fred("ICSA").dropna().tail(6).to_string())

print("\n== 4. SHARE OF MONTHS AT THE CLIP (|z| >= 2.99), BY DECADE")
for name, df in [("G", G), ("I", I)]:
    z = zrobust(df)
    print(name); print((z.abs() >= 2.99).groupby(z.index.year // 10 * 10).mean().round(2))

print("\n== 5. MAD SCALE OVER TIME (every 2 years)")
def scale(s, w=120):
    return 1.4826 * s.rolling(w, min_periods=60).apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
print(pd.concat({c: scale(G[c]) for c in G}, axis=1).resample("2YE").last().round(2).dropna(how="all"))
print(pd.concat({c: scale(I[c]) for c in I}, axis=1).resample("2YE").last().round(2).dropna(how="all"))

print("\n== 6. COMPONENT CORRELATIONS (robust z, post-1999)")
print(zrobust(G).loc["1999":].corr().round(2)); print(zrobust(I).loc["1999":].corr().round(2))

print("\n== 7. CANDIDATE FRED SUBSTITUTES")
for sid in ["GACDFSA066MSFRBPHI", "GACDISA066MSFRBNY", "NOCDFSA066MSFRBPHI", "PCDFSA066MSFRBPHI",
            "MICH", "UMCSENT", "T10YIE", "T5YIE", "AWHMAN", "CPIAUCSL", "USREC"]:
    try:
        s = fred(sid).dropna()
        print(f"{sid:22s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")
    except Exception as e:
        print(f"{sid:22s} FAILED: {str(e)[:90]}")

print("\n== 8. SECTOR RETURNS")
r = sector_returns()
print(list(r.columns), r.index.min().date(), r.index.max().date(), len(r)); print(r.tail(3).round(4))

print("\n== 9. REGIME TAIL")
g, i, reg = build_regime()
print(pd.concat([g, i, reg], axis=1).loc["2025-07":].round(2))