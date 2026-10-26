"""Import Bloomberg exports into data/raw/ and validate them.

Usage (from the repo root):
    python -m src.import_bloomberg data/raw/bloomberg.xlsx     # one workbook, one sheet per ticker
    python -m src.import_bloomberg                             # just re-validate the existing CSVs

The workbook can have the four tickers on separate sheets OR side by side on one sheet; each block
must be a date column followed by a value column (what BDH with Dts=S returns). The importer finds
each ticker by name (sheet name or a header cell containing the ticker), cleans the data
(#N/A, blanks, text dates, Excel serial dates, duplicates), and writes clean two-column CSVs:

    NAPMNEWO Index  -> data/raw/ism_new_orders.csv
    NAPMPRIC Index  -> data/raw/ism_prices.csv
    NAPMINV  Index  -> data/raw/ism_inventories.csv
    LF98OAS  Index  -> data/raw/hy_oas.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

SPECS = {
    # ticker key: (output name, description, plausible range, typical frequency)
    "NAPMNEWO": ("ism_new_orders", "ISM manufacturing new orders (diffusion index)", (15, 90), "monthly"),
    "NAPMPRIC": ("ism_prices", "ISM manufacturing prices paid (diffusion index)", (10, 100), "monthly"),
    "NAPMINV":  ("ism_inventories", "ISM manufacturing inventories (diffusion index)", (20, 80), "monthly"),
    "LF98OAS":  ("hy_oas", "Bloomberg US Corporate High Yield OAS (basis points)", (150, 2500), "daily/monthly"),
}


# ------------------------------------------------------------------ parsing
def _to_dates(col):
    s = pd.Series(col)
    num = pd.to_numeric(s, errors="coerce")
    serial = num.between(20000, 80000)                       # Excel serial day numbers
    out = pd.to_datetime(s.where(~serial).astype(str), errors="coerce", format="mixed")
    out[serial] = pd.to_datetime("1899-12-30") + pd.to_timedelta(num[serial], unit="D")
    return out


def _to_values(col):
    s = pd.Series(col).astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.replace({"#N/A": np.nan, "#N/A N/A": np.nan, "": np.nan, "nan": np.nan}),
                         errors="coerce")


def _blocks(df):
    """Yield (label, Series) for every date-column + value-column pair in a raw sheet."""
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    cols = list(df.columns)
    for k in range(len(cols) - 1):
        d, v = _to_dates(df[cols[k]]), _to_values(df[cols[k + 1]])
        ok = d.notna() & v.notna() & (d.dt.year.between(1900, 2100))
        if ok.sum() < 24:
            continue
        # label = any text found above the data in these two columns (ticker header), if present
        first = ok.idxmax()
        header = " ".join(str(x) for x in df.loc[:first, [cols[k], cols[k + 1]]].values.ravel()
                          if isinstance(x, str))
        s = pd.Series(v[ok].values, index=pd.DatetimeIndex(d[ok].values), name="value").sort_index()
        yield header, s[~s.index.duplicated(keep="last")]


def _match(text):
    # tolerate common typing slips: spaces, and digit zero typed for letter O (LF980AS -> LF98OAS)
    t = str(text).upper().replace(" ", "").replace("0", "O")
    for key in SPECS:
        # NAPMINV must not match NAPMINVT-style variants of other tickers incorrectly; all keys are distinct
        if key in t:
            return key
    return None


def parse_workbook(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    else:
        sep = ";" if ";" in path.read_text(errors="ignore")[:2000] else ","
        sheets = {path.stem: pd.read_csv(path, header=None, sep=sep, dtype=str)}
    found = {}
    for sheet_name, df in sheets.items():
        blocks = list(_blocks(df))
        for k, (header, s) in enumerate(blocks):
            key = _match(header) or (_match(sheet_name) if len(blocks) == 1 else None)
            if key is None:
                print(f"  [skip] sheet '{sheet_name}', block {k + 1}: no ticker name found "
                      f"(put the ticker in the sheet name or in a cell above the data)")
                continue
            if key in found:
                print(f"  [warn] {key} found twice; keeping the longer history")
                if len(s) <= len(found[key]):
                    continue
            found[key] = s
            print(f"  [found] {key:9s} in sheet '{sheet_name}': {len(s)} rows, "
                  f"{s.index.min():%Y-%m-%d} -> {s.index.max():%Y-%m-%d}")
    return found


# ------------------------------------------------------------------ validation
def validate(key, s):
    name, desc, (lo, hi), _ = SPECS[key]
    problems, notes = [], []
    if len(s) < 60:
        problems.append(f"only {len(s)} observations")
    med = s.median()
    if key == "LF98OAS" and med < 30:
        problems.append("values look like PERCENT, not basis points; re-import with this script to convert")
    elif not (lo <= med <= hi):
        problems.append(f"median {med:.1f} outside plausible range {lo}-{hi}: wrong ticker or field?")
    if s.index.has_duplicates:
        problems.append("duplicate dates")
    gap_days = s.index.to_series().diff().dt.days.max()
    if key != "LF98OAS" and gap_days and gap_days > 62:
        notes.append(f"largest gap between observations: {gap_days} days")
    last = s.index.max()
    today = pd.Timestamp.today().normalize()
    if key != "LF98OAS":
        expected = (today - pd.offsets.MonthBegin(1)) - pd.offsets.Day(1)   # end of previous month
        if last.to_period("M") > (today - pd.offsets.MonthBegin(1)).to_period("M"):
            problems.append(f"last date {last:%Y-%m-%d} is in the CURRENT month: Bloomberg may be stamping "
                            "release dates, not reference months. Check on the terminal (see instructions)")
        elif last.to_period("M") < (expected - pd.offsets.MonthEnd(1)).to_period("M"):
            notes.append(f"last observation {last:%Y-%m} looks stale; refresh the BDH before exporting")
    elif (today - last).days > 10:
        notes.append(f"last observation {last:%Y-%m-%d} is more than 10 days old")
    return problems, notes


def cross_check(found):
    """Sanity: the ISM series should co-move with their free FRED cousins."""
    pairs = [("NAPMNEWO", "NOCDFSA066MSFRBPHI", "Philly Fed new orders"),
             ("LF98OAS", "BAMLH0A0HYM2", "ICE BofA HY OAS (FRED)")]
    for key, fred_id, label in pairs:
        f = RAW / f"{fred_id}.csv"
        if key not in found or not f.exists():
            continue
        other = pd.read_csv(f, index_col=0, parse_dates=True).squeeze("columns")
        a = found[key].resample("ME").mean()
        b = other.dropna().resample("ME").mean()
        both = pd.concat([a, b], axis=1).dropna()
        if len(both) > 36:
            c = both.corr().iloc[0, 1]
            flag = "" if c > 0.4 else "   <-- WARNING: too low; wrong ticker, field, or date alignment?"
            print(f"  [check] corr({key}, {label}) = {c:.2f} over {len(both)} months (expect > 0.6){flag}")


def write(found):
    for key, s in found.items():
        if key == "LF98OAS" and s.median() < 30:
            print("  [units] LF98OAS is in percent (e.g. 3.37); converting to basis points (x100)")
            s = s * 100
        name = SPECS[key][0]
        out = s.rename("value")
        out.index.name = "date"
        out.to_csv(RAW / f"{name}.csv", date_format="%Y-%m-%d")
        print(f"  [write] data/raw/{name}.csv")


def load_existing():
    found = {}
    for key, (name, *_ ) in SPECS.items():
        f = RAW / f"{name}.csv"
        if f.exists():
            found[key] = pd.read_csv(f, index_col=0, parse_dates=True).squeeze("columns").dropna()
    return found


def main(argv):
    if len(argv) > 1:
        print(f"Reading {argv[1]} ...")
        found = parse_workbook(argv[1])
        write(found)
    else:
        print("No file given: validating existing CSVs in data/raw/")
    found = load_existing()

    print("\nVALIDATION")
    ok = True
    for key, (name, desc, *_ ) in SPECS.items():
        if key not in found:
            print(f"  [MISSING] {key:9s} -> data/raw/{name}.csv  ({desc})")
            ok = False
            continue
        s = found[key]
        problems, notes = validate(key, s)
        status = "FAIL" if problems else "ok"
        ok &= not problems
        print(f"  [{status:4s}] {key:9s} {s.index.min():%Y-%m} -> {s.index.max():%Y-%m-%d}  n={len(s):5d}  "
              f"median={s.median():8.1f}  min={s.min():8.1f}  max={s.max():8.1f}")
        for p in problems:
            print(f"           PROBLEM: {p}")
        for n in notes:
            print(f"           note: {n}")
        if key != "LF98OAS":
            print("           last 3: " + ", ".join(f"{d:%Y-%m}={v:.1f}" for d, v in s.tail(3).items()))
    cross_check(found)
    print("\nRESULT:", "all four series ready -> run  python -m src.regime" if ok
          else "fix the items above, then rerun this command")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))