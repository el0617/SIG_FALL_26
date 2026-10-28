"""Data layer: FRED + optional Bloomberg CSVs + sector ETF prices, all cached in data/raw/.

v2 changes
- API key comes from the FRED_API_KEY environment variable (Codespaces secret), never from code.
- level(): single missing months (e.g. Oct-2025, government shutdown) are interpolated and logged.
- fred_optional(): returns None instead of crashing when a series is unavailable.
- sector_returns(): adds IYR (real-estate proxy for SCHH) and XLC; re-downloads if the cache lacks them.
- complete_months(): drops the month in progress so partial returns never enter the statistics.

v3 changes (code review, before the data freeze)
- Every download is recorded in data/raw/_downloads.json (file -> download time).
- Cached prices drop the month that was still in progress when they were downloaded. Without this,
  a cache pulled on Oct 22 kept a partial October row that complete_months() accepted as complete
  once November started.
- `python -m src.data --status` lists every cache with its last observation and download time;
  `python -m src.data --refresh` re-downloads all of them (FRED, prices, Ken French). Bloomberg files
  are never touched: re-import those with src.import_bloomberg / src.valuation.
"""
import os
import re
import sys
import json
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

FILL_LOG = {}   # series_id -> months filled by interpolation (disclose on the methodology slide)

TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "IYR", "XLC"]


def _client():
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it as a Codespaces secret (GitHub Settings -> Codespaces "
            "-> Secrets) and reopen the Codespace, or run:  export FRED_API_KEY=your_key")
    return Fred(api_key=key)


MANIFEST = RAW / "_downloads.json"
FRED_FILE = re.compile(r"^[A-Z0-9]+\.csv$")   # FRED caches are upper-case ids; Bloomberg exports are lower-case


def _today():
    return pd.Timestamp.today()


def _manifest():
    try:
        return json.loads(MANIFEST.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def stamp(f):
    """Record that file f was downloaded now."""
    m = _manifest()
    m[Path(f).name] = _today().isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True))


def downloaded_at(f):
    """When file f was downloaded: from the manifest, else (older caches) the file's modification time."""
    t = _manifest().get(Path(f).name)
    if t:
        return pd.Timestamp(t), "manifest"
    return pd.Timestamp(Path(f).stat().st_mtime, unit="s"), "file time"


def drop_unfinished(df, f):
    """Drop month-end rows for months that were still in progress when f was downloaded."""
    at, how = downloaded_at(f)
    cut = (at.normalize() + pd.offsets.MonthEnd(0)).normalize()   # month-end of the download month
    bad = df.index >= cut
    if bad.any():
        print(f"  [data] {Path(f).name}: dropped {', '.join(f'{d:%Y-%m}' for d in df.index[bad])} "
              f"(month in progress when downloaded {at:%Y-%m-%d}, per {how})")
    return df[~bad]


def warn_if_stale(df, f):
    """Warn when a cache ends before the last complete month."""
    want = current_month_end() - pd.offsets.MonthEnd(1)
    have = df.dropna(how="all").index.max()
    if have < want:
        print(f"  [data] WARNING {Path(f).name} ends {have:%Y-%m}; last complete month is {want:%Y-%m}. "
              f"Run: python -m src.data --refresh")


def _read(f):
    return pd.read_csv(f, index_col=0, parse_dates=True).squeeze("columns")


def fred(series_id, refresh=False):
    f = RAW / f"{series_id}.csv"
    if f.exists() and not refresh:
        return _read(f)
    s = _client().get_series(series_id)
    s.name = series_id
    s.to_csv(f)
    stamp(f)
    return s


def fred_optional(series_id):
    """Like fred(), but returns None (with a note) if the series can't be fetched."""
    try:
        return fred(series_id)
    except Exception as e:
        print(f"  [data] optional series {series_id} unavailable: {str(e)[:70]}")
        return None


def monthly(s, how="last"):
    """Collapse any frequency to month-end."""
    return s.dropna().resample("ME").agg(how)


def fill_single_gaps(s, name):
    """Linearly interpolate isolated one-month holes only (longer gaps stay NaN)."""
    hole = s.isna() & s.shift(1).notna() & s.shift(-1).notna()
    if hole.any():
        s = s.copy()
        s[hole] = ((s.shift(1) + s.shift(-1)) / 2)[hole]
        FILL_LOG[name] = [d.strftime("%Y-%m") for d in s.index[hole]]
    return s


def level(series_id, how="last", optional=False):
    """Month-end level of a FRED series with single-month holes filled."""
    raw = fred_optional(series_id) if optional else fred(series_id)
    if raw is None:
        return None
    return fill_single_gaps(monthly(raw, how), series_id)


def optional_csv(name):
    """Bloomberg exports: data/raw/<name>.csv with exactly two columns: date,value."""
    f = RAW / f"{name}.csv"
    if not f.exists():
        return None
    s = _read(f)
    if isinstance(s, pd.DataFrame):
        raise ValueError(f"{f} must have exactly two columns (date,value); found {list(s.columns)}")
    return pd.to_numeric(s, errors="coerce").dropna()


def sector_returns(refresh=False):
    f = RAW / "prices_monthly.csv"
    if f.exists() and not refresh:
        r = pd.read_csv(f, index_col=0, parse_dates=True)
        if set(TICKERS) <= set(r.columns):
            r = drop_unfinished(r, f)
            warn_if_stale(r, f)
            return r
        print("  [data] cached prices are missing some tickers -> re-downloading")
    import yfinance as yf
    px = yf.download(TICKERS, start="1998-12-01", auto_adjust=True, progress=False)["Close"]
    m = px.resample("ME").last()
    r = (m / m.shift(1) - 1).iloc[1:]
    r.to_csv(f)
    stamp(f)
    return drop_unfinished(r, f)


def current_month_end():
    return (_today().normalize() + pd.offsets.MonthEnd(0)).normalize()


def complete_months(r):
    """Drop the month in progress (a partial return is not a monthly return)."""
    return r[r.index < current_month_end()]

# ------------------------------------------------------------------ cache maintenance
def _caches():
    """(file, kind) for every download cache in data/raw."""
    out = [(f, "FRED") for f in sorted(RAW.glob("*.csv")) if FRED_FILE.match(f.name)]
    for name, kind in [("prices_monthly.csv", "Yahoo (sector ETFs)"), ("market_prices.csv", "Yahoo (SPY/IWM/gold/copper)"),
                       ("holdings_prices.csv", "Yahoo (VGT/SCHH)"), ("ff_factors.csv", "Ken French")]:
        if (RAW / name).exists():
            out.append((RAW / name, kind))
    return out


def status():
    """Print each cache's last observation and when it was downloaded."""
    rows = []
    for f, kind in _caches():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        at, how = downloaded_at(f)
        rows.append({"file": f.name, "source": kind, "last obs": f"{df.dropna(how='all').index.max():%Y-%m-%d}",
                     "downloaded": f"{at:%Y-%m-%d %H:%M}", "per": how})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if (t["per"] == "file time").any():
        print("\n  'file time' = no manifest entry; the date is the file's modification time, which a fresh git "
              "clone resets. Run --refresh to record real download times.")


def refresh_all():
    """Re-download every cache: all FRED series already cached, the three price files, Ken French."""
    fails = []
    for f, kind in _caches():
        if kind != "FRED":
            continue
        try:
            fred(f.stem, refresh=True)
            print(f"  [refresh] FRED {f.stem}")
        except Exception as e:
            fails.append(f"FRED {f.stem}: {str(e)[:70]}")
    sector_returns(refresh=True); print("  [refresh] prices_monthly.csv")
    from src.market import extra_prices
    extra_prices(refresh=True); stamp(RAW / "market_prices.csv"); print("  [refresh] market_prices.csv")
    from src.portfolio import etf_returns
    etf_returns(refresh=True); stamp(RAW / "holdings_prices.csv"); print("  [refresh] holdings_prices.csv")
    from src.factors import french_factors
    if french_factors(refresh=True) is not None:
        stamp(RAW / "ff_factors.csv"); print("  [refresh] ff_factors.csv")
    else:
        fails.append("Ken French download failed (see message above)")
    print("\n" + ("  FAILED:\n    " + "\n    ".join(fails) if fails else "  all downloads succeeded"))
    print("  Bloomberg files were not touched: re-run src.import_bloomberg and src.valuation on fresh exports.\n")
    status()


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        refresh_all()
    else:
        status()
