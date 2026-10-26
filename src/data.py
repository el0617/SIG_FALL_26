"""Data layer: FRED + optional Bloomberg CSVs + sector ETF prices, all cached in data/raw/.

v2 changes
- API key comes from the FRED_API_KEY environment variable (Codespaces secret), never from code.
- level(): single missing months (e.g. Oct-2025, government shutdown) are interpolated and logged.
- fred_optional(): returns None instead of crashing when a series is unavailable.
- sector_returns(): adds IYR (real-estate proxy for SCHH) and XLC; re-downloads if the cache lacks them.
- complete_months(): drops the month in progress so partial returns never enter the statistics.
"""
import os
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


def _read(f):
    return pd.read_csv(f, index_col=0, parse_dates=True).squeeze("columns")


def fred(series_id, refresh=False):
    f = RAW / f"{series_id}.csv"
    if f.exists() and not refresh:
        return _read(f)
    s = _client().get_series(series_id)
    s.name = series_id
    s.to_csv(f)
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
            return r
        print("  [data] cached prices are missing some tickers -> re-downloading")
    import yfinance as yf
    px = yf.download(TICKERS, start="1998-12-01", auto_adjust=True, progress=False)["Close"]
    m = px.resample("ME").last()
    r = (m / m.shift(1) - 1).iloc[1:]
    r.to_csv(f)
    return r


def current_month_end():
    return (pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)).normalize()


def complete_months(r):
    """Drop the month in progress (a partial return is not a monthly return)."""
    return r[r.index < current_month_end()]