
import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from fredapi import Fred

# Paste your FRED API key here
FRED_API_KEY = "REDACTED"

# Initialize FRED client
fred_client = Fred(api_key=FRED_API_KEY)

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


def fred(series_id, refresh=False):
    f = RAW / f"{series_id}.csv"

    if f.exists() and not refresh:
        return pd.read_csv(
            f, index_col=0, parse_dates=True
        ).squeeze("columns")

    s = fred_client.get_series(series_id)
    s.name = series_id
    s.to_csv(f)

    return s


def monthly(s, how="last"):
    """Collapse any frequency to month-end."""
    return s.dropna().resample("ME").agg(how)


def optional_csv(name):
    """Bloomberg exports: data/raw/<name>.csv with columns date,value."""
    f = RAW / f"{name}.csv"

    if not f.exists():
        return None

    return pd.read_csv(
        f, index_col=0, parse_dates=True
    ).squeeze("columns")


def sector_returns(refresh=False):
    f = RAW / "prices_monthly.csv"

    if f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True)

    tickers = [
        "SPY", "XLK", "XLF", "XLE", "XLV",
        "XLY", "XLP", "XLI", "XLB", "XLU"
    ]

    px = yf.download(
        tickers,
        start="1998-12-01",
        auto_adjust=True
    )["Close"]

    r = px.resample("ME").last().pct_change().dropna(how="all")
    r.to_csv(f)

    return r
