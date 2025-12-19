from __future__ import annotations
import pandas as pd

PRICE_COLS = [
    "symbol", "date", "open", "high", "low", "close", "close_adjusted", "volume", "split_coefficient"
]

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in PRICE_COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "close_adjusted"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # numeric coercion (protect against weird strings)
    for c in ["open","high","low","close","close_adjusted","volume","split_coefficient"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close_adjusted"])
    return df

def load_earnings(path: str) -> pd.DataFrame:
    e = pd.read_csv(path)
    # try common column names; we’ll normalize
    # keep only symbol + earnings date for v1
    cols = {c.lower(): c for c in e.columns}

    sym_col = cols.get("symbol", None)
    date_col = cols.get("date", None) or cols.get("earnings_date", None) or cols.get("reporteddate", None)

    if sym_col is None or date_col is None:
        raise ValueError(f"Could not find symbol/date columns in earnings file. Columns: {list(e.columns)}")

    e = e[[sym_col, date_col]].rename(columns={sym_col: "symbol", date_col: "earnings_date"})
    e["earnings_date"] = pd.to_datetime(e["earnings_date"], errors="coerce")
    e = e.dropna(subset=["symbol", "earnings_date"])
    e = e.sort_values(["symbol", "earnings_date"]).drop_duplicates()
    return e
