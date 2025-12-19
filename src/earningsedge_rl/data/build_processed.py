from __future__ import annotations
import os
import pandas as pd

from earningsedge_rl.data.load import load_prices, load_earnings
from earningsedge_rl.features.earnings import add_earnings_distance

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/panel.parquet"

def main():
    prices_path = os.path.join(RAW_DIR, "stock_prices_latest.csv")
    earnings_path = os.path.join(RAW_DIR, "earnings_latest.csv")

    prices = load_prices(prices_path)
    earnings = load_earnings(earnings_path)

    panel = add_earnings_distance(prices, earnings)
    panel = panel.rename(columns={"close_adjusted": "adj_close"})

    # Use adjusted close for return calcs later; keep a clean name too
    panel = panel.rename(columns={"close_adjusted": "adj_close"})

    # Basic sanity: remove symbols with too few rows
    counts = panel.groupby("symbol")["date"].count()
    keep = counts[counts >= 252].index  # at least ~1 year
    panel = panel[panel["symbol"].isin(keep)].reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(panel), "Symbols:", panel["symbol"].nunique())
    print(panel.head())

if __name__ == "__main__":
    main()
