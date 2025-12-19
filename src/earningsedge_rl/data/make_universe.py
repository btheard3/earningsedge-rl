import pandas as pd

def main():
    df = pd.read_parquet("data/processed/panel.parquet")
    price_col = "adj_close" if "adj_close" in df.columns else "close_adjusted"
    df["dollar_vol"] = df[price_col] * df["volume"]

    top = (
        df.groupby("symbol")["dollar_vol"]
          .median()
          .sort_values(ascending=False)
          .head(200)
          .index
    )
    pd.DataFrame({"symbol": top}).to_csv("data/processed/universe_top200.csv", index=False)
    print("Saved data/processed/universe_top200.csv")

if __name__ == "__main__":
    main()
