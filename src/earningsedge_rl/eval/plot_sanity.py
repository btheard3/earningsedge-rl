import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_parquet("data/processed/panel.parquet")

    # pick a well-known ticker
    sym = "MSFT"
    d = df[df["symbol"] == sym].sort_values("date").tail(1500)

    price_col = "adj_close"

    plt.figure(figsize=(12, 5))
    plt.plot(d["date"], d[price_col], label="Adj Close")

    # highlight earnings windows
    e = d[d["is_earnings_window"] == True]
    plt.scatter(
        e["date"],
        e[price_col],
        color="red",
        s=15,
        label="Earnings Window"
    )

    plt.title(f"{sym} — Adjusted Close with Earnings Windows")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
