from __future__ import annotations
import pandas as pd

def add_earnings_distance(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    e = earnings.copy()

    # --- Clean + type enforcement ---
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p.dropna(subset=["symbol", "date"]).copy()
    p = p.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    e["earnings_date"] = pd.to_datetime(e["earnings_date"], errors="coerce")
    e = e.dropna(subset=["symbol", "earnings_date"]).copy()
    e = (
        e.sort_values(["symbol", "earnings_date"], kind="mergesort")
         .drop_duplicates(["symbol", "earnings_date"])
         .reset_index(drop=True)
    )

    out_parts = []

    # --- Merge per symbol (robust against pandas merge_asof issues) ---
    for sym, ps in p.groupby("symbol", sort=False):
        es = e[e["symbol"] == sym][["earnings_date"]].sort_values(
            "earnings_date", kind="mergesort"
        )
        ps = ps.sort_values("date", kind="mergesort").reset_index(drop=True)

        if es.empty:
            ps["next_earnings_date"] = pd.NaT
            ps["prev_earnings_date"] = pd.NaT
        else:
            es_next = es.rename(columns={"earnings_date": "next_earnings_date"})
            es_prev = es.rename(columns={"earnings_date": "prev_earnings_date"})

            # forward (next earnings)
            m_next = pd.merge_asof(
                ps[["date"]],
                es_next,
                left_on="date",
                right_on="next_earnings_date",
                direction="forward",
                allow_exact_matches=True,
            )

            # backward (previous earnings)
            m_prev = pd.merge_asof(
                ps[["date"]],
                es_prev,
                left_on="date",
                right_on="prev_earnings_date",
                direction="backward",
                allow_exact_matches=True,
            )

            ps["next_earnings_date"] = m_next["next_earnings_date"].values
            ps["prev_earnings_date"] = m_prev["prev_earnings_date"].values

        # --- Distance features ---
        ps["days_to_earnings"] = (ps["next_earnings_date"] - ps["date"]).dt.days
        ps["days_since_earnings"] = (ps["date"] - ps["prev_earnings_date"]).dt.days

        # 🔧 IMPORTANT FIX: remove NaNs so downstream logic is deterministic
        ps["days_to_earnings"] = ps["days_to_earnings"].fillna(99999).astype(int)
        ps["days_since_earnings"] = ps["days_since_earnings"].fillna(99999).astype(int)

        # earnings window flag (±5 days)
        ps["is_earnings_window"] = (
            ps["days_to_earnings"].between(0, 5)
            | ps["days_since_earnings"].between(0, 5)
        )

        out_parts.append(ps)

    out = pd.concat(out_parts, ignore_index=True)
    return out


