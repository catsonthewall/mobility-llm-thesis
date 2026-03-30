# ---------------------------------------------------------------------------
# src/data/sample.py
# Sample one continuous week of staypoints per user
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def sample_one_week_per_user(sp: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    For each user, find a random contiguous 7-day window and return
    only staypoints within that window.

    Rules:
    - User must have at least 7 consecutive days of data
    - Days must be consecutive (no gap > 1 day between them)
    - Start date is chosen randomly among valid 7-day windows

    Parameters
    ----------
    sp   : DataFrame with columns [user_id, date, ...]
           'date' should be a date or datetime column
    seed : random seed for reproducibility

    Returns
    DataFrame with same columns as sp, filtered to one week per user
    """
    rng = np.random.default_rng(seed)
    sp = sp.copy()
    sp["date"] = pd.to_datetime(sp["date"])

    out = []
    skipped = 0

    for uid, df_u in sp.groupby("user_id"):
        days = (
            df_u[["date"]]
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(days) < 7:
            skipped += 1
            continue

        # Find consecutive blocks
        days["delta"] = days["date"].diff().dt.days.fillna(1).astype(int)
        days["block"] = (days["delta"] > 1).cumsum()

        # Find blocks with >= 7 consecutive days
        valid = days.groupby("block").filter(lambda x: len(x) >= 7)

        if valid.empty:
            skipped += 1
            continue

        # Collect all valid 7-day start dates
        candidate_starts = []
        for _, g in valid.groupby("block"):
            block_dates = g["date"].sort_values().reset_index(drop=True)
            for i in range(len(block_dates) - 6):
                candidate_starts.append(block_dates.iloc[i])

        start_date = rng.choice(candidate_starts)
        week_dates = pd.date_range(start=start_date, periods=7, freq="D")

        out.append(df_u[df_u["date"].isin(week_dates)])

    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=sp.columns)
    print(f"Sampled {len(result)} rows from {len(out)} users ({skipped} skipped — <7 days)")
    return result


# --------------------------------------
# CLI test
# --------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.data.load_data import load_staypoints

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/data/baliu/python_code/data/sp_all copy.csv")

    sp = load_staypoints(path)
    sp_week = sample_one_week_per_user(sp)
    print(f"Shape after sampling: {sp_week.shape}")
    print(sp_week[["user_id", "date"]].drop_duplicates().groupby("user_id").size().describe())
