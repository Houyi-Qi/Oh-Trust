# data_utils.py
# ------------------------------------------------------------
# Real-world dataset (Chicago Taxi Trips) processing module.
# This script is designed to work with the dataset whose header
# strictly matches the following fields:
#
# "Trip ID","Taxi ID","Trip Start Timestamp","Trip End Timestamp",
# "Trip Seconds","Trip Miles","Pickup Census Tract","Dropoff Census Tract",
# "Pickup Community Area","Dropoff Community Area","Fare","Tips","Tolls",
# "Extras","Trip Total","Payment Type","Company",
# "Pickup Centroid Latitude","Pickup Centroid Longitude",
# "Pickup Centroid Location","Dropoff Centroid Latitude",
# "Dropoff Centroid Longitude","Dropoff Centroid Location"
#
# Functions:
# 1) Read only the required fields: Taxi ID / Trip Start Timestamp / Pickup Community Area.
# 2) Filter rows where PU_CA == 77 (community area).
# 3) Aggregate data by (TaxiID, Date) to compute daily trip counts NumTrips.
# 4) Compute rolling-window statistics (default = 30 days):
#       qmax = rolling maximum of NumTrips,
#       qmin = rolling minimum of NumTrips.
# 5) Provide sample_tasks(): sample a set of MUs (mobile users)
#    with their daily task counts and bounds (qmax/qmin).
#
# Key improvements:
# - Explicitly specify datetime format ("%m/%d/%Y %I:%M:%S %p")
#   to avoid pandas UserWarning and ensure fast parsing.
# - Support chunked reading to reduce memory consumption.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

CSV_DEFAULT = "Taxi_Trips_-_2013_20250928.csv"
REQ_COLS = ["Taxi ID", "Trip Start Timestamp", "Pickup Community Area"]

# Explicit timestamp format: e.g., "01/01/2013 12:30:00 AM"
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def preprocess_csv(
    csv_path: str = CSV_DEFAULT,
    community_area: int = 77,
    window: int = 30,
    use_chunks: bool = True,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Load and preprocess the raw taxi dataset.

    Steps:
        1. Read required columns from CSV (chunked or full read).
        2. Filter by specified community_area (default: 77).
        3. Convert timestamps to datetime with explicit format.
        4. Aggregate trips into daily counts per TaxiID.
        5. Compute rolling statistics (qmax, qmin).

    Args:
        csv_path: str, path to the CSV file.
        community_area: int, filter value for Pickup Community Area.
        window: int, rolling window size for qmax/qmin.
        use_chunks: bool, whether to read file in chunks.
        chunksize: int, number of rows per chunk if use_chunks=True.

    Returns:
        pd.DataFrame with columns:
            ["TaxiID", "Date", "NumTrips", "qmax", "qmin"]
    """
    def _read_chunks() -> pd.DataFrame:
        parts = []
        for chunk in pd.read_csv(csv_path, usecols=REQ_COLS, chunksize=chunksize, low_memory=False):
            c = chunk.rename(columns={
                "Taxi ID": "TaxiID",
                "Trip Start Timestamp": "StartTS",
                "Pickup Community Area": "PU_CA",
            })
            c = c[c["PU_CA"] == community_area]
            c["StartTS"] = pd.to_datetime(c["StartTS"], format=TS_FORMAT, errors="coerce")
            c = c.dropna(subset=["StartTS", "TaxiID"])
            c["Date"] = c["StartTS"].dt.date
            parts.append(c[["TaxiID", "Date"]])
        if not parts:
            raise ValueError("No valid rows found after filtering; check CSV path and community_area.")
        return pd.concat(parts, ignore_index=True)

    def _read_all() -> pd.DataFrame:
        df = pd.read_csv(csv_path, usecols=REQ_COLS, low_memory=False)
        df = df.rename(columns={
            "Taxi ID": "TaxiID",
            "Trip Start Timestamp": "StartTS",
            "Pickup Community Area": "PU_CA",
        })
        df = df[df["PU_CA"] == community_area]
        df["StartTS"] = pd.to_datetime(df["StartTS"], format=TS_FORMAT, errors="coerce")
        df = df.dropna(subset=["StartTS", "TaxiID"])
        df["Date"] = df["StartTS"].dt.date
        return df[["TaxiID", "Date"]]

    base = _read_chunks() if use_chunks else _read_all()

    # Aggregate daily trips
    g = base.groupby(["TaxiID", "Date"]).size().reset_index(name="NumTrips")
    g = g.sort_values(["TaxiID", "Date"]).reset_index(drop=True)

    # Rolling statistics within each TaxiID
    g["Date"] = pd.to_datetime(g["Date"])
    g["qmax"] = g.groupby("TaxiID")["NumTrips"].transform(lambda x: x.rolling(window, min_periods=1).max())
    g["qmin"] = g.groupby("TaxiID")["NumTrips"].transform(lambda x: x.rolling(window, min_periods=1).min())
    g["Date"] = g["Date"].dt.date

    return g[["TaxiID", "Date", "NumTrips", "qmax", "qmin"]]


def sample_tasks(
    daily_table: pd.DataFrame,
    num_mu: int,
    date: Optional[str] = None,
    replace_if_short: bool = True,
) -> Dict[str, Any]:
    """
    Sample a batch of MUs (mobile users) from the daily table.

    If a specific date is provided, select rows for that date.
    Otherwise, take the latest record for each TaxiID.

    Args:
        daily_table: pd.DataFrame, result of preprocess_csv().
        num_mu: int, number of users to sample.
        date: Optional[str], target date (format: YYYY-MM-DD).
        replace_if_short: bool, allow sampling with replacement
                          if available rows < num_mu.

    Returns:
        dict containing:
            - NumTask: np.ndarray[int], sampled daily task counts.
            - qmax: np.ndarray[int], rolling maximum values.
            - qmin: np.ndarray[int], rolling minimum values.
            - TaxiID: np.ndarray[str], corresponding Taxi IDs.
            - Date: np.ndarray[str], corresponding Dates.
    """
    df = daily_table.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values(["TaxiID", "Date"])

    if date:
        tgt = pd.to_datetime(date).date()
        pool = df[df["Date"] == tgt]
        if pool.empty:
            pool = df.groupby("TaxiID").tail(1)
    else:
        pool = df.groupby("TaxiID").tail(1)

    if pool.empty:
        raise ValueError("Sampling pool is empty; check daily_table or date.")

    sel = pool.sample(
        n=num_mu,
        replace=(replace_if_short and len(pool) < num_mu),
        random_state=None
    )

    return dict(
        NumTask=sel["NumTrips"].astype(int).to_numpy(),
        qmax=sel["qmax"].astype(int).to_numpy(),
        qmin=sel["qmin"].astype(int).to_numpy(),
        TaxiID=sel["TaxiID"].astype(str).to_numpy(),
        Date=sel["Date"].astype(str).to_numpy(),
    )


# ---- Manual test (optional) ----
if __name__ == "__main__":
    daily = preprocess_csv(
        csv_path=CSV_DEFAULT,
        community_area=77,
        window=30,
        use_chunks=True,
        chunksize=500_000,
    )
    print("[OK] daily shape:", daily.shape, "; columns:", list(daily.columns))
    batch = sample_tasks(daily, num_mu=30)
    print("NumTask[:10]:", batch["NumTask"][:10])
    print("qmax[:10]:", batch["qmax"][:10])
    print("qmin[:10]:", batch["qmin"][:10])
