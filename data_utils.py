from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from Utility import counts, values_array

CSV_DEFAULT = "Taxi_Trips_-_2013_20250928.csv"
REQ_COLS = ["Taxi ID", "Trip Start Timestamp", "Pickup Community Area"]
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True)
class DemandTrace:
    history: np.ndarray
    demands: np.ndarray
    values: np.ndarray
    num_fb: int
    label: str = "synthetic"
    buyer_ids: tuple = ()

    def __post_init__(self):
        h = counts(self.history, "history", ndim=2).copy()
        d = counts(self.demands, "demands", ndim=2).copy()
        if not len(h) or not len(d) or h.shape[1] != d.shape[1]:
            raise ValueError("nonempty warmup and demand arrays must have identical buyer columns")
        if int(self.num_fb) != self.num_fb or not 0 <= self.num_fb <= d.shape[1]:
            raise ValueError("invalid num_fb")
        v = values_array(self.values, d.shape[1]).copy()
        if self.buyer_ids and len(self.buyer_ids) != len(v):
            raise ValueError("buyer IDs must match demand columns")
        for name, a in (("history", h), ("demands", d), ("values", v)):
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @property
    def digest(self):
        h = hashlib.sha256()
        for a in (self.history, self.demands, self.values):
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        h.update(str(self.num_fb).encode())
        h.update(str(self.buyer_ids).encode())
        return h.hexdigest()


def synthetic_trace(seed=0, slots=12, history_rows=12, num_fb=3, num_ob=2):
    """Small shifting demand DEMO, independent of policy. Zero OB demand = absent."""
    if slots < 1 or history_rows < 1:
        raise ValueError("positive slots/history_rows required")
    rng = np.random.default_rng(seed)
    total = history_rows + slots
    fb = rng.integers(2, 5, size=(total, num_fb))
    fb[history_rows + slots // 2:] += 2
    ob = rng.integers(1, 4, size=(total, num_ob)) * (rng.random((total, num_ob)) < 0.6)
    matrix = np.concatenate([fb, ob], axis=1)
    values = np.linspace(8.0, 10.0, num_fb + num_ob)
    return DemandTrace(matrix[:history_rows], matrix[history_rows:], values, num_fb,
                       f"synthetic seed={seed}", tuple(f"buyer-{i}" for i in range(len(values))))


def add_history_bounds(daily_table, window=30, window_mode="records"):
    """Past-only bounds; calendar mode inserts explicit zero-demand days."""
    if window < 1 or int(window) != window or window_mode not in ("records", "calendar"):
        raise ValueError("invalid history window")
    g = daily_table[["TaxiID", "Date", "NumTrips"]].copy()
    g["Date"] = pd.to_datetime(g["Date"]).dt.normalize()
    counts(g["NumTrips"].to_numpy(), "NumTrips")
    if g.duplicated(["TaxiID", "Date"]).any():
        raise ValueError("duplicate taxi/date rows")
    parts = []
    for taxi, group in g.groupby("TaxiID", sort=True):
        series = group.sort_values("Date").set_index("Date")["NumTrips"]
        if window_mode == "calendar":
            series = series.reindex(pd.date_range(series.index.min(), series.index.max()), fill_value=0)
        past = series.shift(1).rolling(window, min_periods=1)
        part = pd.DataFrame({"Date": series.index, "TaxiID": taxi, "NumTrips": series.to_numpy(),
                             "qmax": past.max().to_numpy(), "qmin": past.min().to_numpy()})
        parts.append(part)
    if not parts:
        raise ValueError("no valid taxi records")
    return pd.concat(parts, ignore_index=True)[["TaxiID", "Date", "NumTrips", "qmax", "qmin"]]


def preprocess_csv(csv_path=CSV_DEFAULT, community_area=77, window=30, use_chunks=True,
                   chunksize=500_000, window_mode="records"):
    reader = pd.read_csv(csv_path, usecols=REQ_COLS,
                         **({"chunksize": chunksize} if use_chunks else {}))
    chunks = reader if use_chunks else [reader]
    parts = []
    for chunk in chunks:
        c = chunk.loc[chunk["Pickup Community Area"] == community_area].copy()
        c["Date"] = pd.to_datetime(c["Trip Start Timestamp"], format=TS_FORMAT,
                                   errors="coerce").dt.normalize()
        c = c.dropna(subset=["Taxi ID", "Date"])
        if not c.empty:
            parts.append(c.groupby(["Taxi ID", "Date"]).size().rename("NumTrips").reset_index())
    if not parts:
        raise ValueError("no valid rows in requested community area")
    g = pd.concat(parts).groupby(["Taxi ID", "Date"], as_index=False)["NumTrips"].sum()
    return add_history_bounds(g.rename(columns={"Taxi ID": "TaxiID"}), window, window_mode)


def sample_tasks(daily_table, num_mu, date=None, replace_if_short=True, seed=0):
    df = daily_table.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values(["TaxiID", "Date"])
    pool = df[df["Date"] == pd.Timestamp(date)] if date is not None else df.groupby("TaxiID").tail(1)
    # Missing date must never fall forward to the latest/future record.
    pool = pool.dropna(subset=["qmin", "qmax"])
    if pool.empty:
        raise ValueError("no observations with prior history on the requested date")
    sel = pool.sample(n=num_mu, replace=(replace_if_short and len(pool) < num_mu), random_state=seed)
    return dict(NumTask=sel["NumTrips"].to_numpy(dtype=int), qmax=sel["qmax"].to_numpy(dtype=int),
                qmin=sel["qmin"].to_numpy(dtype=int), TaxiID=sel["TaxiID"].astype(str).to_numpy(),
                Date=sel["Date"].astype(str).to_numpy())


def split_chronologically(matrix, values, num_fb, warmup=30, label="local CSV", buyer_ids=()):
    """60/20/20 outcomes; each split uses only preceding rows as its warmup."""
    matrix = counts(matrix, "matrix", ndim=2)
    remaining = len(matrix) - warmup
    if warmup < 1 or remaining < 3:
        raise ValueError("need warmup + at least 3 chronological observations")
    n_train = max(1, int(remaining * 0.6))
    n_val = max(1, int(remaining * 0.2))
    cuts = [warmup, warmup + n_train, warmup + n_train + n_val, len(matrix)]
    return {name: DemandTrace(matrix[max(0, start-warmup):start], matrix[start:end],
                              values, num_fb, f"{label} {name}", tuple(buyer_ids))
            for name, start, end in zip(("train", "validation", "test"), cuts, cuts[1:])}


def load_trace_csv(path, num_fb=3, warmup=30, *, raw_taxi=False, num_buyers=5):
    """Load local chronological demands; valuations remain demo inputs."""
    path = Path(path)
    if raw_taxi:
        daily = preprocess_csv(path, window=warmup, window_mode="calendar")
        dates = pd.date_range(daily["Date"].min(), daily["Date"].max())
        if len(dates) <= warmup:
            raise ValueError("not enough taxi dates")
        # Fix identities from warmup only; never select taxis for future activity.
        ids = sorted(daily.loc[daily["Date"] < dates[warmup], "TaxiID"].unique(), key=str)[:num_buyers]
        if len(ids) < num_fb:
            raise ValueError("not enough warmup buyers")
        wide = daily.pivot(index="Date", columns="TaxiID", values="NumTrips").reindex(index=dates, columns=ids).fillna(0)
    else:
        wide = pd.read_csv(path, index_col="Date")
        dates = pd.to_datetime(wide.index)
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Date must be unique and chronological")
        if wide.isna().any().any():
            raise ValueError("missing demand entries require explicit preprocessing")
    return split_chronologically(wide.to_numpy(), np.linspace(8, 10, wide.shape[1]), num_fb,
                                 warmup, str(path.name) + " (demo valuations)", wide.columns)
