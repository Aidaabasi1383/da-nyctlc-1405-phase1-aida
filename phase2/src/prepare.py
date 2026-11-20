"""Reproduce the Phase 1 cleaning funnel and build Phase 2 features."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITY_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}
TRIP_FILE = "yellow_tripdata_2024-01.parquet"
ZONE_FILE = "taxi_zone_lookup.csv"


def load_and_clean(raw_dir, config):
    trips = pd.read_parquet(raw_dir / TRIP_FILE)
    zones = pd.read_csv(raw_dir / ZONE_FILE).set_index("LocationID", verify_integrity=True)
    raw_rows = len(trips)
    trips = trips.drop_duplicates()
    steps = [("Exact duplicates", raw_rows - len(trips), len(trips))]
    time = trips["tpep_pickup_datetime"]
    trips = trips.assign(
        duration_min=(trips["tpep_dropoff_datetime"] - time).dt.total_seconds() / 60,
    )
    trips["speed_mph"] = trips["trip_distance"].div(trips["duration_min"] / 60)
    inside = time.between(config["start"], config["end_exclusive"], inclusive="left")
    scoped = trips.loc[inside]
    steps.append(("Pickup outside January or missing", len(trips) - len(scoped), len(scoped)))
    trips = scoped
    rules = [
        ("Duration missing or non-positive", lambda d: d["duration_min"].gt(0)),
        ("Duration over four hours", lambda d: d["duration_min"].le(240)),
        ("Distance missing or non-positive", lambda d: d["trip_distance"].gt(0)),
        ("Base fare missing or non-positive", lambda d: d["fare_amount"].gt(0)),
        ("Speed over 80 mph or unavailable", lambda d: d["speed_mph"].le(80)),
    ]
    for label, keep in rules:
        previous = len(trips)
        trips = trips.loc[keep(trips)]
        steps.append((label, previous - len(trips), len(trips)))
    audit = pd.DataFrame(steps, columns=["rule", "removed", "remaining"])
    return trips, scoped, zones, audit


def add_spatial_features(trips, zones):
    """Map each endpoint independently. Preserve unmatched trips in city counts."""
    geo = trips[["tpep_pickup_datetime", "PULocationID", "DOLocationID"]].rename(
        columns={"PULocationID": "pickup_zone_id", "DOLocationID": "dropoff_zone_id"},
    )
    for endpoint in ("pickup", "dropoff"):
        boroughs = geo[f"{endpoint}_zone_id"].map(zones["Borough"])
        geo[f"{endpoint}_borough"] = boroughs.where(
            boroughs.isin(CITY_BOROUGHS | {"EWR"}), "Unknown"
        ).astype("category")
    return geo


def hourly_features(trips, config):
    index = pd.date_range(config["start"], config["end_exclusive"], freq="h", inclusive="left")
    data = pd.DataFrame(index=index.rename("pickup_hour_start"))
    data["n_trips"] = trips.groupby(trips["tpep_pickup_datetime"].dt.floor("h")).size().reindex(
        index, fill_value=0
    )
    data["split"] = np.select(
        [index < config["validation_start"], index < config["test_start"]],
        ["train", "validation"], default="test",
    )
    data["hour"] = index.hour
    data["day_of_week"] = index.dayofweek
    data["day_of_month"] = index.day
    data["month"] = index.month
    data["is_weekend"] = (index.dayofweek >= 5).astype(int)
    for name, period in (("hour", 24), ("day_of_week", 7), ("month", 12)):
        offset = int(name == "month")
        angle = 2 * np.pi * (data[name] - offset) / period
        data[f"{name}_sin"] = np.sin(angle)
        data[f"{name}_cos"] = np.cos(angle)
    for lag in config["lags_hours"]:
        data[f"lag_{lag}h"] = data["n_trips"].shift(lag)
    last_safe_lag = min(config["lags_hours"])
    window = config["rolling_window_hours"]
    history = data["n_trips"].shift(last_safe_lag).rolling(window, min_periods=window)
    data[f"mean_{window}h_through_lag_{last_safe_lag}h"] = history.mean()
    data["has_complete_history"] = data.filter(regex=r"^(lag_|mean_)").notna().all(axis=1)
    if int(data["n_trips"].sum()) != len(trips) or not data.index.is_unique:
        raise ValueError("Hourly grid did not reconcile with the cleaned trip population.")
    return data


def missingness(scoped, trips, zones):
    """Count missing fields and unresolved endpoint groups on both populations."""
    rows = []
    for column in ("passenger_count", "RatecodeID", "store_and_fwd_flag",
                   "trip_distance", "fare_amount", "duration_min"):
        rows.append((column, int(scoped[column].isna().sum()), int(trips[column].isna().sum())))
    for column in ("PULocationID", "DOLocationID"):
        valid = zones.index[zones["Borough"].isin(CITY_BOROUGHS | {"EWR"})]
        rows.append((f"{column} missing or unknown", int((~scoped[column].isin(valid)).sum()),
                     int((~trips[column].isin(valid)).sum())))
    return pd.DataFrame(rows, columns=["field", "before_cleaning", "after_cleaning"])


def training_spatial_tables(geo, zones, config):
    train = geo.loc[geo["tpep_pickup_datetime"] < config["validation_start"]]
    boroughs = train.groupby(["pickup_borough", "dropoff_borough"], observed=True).size()
    flows = boroughs[boroughs >= config["min_published_od_trips"]].rename("n_trips").reset_index()
    location = train["pickup_zone_id"]
    valid = zones.index[zones["Borough"].isin(CITY_BOROUGHS)]
    known = train.loc[location.isin(valid)]
    counts = pd.crosstab(known["pickup_zone_id"], known["tpep_pickup_datetime"].dt.hour)
    counts = counts.reindex(index=valid, columns=range(24), fill_value=0).sort_index()
    support = counts.sum(axis=1)
    active = counts.loc[support >= config["min_training_zone_trips"]]
    profiles = active.div(active.sum(axis=1), axis=0)
    summary = pd.DataFrame({
        "n_training_pickups": support,
        "cluster_eligible": support.index.isin(active.index),
    }).join(zones[["Borough", "Zone"]])
    summary.index.name = "pickup_zone_id"
    profiles.index.name = summary.index.name
    return profiles, summary, flows, len(train), int(boroughs.sum() - flows["n_trips"].sum())


def temporal_anomalies(hourly, config):
    """Discover unusual demand against training weekday/weekend hourly medians."""
    observed = hourly.loc[hourly["split"] != "test"].copy()
    train = observed.loc[observed["split"] == "train"]
    keys = ["is_weekend", "hour"]
    profile = train.groupby(keys)["n_trips"].median().rename("expected")
    observed = observed.join(profile, on=keys, validate="many_to_one")
    observed["residual"] = observed["n_trips"] - observed["expected"]
    calibration = observed.loc[observed["split"] == "train", "residual"]
    center = calibration.median()
    mad = (calibration - center).abs().median()
    if mad <= 0:
        raise ValueError("Demand anomaly calibration requires a positive training MAD.")
    observed["robust_z"] = (observed["residual"] - center) / (1.4826 * mad)
    candidates = observed.loc[observed["robust_z"].abs() >= config["anomaly_score_threshold"]]
    candidates = candidates[["split", "n_trips", "expected", "residual", "robust_z"]]
    return candidates.sort_values("robust_z", key=lambda score: score.abs(), ascending=False), mad


def raw_anomalies(scoped, config):
    """Aggregate candidate flags from pre-cleaning data, without trip details."""
    dates = scoped["tpep_pickup_datetime"]
    dev = scoped.loc[dates < config["test_start"]]
    distance, duration, speed, fare = (
        dev[c] for c in ("trip_distance", "duration_min", "speed_mph", "fare_amount")
    )
    flags = {
        "Non-positive duration": duration.le(0),
        "Duration over four hours": duration.gt(240),
        "Speed over 80 mph with positive duration": duration.gt(0) & speed.gt(80),
        "Non-positive distance": distance.le(0),
        "Non-positive base fare": fare.le(0),
        "Base fare at least $250": fare.ge(250),
    }
    training = dev["tpep_pickup_datetime"] < config["validation_start"]
    rows = [(name, int((mask & training).sum()), int((mask & ~training).sum()))
            for name, mask in flags.items()]
    return pd.DataFrame(rows, columns=["candidate", "train_count", "validation_count"])
