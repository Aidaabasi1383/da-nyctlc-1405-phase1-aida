"""
Phase 1 analysis helpers for the NYC TLC Yellow Taxi Trip Records project
(Data Analysis course, Fall 1405, Path A).

Main analytical question (working version, Phase 1):
Can hourly NYC Yellow Taxi pickup demand be forecast from time-of-day,
day-of-week, and recent history, to support supply/dispatch planning?

All functions are pure / side-effect-light (aside from the explicit
fig_*() plotting helpers, which write PNGs to figures/) so they can be
called identically from the exploration notebook and from the report
builder, keeping the two deliverables in sync with a single source of
truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

RAW_FILE = RAW_DIR / "yellow_tripdata_2024-01.parquet"
ZONE_FILE = RAW_DIR / "taxi_zone_lookup.csv"

SEED = 42
MONTH_START = "2024-01-01"
MONTH_END = "2024-02-01"  # exclusive

UNIVARIATE_COLS = [
    "trip_distance", "fare_amount", "total_amount",
    "trip_duration_min", "speed_mph", "passenger_count",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw Jan-2024 Yellow Taxi parquet file and the TLC zone
    lookup table, exactly as downloaded (see data/README.md)."""
    df = pd.read_parquet(RAW_FILE)
    zones = pd.read_csv(ZONE_FILE)
    return df, zones


# --------------------------------------------------------------------------
# Structural audit (Section 5, "ممیزی ساختار داده")
# --------------------------------------------------------------------------

def structural_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(3),
        "n_unique": df.nunique(),
    })
    audit.index.name = "column"
    return audit.reset_index()


# --------------------------------------------------------------------------
# Derive fields needed throughout the pipeline
# --------------------------------------------------------------------------

def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["speed_mph"] = np.where(
        df["trip_duration_min"] > 0,
        df["trip_distance"] / (df["trip_duration_min"] / 60.0),
        np.nan,
    )
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Mon
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["is_weekend"] = df["pickup_dow"].isin([5, 6])
    df["fare_per_mile"] = np.where(
        df["trip_distance"] > 0, df["fare_amount"] / df["trip_distance"], np.nan
    )
    return df


# --------------------------------------------------------------------------
# Quality audit (Section 5, "ممیزی کیفیت") — quantify issues *before*
# deciding what to drop, with an explicit decision per issue.
# --------------------------------------------------------------------------

def quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []

    def add(issue, count, decision):
        rows.append({
            "issue": issue,
            "n_rows": int(count),
            "pct_of_raw": round(100 * count / n, 3),
            "initial_decision": decision,
        })

    add(
        "Exact duplicate rows",
        df.duplicated().sum(),
        "Drop during cleaning (true duplicate, not a legitimate repeat trip)",
    )
    add(
        "Missing value in passenger_count",
        df["passenger_count"].isna().sum(),
        "Keep row; drop only in analyses specifically centered on passenger_count",
    )
    add(
        "Missing value in RatecodeID",
        df["RatecodeID"].isna().sum(),
        "Keep row; RatecodeID is not used centrally in this phase",
    )
    add(
        "Missing value in congestion_surcharge / Airport_fee",
        df["congestion_surcharge"].isna().sum(),
        "Keep row; these columns are not used in the main question (demand forecasting)",
    )
    add(
        "Pickup timestamp outside January 2024 (data-entry error)",
        (~df["tpep_pickup_datetime"].between(MONTH_START, MONTH_END, inclusive="left")).sum(),
        "Drop; outside the project's defined time scope",
    )
    dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    add(
        "Implausible trip duration: zero or negative",
        (dur <= 0).sum(),
        "Drop; timestamp error, physically impossible",
    )
    add(
        "Implausible trip duration: over 4 hours",
        (dur > 240).sum(),
        "Drop; most likely a meter left running (abandoned trip), not a real ride",
    )
    add(
        "Implausible trip distance: zero or negative",
        (df["trip_distance"] <= 0).sum(),
        "Drop; without distance, a real trip cannot be defined",
    )
    add(
        "Implausible base fare: zero or negative (fare_amount <= 0)",
        (df["fare_amount"] <= 0).sum(),
        "Drop; likely a voided transaction or logging error",
    )
    speed = np.where(dur > 0, df["trip_distance"] / (dur / 60.0), np.nan)
    add(
        "Implausible implied speed: over 80 mph",
        (speed > 80).sum(),
        "Drop from the main analytic population; kept separately as an anomaly candidate (Section 12 of the report)",
    )
    add(
        "Zero passenger count (passenger_count == 0)",
        (df["passenger_count"] == 0).sum(),
        "Keep in the main analytic population (common and documented in TLC data); flag only",
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cleaning: apply the "initial_decision" column above to produce the
# Phase-1 analytic population, with an explicit, ordered exclusion count.
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = derive_fields(df)
    counts = {"raw_rows": len(df)}

    before = len(df)
    df = df.drop_duplicates()
    counts["excl_duplicate_rows"] = before - len(df)

    before = len(df)
    df = df[df["tpep_pickup_datetime"].between(MONTH_START, MONTH_END, inclusive="left")]
    counts["excl_outside_jan2024"] = before - len(df)

    before = len(df)
    df = df[df["trip_duration_min"] > 0]
    counts["excl_nonpositive_duration"] = before - len(df)

    before = len(df)
    df = df[df["trip_duration_min"] <= 240]
    counts["excl_duration_over_4h"] = before - len(df)

    before = len(df)
    df = df[df["trip_distance"] > 0]
    counts["excl_nonpositive_distance"] = before - len(df)

    before = len(df)
    df = df[df["fare_amount"] > 0]
    counts["excl_nonpositive_fare"] = before - len(df)

    before = len(df)
    df = df[df["speed_mph"] <= 80]
    counts["excl_speed_over_80mph"] = before - len(df)

    counts["analytic_rows"] = len(df)
    return df, counts


# --------------------------------------------------------------------------
# Univariate summary (Section 5, "تحلیل اکتشافی تک‌متغیره")
# --------------------------------------------------------------------------

def univariate_table(df: pd.DataFrame, cols: list[str] = UNIVARIATE_COLS) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "variable": c,
            "mean": round(s.mean(), 3),
            "median": round(s.median(), 3),
            "std": round(s.std(), 3),
            "skew": round(s.skew(), 3),
            "min": round(s.min(), 3),
            "p95": round(s.quantile(0.95), 3),
            "max": round(s.max(), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Spatial snapshot (Path A requirement, "ساخت نمای مکانی اولیه")
# --------------------------------------------------------------------------

def spatial_summary(df: pd.DataFrame, zones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zones.rename(columns={"LocationID": "PULocationID", "Borough": "PU_Borough", "Zone": "PU_Zone"})
    merged = df.merge(z[["PULocationID", "PU_Borough", "PU_Zone"]], on="PULocationID", how="left")
    merged["PU_Borough"] = merged["PU_Borough"].fillna("Unmatched (PULocationID not in zone lookup)")

    borough_counts = (
        merged["PU_Borough"].value_counts(dropna=False)
        .rename_axis("borough").reset_index(name="n_trips")
    )
    borough_counts["pct"] = round(100 * borough_counts["n_trips"] / len(merged), 2)

    zone_counts = (
        merged["PU_Zone"].value_counts(dropna=False)
        .head(10).rename_axis("zone").reset_index(name="n_trips")
    )
    zone_counts["pct"] = round(100 * zone_counts["n_trips"] / len(merged), 2)

    return borough_counts, zone_counts


# --------------------------------------------------------------------------
# Anomaly candidates (Path A requirement, "نامزدهای ناهنجاری")
# --------------------------------------------------------------------------

def anomaly_candidates(df_derived: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious records on the *pre-filter* derived frame, so we can
    report how many plausible-but-weird trips exist, separately from the
    hard data-entry errors already excluded in clean()."""
    d = df_derived
    flags = pd.DataFrame({
        "implausible_speed_gt_80mph": (d["speed_mph"] > 80),
        "idle_meter_long_duration_zero_fare_move": (d["trip_duration_min"] > 20) & (d["trip_distance"] < 0.2),
        "very_high_fare_per_mile": (d["fare_per_mile"] > d["fare_per_mile"].quantile(0.999)),
        "zero_passenger_count": (d["passenger_count"] == 0),
    })
    summary = flags.sum().rename("n_flagged").to_frame()
    summary["pct_of_raw"] = round(100 * summary["n_flagged"] / len(d), 4)
    return summary.reset_index(names="anomaly_type")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_univariate(df: pd.DataFrame, col: str, name: str, xlim_p99: bool = True):
    s = df[col].dropna()
    if xlim_p99:
        s = s[s <= s.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=50, color="#2c7fb8", edgecolor="white")
    ax.axvline(s.mean(), color="#d7191c", linestyle="--", label=f"mean={s.mean():.2f}")
    ax.axvline(s.median(), color="#1a9641", linestyle="--", label=f"median={s.median():.2f}")
    ax.set_xlabel(col)
    ax.set_ylabel("n trips")
    ax.set_title(f"Distribution of {col} (p99-trimmed)" if xlim_p99 else f"Distribution of {col}")
    ax.legend()
    return _save(fig, name)


def fig_scatter_distance_fare(df: pd.DataFrame, name: str):
    sample = df.sample(min(20000, len(df)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sample["trip_distance"], sample["fare_amount"], s=4, alpha=0.25, color="#2c7fb8")
    r = df["trip_distance"].corr(df["fare_amount"])
    ax.set_xlim(0, df["trip_distance"].quantile(0.99))
    ax.set_ylim(0, df["fare_amount"].quantile(0.99))
    ax.set_xlabel("trip_distance (mi)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title(f"Distance vs. fare (n={len(sample):,} sampled), r={r:.3f}")
    return _save(fig, name)


def fig_box_duration_by_hour(df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [df.loc[df["pickup_hour"] == h, "trip_duration_min"].clip(upper=60) for h in range(24)]
    ax.boxplot(data, positions=range(24), showfliers=False)
    ax.set_xlabel("pickup hour of day")
    ax.set_ylabel("trip_duration_min (clipped at 60)")
    ax.set_title("Trip duration by hour of day")
    return _save(fig, name)


def fig_hourly_demand(df: pd.DataFrame, name: str):
    weekday = df[~df["is_weekend"]].groupby("pickup_hour").size() / df.loc[~df["is_weekend"], "pickup_date"].nunique()
    weekend = df[df["is_weekend"]].groupby("pickup_hour").size() / df.loc[df["is_weekend"], "pickup_date"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(weekday.index, weekday.values, marker="o", label="weekday (avg/day)")
    ax.plot(weekend.index, weekend.values, marker="o", label="weekend (avg/day)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("avg trips per hour")
    ax.set_title("Average hourly pickup demand: weekday vs weekend")
    ax.legend()
    return _save(fig, name)


def fig_daily_timeseries(df: pd.DataFrame, name: str):
    daily = df.groupby("pickup_date").size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(daily.index), daily.values, marker="o", markersize=3, color="#2c7fb8")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("date")
    ax.set_ylabel("n trips / day")
    ax.set_title("Daily pickup volume, January 2024")
    fig.autofmt_xdate()
    return _save(fig, name)


def fig_borough_share(borough_counts: pd.DataFrame, name: str):
    top = borough_counts.head(7)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top["borough"].astype(str), top["pct"], color="#2c7fb8")
    ax.invert_yaxis()
    ax.set_xlabel("% of trips (pickup borough)")
    ax.set_title("Pickup share by borough")
    return _save(fig, name)


def fig_speed_by_daytype(df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    data = [
        df.loc[~df["is_weekend"], "speed_mph"].clip(upper=40),
        df.loc[df["is_weekend"], "speed_mph"].clip(upper=40),
    ]
    ax.boxplot(data, tick_labels=["weekday", "weekend"], showfliers=False)
    ax.set_ylabel("speed_mph (clipped at 40)")
    ax.set_title("Implied speed: weekday vs weekend")
    return _save(fig, name)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline() -> dict:
    """Runs the full Phase-1 pipeline once and writes processed CSVs and all
    figures. Returns a dict of summary objects used by both the notebook
    and the PDF report builder."""
    raw, zones = load_raw()
    struct = structural_audit(raw)
    derived = derive_fields(raw)
    qa = quality_audit(derived)
    analytic, filter_counts = clean(raw)
    uni = univariate_table(analytic)
    borough_counts, zone_counts = spatial_summary(analytic, zones)
    anomalies = anomaly_candidates(derived)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    struct.to_csv(PROCESSED_DIR / "structural_audit.csv", index=False)
    qa.to_csv(PROCESSED_DIR / "quality_audit.csv", index=False)
    uni.to_csv(PROCESSED_DIR / "univariate_summary.csv", index=False)
    borough_counts.to_csv(PROCESSED_DIR / "borough_counts.csv", index=False)
    zone_counts.to_csv(PROCESSED_DIR / "top_zones.csv", index=False)
    anomalies.to_csv(PROCESSED_DIR / "anomaly_candidates.csv", index=False)
    pd.DataFrame([filter_counts]).to_csv(PROCESSED_DIR / "filter_counts.csv", index=False)
    analytic.sample(min(50000, len(analytic)), random_state=SEED).to_csv(
        PROCESSED_DIR / "phase1_analytic_sample.csv", index=False
    )

    fig_univariate(analytic, "trip_distance", "fig1_hist_trip_distance.png")
    fig_univariate(analytic, "fare_amount", "fig2_hist_fare_amount.png")
    fig_univariate(analytic, "trip_duration_min", "fig3_hist_trip_duration.png")
    fig_univariate(analytic, "speed_mph", "fig4_hist_speed.png", xlim_p99=False)
    fig_scatter_distance_fare(analytic, "fig5_scatter_distance_fare.png")
    fig_box_duration_by_hour(analytic, "fig6_box_duration_by_hour.png")
    fig_hourly_demand(analytic, "fig7_hourly_demand_weekday_weekend.png")
    fig_daily_timeseries(analytic, "fig8_daily_timeseries.png")
    fig_borough_share(borough_counts, "fig9_borough_share.png")
    fig_speed_by_daytype(analytic, "fig10_speed_by_daytype.png")

    return {
        "raw_shape": raw.shape,
        "analytic_shape": analytic.shape,
        "struct": struct,
        "qa": qa,
        "filter_counts": filter_counts,
        "uni": uni,
        "borough_counts": borough_counts,
        "zone_counts": zone_counts,
        "anomalies": anomalies,
    }


if __name__ == "__main__":
    out = run_pipeline()
    print("raw:", out["raw_shape"], "-> analytic:", out["analytic_shape"])
    print(out["filter_counts"])
