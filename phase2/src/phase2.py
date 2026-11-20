"""Build the Phase 2 tables and figures from the official January files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from patterns import analyze_profiles
from prepare import (
    ROOT, TRIP_FILE, ZONE_FILE, add_spatial_features, hourly_features, load_and_clean,
    missingness, raw_anomalies, temporal_anomalies, training_spatial_tables,
)

COLORS = ["#126468", "#a95527", "#6556a5", "#2b73a9", "#a13564", "#697638", "#53717c", "#a0473a"]


def save_figures(output, selection, variance, zones, centroids, hourly, anomalies):
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.16, "axes.axisbelow": True, "figure.dpi": 150,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 3.25), layout="constrained")
    pcs = np.arange(1, len(variance) + 1)
    left.bar(pcs, 100 * variance, color=COLORS[0])
    left.plot(pcs, 100 * variance.cumsum(), ".-", color=COLORS[1], label="Cumulative")
    left.set(xlabel="Principal component", ylabel="Explained variance (%)", title="PCA variance")
    left.legend(frameon=False)
    best = selection.loc[selection["silhouette"].idxmax()]
    right.plot(selection["k"], selection["silhouette"], "o-", color=COLORS[2])
    right.scatter([best["k"]], [best["silhouette"]], s=100, color=COLORS[0], zorder=3)
    right.set(xlabel="Number of clusters (k)", ylabel="Mean silhouette",
              title=f"Cluster selection  |  k = {int(best['k'])}")
    right.set_xticks(selection["k"])
    fig.savefig(output / "01_pca_selection.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    for cluster, rows in zones.groupby("cluster"):
        ax.scatter(rows["pc1"] * 100, rows["pc2"] * 100, color=COLORS[cluster - 1],
                   s=29, alpha=0.80, label=f"Cluster {cluster} ({len(rows)} zones)")
    ax.set(xlabel=f"PC1 ({variance[0]:.1%} variance)", ylabel=f"PC2 ({variance[1]:.1%} variance)",
           title="Pickup zones in the first two principal components")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output / "02_zone_pca.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 3.7), layout="constrained")
    for cluster, row in centroids.iterrows():
        ax.plot(row.index, 100 * row, marker="o", markersize=2.4,
                color=COLORS[cluster - 1], label=f"Cluster {cluster}")
    ax.set(xlabel="Pickup hour (New York local time)", ylabel="Share of pickups (%)",
           title="Average hourly pickup profile by cluster", xticks=range(0, 24, 2))
    ax.margins(y=0.17)
    ax.legend(frameon=False, ncol=min(4, len(centroids)), loc="upper center")
    fig.savefig(output / "03_cluster_profiles.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 3.4), layout="constrained")
    observed = hourly.loc[hourly["split"] != "test"]
    ax.plot(observed.index, observed["n_trips"], linewidth=1, color="#536e80", label="Hourly pickups")
    ax.axvspan(observed.index[observed["split"] == "validation"][0], observed.index[-1],
               color="#d9e9e5", alpha=0.65, label="Validation")
    ax.scatter(anomalies.index, anomalies["n_trips"], s=17, color=COLORS[1],
               label="Demand candidate", zorder=3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set(xlabel="Pickup date (2024)", ylabel="Trips per hour", title="Unusual hourly demand")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.savefig(output / "04_demand_anomalies.png")
    plt.close(fig)


def calculate(raw_dir, config):
    if not config["start"] < config["validation_start"] < config["test_start"] < config["end_exclusive"]:
        raise ValueError("Split dates must be increasing.")
    if min(config["lags_hours"]) < 6:
        raise ValueError("The feature design requires a lag of at least six hours.")
    trips, scoped, lookup, audit = load_and_clean(raw_dir, config)
    hourly = hourly_features(trips, config)
    quality = missingness(scoped, trips, lookup)
    raw_flags = raw_anomalies(scoped, config)
    del scoped
    geography = add_spatial_features(trips, lookup)
    profiles, support, od, training_trips, suppressed = training_spatial_tables(geography, lookup, config)
    del geography, trips
    selection, labels, centroids, stability, variance, axes = analyze_profiles(profiles, config)
    zones = support.join(labels)
    zones["cluster"] = zones["cluster"].astype("Int64")
    c_summary = labels.join(support).groupby("cluster").agg(
        zones=("n_training_pickups", "size"), training_pickups=("n_training_pickups", "sum"),
    )
    c_summary["peak_hour"] = centroids.idxmax(axis=1)
    c_summary["share_00_05"] = centroids.loc[:, 0:5].sum(axis=1)
    c_summary["share_07_10"] = centroids.loc[:, 7:10].sum(axis=1)
    c_summary["share_16_20"] = centroids.loc[:, 16:20].sum(axis=1)
    borough_mix = pd.crosstab(zones["cluster"], zones["Borough"])
    anomaly, mad = temporal_anomalies(hourly, config)
    pca_variance = pd.DataFrame({
        "component": np.arange(1, len(variance) + 1),
        "explained_variance_ratio": variance, "cumulative_variance_ratio": variance.cumsum(),
    })
    pca_loadings = pd.DataFrame(axes[:2].T, columns=["pc1", "pc2"]).rename_axis("hour")
    split_summary = hourly.groupby("split", sort=False).agg(
        hours=("n_trips", "size"), trips=("n_trips", "sum"),
        history_ready_hours=("has_complete_history", "sum"),
    )
    if training_trips != int(split_summary.loc["train", "trips"]):
        raise ValueError("Training spatial counts do not match citywide training demand.")
    assert np.allclose(profiles.sum(axis=1), 1)
    assert (zones["cluster"].notna() == zones["cluster_eligible"]).all()
    assert int(c_summary["training_pickups"].sum()) == int(zones.loc[labels.index, "n_training_pickups"].sum())
    assert not anomaly["split"].eq("test").any()

    tables = {
        "cleaning_funnel": audit, "missingness": quality, "hourly_features": hourly,
        "split_summary": split_summary, "training_borough_od": od, "zone_profiles": profiles,
        "zone_clusters": zones, "cluster_selection": selection, "cluster_centroids": centroids,
        "cluster_summary": c_summary, "cluster_borough_mix": borough_mix,
        "cluster_stability": stability, "pca_variance": pca_variance, "pca_loadings": pca_loadings,
        "demand_anomalies": anomaly, "raw_anomaly_flags": raw_flags,
    }
    summary = {
        "raw_rows": int(audit.iloc[0][["removed", "remaining"]].sum()),
        "clean_rows": int(audit.iloc[-1]["remaining"]),
        "training_trips": training_trips, "official_city_zones": len(zones),
        "clustered_zones": len(labels), "clustered_zone_pickups": int(c_summary["training_pickups"].sum()),
        "suppressed_od_trips": suppressed, "demand_residual_mad": float(mad),
        "demand_anomaly_count": len(anomaly),
        "source_sha256": {
            name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
            for name in (TRIP_FILE, ZONE_FILE)
        },
    }
    return tables, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tables, summary = calculate(args.raw_dir, config)
    data_dir = args.output_dir / "data" / "processed"
    figures_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(data_dir / f"{name}.csv", index=table.index.name is not None,
                     float_format="%.10g")
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_figures(figures_dir, tables["cluster_selection"],
                 tables["pca_variance"]["explained_variance_ratio"].to_numpy(),
                 tables["zone_clusters"].dropna(subset=["cluster"]),
                 tables["cluster_centroids"], tables["hourly_features"], tables["demand_anomalies"])
    print(f"{summary['clean_rows']:,} clean trips; {len(tables['hourly_features'])} hours; "
          f"{summary['clustered_zones']} zones clustered.")


if __name__ == "__main__":
    main()
