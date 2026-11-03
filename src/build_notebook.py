"""Builds notebooks/phase1_exploration.ipynb from scratch using nbformat.
The notebook calls the same analysis.py functions used by build_report.py,
so after `jupyter nbconvert --execute` it reproduces the exact same numbers
and figures as reports/phase1.pdf.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "phase1_exploration.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Phase 1 Exploration — NYC TLC Yellow Taxi Trip Records (January 2024)

**Data Analysis course, Fall 1405 — Path A**
**Student:** Aida Abbasi — ID 401170897

This notebook runs the Phase 1 pipeline end to end: load the raw TLC parquet
file, audit its structure and quality, clean it into the Phase-1 analytic
population, and explore it (univariate, multivariate, time, space,
anomalies). All functions live in `src/analysis.py`; `src/build_report.py`
calls the same functions to build `reports/phase1.pdf`, so this notebook and
the PDF report are guaranteed to show the same numbers.""")

code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import analysis as A
import pandas as pd
pd.set_option("display.width", 140)
import matplotlib.pyplot as plt
%matplotlib inline""")

md("## 1. Load raw data")
code("""\
raw, zones = A.load_raw()
print("Raw shape:", raw.shape)
raw.head()""")

md("## 2. Structural audit")
code("""\
struct = A.structural_audit(raw)
struct""")

md("""\
## 3. Quality audit

Each issue is quantified against the raw row count, with an explicit
initial decision (kept / dropped / flagged).""")
code("""\
qa = A.quality_audit(A.derive_fields(raw))
qa""")

md("## 4. Clean into the Phase-1 analytic population")
code("""\
analytic, filter_counts = A.clean(raw)
filter_counts""")

md("## 5. Univariate exploratory analysis")
code("""\
uni = A.univariate_table(analytic)
uni""")

code("""\
for col in ["trip_distance", "fare_amount", "trip_duration_min"]:
    s = analytic[col]
    s = s[s <= s.quantile(0.99)]
    plt.figure(figsize=(6, 3.5))
    plt.hist(s, bins=50, color="#2c7fb8", edgecolor="white")
    plt.axvline(s.mean(), color="#d7191c", linestyle="--", label=f"mean={s.mean():.2f}")
    plt.axvline(s.median(), color="#1a9641", linestyle="--", label=f"median={s.median():.2f}")
    plt.title(f"Distribution of {col} (p99-trimmed)")
    plt.legend()
    plt.show()""")

md("## 6. Multivariate exploratory analysis")
code("""\
sample = analytic.sample(min(20000, len(analytic)), random_state=A.SEED)
r = analytic["trip_distance"].corr(analytic["fare_amount"])
plt.figure(figsize=(6, 4.5))
plt.scatter(sample["trip_distance"], sample["fare_amount"], s=4, alpha=0.25, color="#2c7fb8")
plt.xlim(0, analytic["trip_distance"].quantile(0.99))
plt.ylim(0, analytic["fare_amount"].quantile(0.99))
plt.xlabel("trip_distance (mi)"); plt.ylabel("fare_amount ($)")
plt.title(f"Distance vs. fare, r={r:.3f}")
plt.show()""")

code("""\
data = [analytic.loc[analytic["pickup_hour"] == h, "trip_duration_min"].clip(upper=60) for h in range(24)]
plt.figure(figsize=(9, 4.5))
plt.boxplot(data, positions=range(24), showfliers=False)
plt.xlabel("pickup hour of day"); plt.ylabel("trip_duration_min (clipped at 60)")
plt.title("Trip duration by hour of day")
plt.show()""")

md("## 7. Path A — Trip-time audit (hourly / weekday-weekend demand)")
code("""\
weekday = analytic[~analytic["is_weekend"]].groupby("pickup_hour").size() / analytic.loc[~analytic["is_weekend"], "pickup_date"].nunique()
weekend = analytic[analytic["is_weekend"]].groupby("pickup_hour").size() / analytic.loc[analytic["is_weekend"], "pickup_date"].nunique()
plt.figure(figsize=(8, 4.5))
plt.plot(weekday.index, weekday.values, marker="o", label="weekday (avg/day)")
plt.plot(weekend.index, weekend.values, marker="o", label="weekend (avg/day)")
plt.xlabel("hour of day"); plt.ylabel("avg trips per hour")
plt.title("Average hourly pickup demand: weekday vs weekend")
plt.legend(); plt.show()""")

code("""\
daily = analytic.groupby("pickup_date").size()
plt.figure(figsize=(9, 4))
plt.plot(pd.to_datetime(daily.index), daily.values, marker="o", markersize=3, color="#2c7fb8")
plt.xlabel("date"); plt.ylabel("n trips / day")
plt.title("Daily pickup volume, January 2024")
plt.xticks(rotation=40)
plt.show()
daily.sort_values().head(3)""")

md("""\
The sharpest dip (Jan 8, 2024) coincides with a winter storm that affected
the NYC area in early January 2024 — noted as a plausible explanation, not a
confirmed causal claim (see leakage/ethics discussion in the PDF report).""")

md("## 8. Path A — Variable consistency (speed, fare-per-mile)")
code("""\
plt.figure(figsize=(6, 4))
plt.hist(analytic["speed_mph"], bins=50, color="#2c7fb8", edgecolor="white")
plt.xlabel("speed_mph"); plt.ylabel("n trips")
plt.title("Implied speed distribution (post-cleaning, capped at 80 mph)")
plt.show()
analytic["fare_per_mile"].describe()""")

md("## 9. Path A — Initial spatial snapshot")
code("""\
borough_counts, zone_counts = A.spatial_summary(analytic, zones)
borough_counts""")

code("""\
zone_counts""")

md("## 10. Path A — Anomaly candidates")
code("""\
anomalies = A.anomaly_candidates(A.derive_fields(raw))
anomalies""")

md("""\
## 11. Summary

See `reports/phase1.pdf` for the full written report: problem framing, the
three candidate questions and the selected main question, data provenance,
leakage risks, baseline definition, and the preliminary train/validation/test
split plan. This notebook is the reproducible evidence behind every number
and figure in that report.""")

nb["cells"] = cells
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT_PATH)
print(f"Wrote {OUT_PATH}")
