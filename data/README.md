# Data

Raw data files are **not** committed to this repository (per course rules,
Section 11 — large GitHub files are not allowed). This file documents the
official source, retrieval process, and how to reproduce `data/raw/` locally.

## Official source

NYC Taxi & Limousine Commission (TLC) Trip Record Data — official public
dataset.

- Landing page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- File distribution (CloudFront): https://d37ci6vzurychx.cloudfront.net/trip-data/

## Files used in this project

| File | Description | Download URL |
|---|---|---|
| `yellow_tripdata_2024-01.parquet` | Yellow Taxi trip records, January 2024 (~2.96M rows) | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet` |
| `taxi_zone_lookup.csv` | Static lookup: `LocationID` → Borough/Zone name | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` |

Retrieved: 2026-11-03 (1405-08-12). No access application was required; all
files are public-use.

## How to reproduce locally

```bash
mkdir -p data/raw
cd data/raw
curl -sS -o yellow_tripdata_2024-01.parquet \
  "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
curl -sS -o taxi_zone_lookup.csv \
  "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
```

Then run `python3 src/analysis.py` (or execute
`notebooks/phase1_exploration.ipynb`), which loads the raw parquet file,
derives duration/speed/time fields, audits structure and quality, and
applies the Phase 1 analytic-population filters documented in
`reports/phase1.pdf` (Section 5). The small, derived outputs below *are*
committed to `data/processed/` (no raw/bulk TLC data, only summary tables
and a capped random sample):

- `structural_audit.csv`, `quality_audit.csv`, `filter_counts.csv`
- `univariate_summary.csv`, `borough_counts.csv`, `top_zones.csv`,
  `anomaly_candidates.csv`
- `phase1_analytic_sample.csv` — a fixed-seed (`SEED=42`) random sample of
  50,000 rows from the cleaned analytic population, for quick inspection
  without re-downloading the full raw file.

## Record of resampling / re-cleaning

| Date | Change |
|---|---|
| 2026-11-03 | Initial download of January 2024 Yellow Taxi data and the zone lookup table; Phase 1 structural/quality audit and cleaning filters defined. |

## Cleaning funnel summary (Phase 1)

- Raw rows: 2,964,624.
- Filters applied in order: exact duplicates; pickup timestamp outside
  January 2024; non-positive trip duration; duration over 4 hours;
  non-positive trip distance; non-positive fare amount; implied speed over
  80 mph.
- **Final Phase-1 analytic population: 2,866,703 trips** (96.7% of raw rows).

See `reports/phase1.pdf` (Section 5) for the full issue-by-issue
quantification and the justification for each decision.
