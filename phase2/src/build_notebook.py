"""Create an inspection notebook from the Phase 2 output schema."""
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook = nbf.v4.new_notebook(
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    cells=[
        md("""# NYC Yellow Taxi | Phase 2
January 2024 · Data Analysis, Fall 1405 · Aida Abbasi

This notebook inspects the published aggregates. Run `python src/phase2.py` from the phase directory with the official raw inputs to rebuild them. No forecast model is fitted here."""),
        code("""import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
out = root / 'data' / 'processed'

def table(name):
    return pd.read_csv(out / f'{name}.csv')

def show_figure(name):
    display(Image(filename=str(root / 'figures' / name), width=720))

summary = json.loads((out / 'summary.json').read_text())
display(table('cleaning_funnel'))
display(table('missingness'))
display(table('split_summary'))"""),
        md("""## Hourly demand and feature availability
The target is the number of pickups passing the Phase 1 cleaning rules. The January grid is contiguous. Lags use earlier observed hours, with a six-hour minimum. Missing start-of-month history is left blank."""),
        code("""hourly = table('hourly_features')
assert len(hourly) == 744
assert hourly['n_trips'].sum() == summary['clean_rows']
for lag in (6, 24, 168):
    pd.testing.assert_series_equal(
        hourly[f'lag_{lag}h'], hourly['n_trips'].shift(lag), check_names=False
    )
display(hourly.loc[[0, 6, 24, 29, 168, 503, 504, 672],
    ['pickup_hour_start', 'split', 'n_trips', 'lag_6h', 'lag_24h', 'lag_168h',
     'mean_24h_through_lag_6h', 'has_complete_history']])
display(table('training_borough_od').nlargest(6, 'n_trips'))"""),
        md("""## PCA and zone patterns
The clustering input is a 24-hour pickup share profile per eligible zone. PCA visualizes the training profiles. K-means uses the full profiles. The highest silhouette selects k."""),
        code("""display(table('pca_variance').head(5).round(4))
display(table('cluster_selection').round(4))
show_figure('01_pca_selection.png')
show_figure('02_zone_pca.png')"""),
        md("""## Interpretation and stability
The first cluster has a strong late-night signature. The second has more daytime and evening activity. Adjusted Rand index is label-order invariant. The 80% refit assesses the effect of changing the zone sample."""),
        code("""display(table('cluster_summary').round(3))
display(table('cluster_borough_mix'))
stability = table('cluster_stability')
display(stability.groupby('check')['adjusted_rand'].agg(['mean', 'median', 'min', 'max']).round(3))
show_figure('03_cluster_profiles.png')"""),
        md("""## Candidate anomalies
Demand candidates use training weekday/weekend-by-hour medians and a training MAD scale. Raw flags may overlap. Event attribution needs independent evidence. The test period is excluded from both screens."""),
        code("""display(table('demand_anomalies').head(8).round(2))
display(table('raw_anomaly_flags'))
show_figure('04_demand_anomalies.png')"""),
    ],
)
nbf.write(notebook, root / "notebooks" / "phase2_preparation.ipynb")
