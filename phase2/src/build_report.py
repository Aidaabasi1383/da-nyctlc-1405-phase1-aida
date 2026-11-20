"""Build the Phase 2 report from the published figures and result tables."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
WIDTH = 491
INK = colors.HexColor("#172b36")
MUTED = colors.HexColor("#526875")
TEAL = colors.HexColor("#126468")
PALE = colors.HexColor("#eef5f4")
RULE = colors.HexColor("#d8e4e7")
WHITE = colors.white

STYLES = {
    "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=30, leading=34,
                            textColor=INK, spaceAfter=15),
    "kicker": ParagraphStyle("Kicker", fontName="Helvetica-Bold", fontSize=9, leading=13,
                             textColor=TEAL, spaceAfter=10),
    "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=22, leading=26,
                         textColor=INK, spaceAfter=12),
    "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13, leading=16,
                         textColor=TEAL, spaceBefore=12, spaceAfter=6),
    "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10.6, leading=15.2,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=9),
    "caption": ParagraphStyle("Caption", fontName="Helvetica", fontSize=9, leading=12,
                              textColor=MUTED, spaceBefore=5, spaceAfter=9),
    "table": ParagraphStyle("Table", fontName="Helvetica", fontSize=9.2, leading=12,
                            textColor=INK),
    "table_head": ParagraphStyle("TableHead", fontName="Helvetica-Bold", fontSize=9.2, leading=12,
                                 textColor=WHITE),
}


def read(name):
    return pd.read_csv(DATA / f"{name}.csv")


def p(value, style="body"):
    return Paragraph(str(value), STYLES[style])


def tab(headers, rows, widths):
    cells = [[p(value, "table_head") for value in headers]]
    cells += [[p(value, "table") for value in row] for row in rows]
    table = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, WHITE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
    ]))
    return table


def graphic(name, caption, width=WIDTH):
    image = Image(str(FIGURES / name))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    image.hAlign = "CENTER"
    return [image, p(caption, "caption")]


def header_footer(canvas, doc):
    page_width, page_height = A4
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(52, page_height - 44, page_width - 52, page_height - 44)
    canvas.line(52, 43, page_width - 52, 43)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(52, page_height - 35, "DATA ANALYSIS  /  FALL 1405")
    canvas.drawRightString(page_width - 52, page_height - 35, "NYC TLC  /  JANUARY 2024")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(52, 29, "Aida Abbasi  |  Phase 2")
    canvas.drawRightString(page_width - 52, 29, str(doc.page))
    canvas.restoreState()


def build():
    data = json.loads((DATA / "summary.json").read_text())
    config = json.loads((ROOT / "configs" / "phase2.json").read_text())
    splits = read("split_summary")
    funnel = read("cleaning_funnel")
    missing = read("missingness")
    selected = read("cluster_selection")
    best = selected.loc[selected["silhouette"].idxmax()]
    clusters = read("cluster_summary")
    first, second = clusters.iloc[0], clusters.iloc[1]
    pca = read("pca_variance")
    captured = pca.loc[1, "cumulative_variance_ratio"]
    stability = read("cluster_stability").groupby("check")["adjusted_rand"].agg(["mean", "median", "min"])
    anomalies = read("demand_anomalies")
    worst = anomalies.iloc[0]
    flags = read("raw_anomaly_flags")
    od = read("training_borough_od")
    dominant = od.sort_values("n_trips", ascending=False).iloc[0]
    all_trips = data["clean_rows"]
    coverage = data["clustered_zone_pickups"] / data["training_trips"]
    document = SimpleDocTemplate(
        str(ROOT / "reports" / "phase2.pdf"), pagesize=A4,
        leftMargin=46, rightMargin=46, topMargin=62, bottomMargin=57,
        title="Phase 2 | Data preparation and pickup patterns", author="Aida Abbasi",
        subject="NYC TLC Yellow Taxi | January 2024 | Data Analysis, Fall 1405",
    )
    pages = [
        [
            Spacer(1, 30),
            p("PHASE 2  /  PATH A", "kicker"),
            p("Data preparation<br/>and pickup patterns", "title"),
            p("NYC TLC Yellow Taxi trip records · January 2024"),
            p("Aida Abbasi  |  Student ID 401170897"),
            Spacer(1, 14),
            tab(["Cleaned trips", "Hourly observations", "Clustered pickup zones"],
                [[f"{all_trips:,}", f"{splits['hours'].sum():,}", data["clustered_zones"]]], [165, 163, 163]),
            Spacer(1, 18),
            p("Question and scope", "h2"),
            p("Can the number of yellow taxi pickups in the next hour be estimated from the calendar and recent counts? "
              "An hourly input table supports the later forecasting experiment. Training-zone profiles show how the timing "
              "of travel varies across pickup areas, which matters when translating a citywide estimate into fleet decisions."),
            p(f"The first two principal components explain {captured:.1%} of the variation across eligible zone profiles. "
              f"The highest silhouette is {best['silhouette']:.3f} at k = {int(best['k'])}. "
              "The two profiles differ most clearly in overnight activity. "
              "The stability check shows that cluster membership can change when fewer areas are available."),
            p("Time windows", "h2"),
            tab(["Partition", "January 2024", "Hours", "Pickups", "With complete history"],
                [[r.split.title(), days, f"{r.hours:,}", f"{r.trips:,}", f"{r.history_ready_hours:,}"]
                 for r, days in zip(splits.itertuples(), ["1 to 21", "22 to 28", "29 to 31"])],
                [98, 100, 60, 97, 136]),
            p("Table 1. Split boundaries occur at midnight, New York local time. Complete history requires all three lags and the rolling mean.", "caption"),
            p("PCA, cluster selection, stability and anomaly calibration use January 1 to 21. "
              "Validation hours are scored against the frozen anomaly reference. The 72 test hours are reserved for the forecasting evaluation."),
        ],
        [
            p("Data preparation", "h1"),
            p("The two inputs are the complete January 2024 Yellow Taxi Parquet and the TLC zone lookup. "
              "The preprocessing script enforces the same ordered filters as Phase 1. "
              f"It retains {all_trips / data['raw_rows']:.1%} of {data['raw_rows']:,} raw records."),
            tab(["Rule, applied in order", "Removed", "Remaining"],
                [[r.rule, f"{r.removed:,}", f"{r.remaining:,}"] for r in funnel.itertuples()], [295, 92, 104]),
            p("Table 2. Removal counts are incremental. Duration and speed are calculated from original timestamps and distance.", "caption"),
            p("Missing values", "h2"),
            tab(["Field", "In month, before cleaning", "After cleaning"],
                [[r.field, f"{r.before_cleaning:,}", f"{r.after_cleaning:,}"]
                 for r in missing.itertuples() if r.before_cleaning or r.after_cleaning], [235, 135, 121]),
            p("Table 3. The first column of counts uses the in-month deduplicated population. Endpoint counts include unresolved IDs.", "caption"),
            p("Passenger count, rate code and the store-and-forward flag each have the same number of missing values. "
              "This pattern is consistent with gaps in reporting. The file alone cannot establish the mechanism. "
              "These fields are omitted from the forecasting table and the trips remain eligible."),
            p("An unknown endpoint is kept as its own category for spatial summaries. A missing pickup time prevents hour assignment. "
              "A missing duration, distance or base fare fails the relevant cleaning rule. "
              "Warm-up hours retain blank lag values and carry an explicit availability flag."),
        ],
        [
            p("Time and location features", "h1"),
            tab(["Feature", "Definition and purpose"],
                [
                    ["Hour and weekday", "Local hour 0 to 23 and weekday 0 to 6, plus a weekend indicator and sine/cosine pairs for a 24-hour and 7-day cycle."],
                    ["Month and day of month", "Calendar audit columns, including a 12-month sine/cosine pair. Month is constant in January and has no predictive variation in this study."],
                    ["6h, 24h and 168h lags", "Counts for the earlier pickup intervals. They represent recent demand, the same hour yesterday and the same hour last week."],
                    ["24-hour trailing mean", "Average of the 24 counts from t-29h to t-6h inclusive. Blanks remain when any required history precedes January."],
                    ["Pickup / dropoff borough", "Maps each TLC endpoint independently. Borough-level OD totals preserve broad regional flows."],
                    ["Pickup-zone profile", "One share per hour, each equal to the zone's training pickups in that hour divided by its total training pickups."],
                ], [161, 330]),
            p("Table 4. The feature inventory gives the schema and availability of every column.", "caption"),
            p("Calendar encoding", "h2"),
            p("A single integer places hours 23 and 0 at a distance of 23. A sine/cosine pair makes their distance 0.261, "
              "the same as for hours 0 and 1. Likewise, Sunday to Monday changes from 6 to 0.868 on the weekly circle. "
              "The paired coordinates encode cycle proximity before any predictive comparison."),
            p("Availability at forecast time", "h2"),
            p("A retained trip has a duration of at most four hours. The most recent allowed bucket ends five hours before the target "
              "hour, leaving at least one hour after the latest possible dropoff. TLC provides neither ingestion timestamps nor a "
              "live feed in this file. Operational use would require coverage and latency checks on a separate source. Per-trip fares, "
              "dropoff times and destination fields belong to retrospective analysis."),
            p("Spatial aggregation", "h2"),
            p(f"The largest training flow is {dominant.pickup_borough} to {dominant.dropoff_borough} ({dominant.n_trips:,} trips). "
              f"Pairs below {config['min_published_od_trips']} training trips are suppressed in the OD table "
              f"({data['suppressed_od_trips']:,} trips omitted there). Manhattan's share limits how much a citywide count says about individual zones."),
        ],
        [
            p("PCA and cluster selection", "h1"),
            p(f"Of {data['official_city_zones']} official zones in the five boroughs, {data['clustered_zones']} met the "
              f"{config['min_training_zone_trips']:,}-training-pickup minimum. They account for {coverage:.1%} of training pickups. "
              "That threshold limits noisy profiles from sparse locations. Each eligible zone gets one equally weighted vector of 24 hourly pickup shares."),
            p(f"PCA uses the centered shares and singular value decomposition. PC1 explains "
              f"{pca.loc[0, 'explained_variance_ratio']:.1%}, PC2 {pca.loc[1, 'explained_variance_ratio']:.1%}. "
              "The first two components are used for visualization. K-means is fitted on all 24 shares with Euclidean distance, "
              f"k-means++ initialization, {config['kmeans_restarts']} starts per fit and base seed {config['seed']}. "
              "Choosing the maximum silhouette keeps the criterion independent of the plotted projection."),
            *graphic("01_pca_selection.png", "Figure 1. Left: variance explained by each component and cumulatively. Right: mean silhouette over k = 2 through 8."),
            *graphic("02_zone_pca.png", "Figure 2. Each point represents one training-zone profile. Axis values are PCA scores scaled by 100 for display.", width=426),
        ],
        [
            p("Cluster profiles and stability", "h1"),
            *graphic("03_cluster_profiles.png", "Figure 3. Centroids are mean shares by local pickup hour. Every zone within a cluster contributes equally."),
            tab(["Pattern", "Zones", "Peak hour", "00:00 to 05:59", "16:00 to 20:59"],
                [
                    ["1 · Overnight activity", int(first.zones), f"{int(first.peak_hour):02d}:00", f"{first.share_00_05:.1%}", f"{first.share_16_20:.1%}"],
                    ["2 · Day and evening activity", int(second.zones), f"{int(second.peak_hour):02d}:00", f"{second.share_00_05:.1%}", f"{second.share_16_20:.1%}"],
                ], [176, 57, 78, 90, 90]),
            p("Table 5. Time-window percentages sum hourly centroid shares within the named window.", "caption"),
            p("Cluster 1 includes East Village, West Village and Lower East Side. Cluster 2 includes Midtown Center, "
              "Upper East Side South and JFK Airport. The airport's placement illustrates how two clusters compress more specific zone patterns. "
              "Zone profiles can help plan where to check overnight supply while the citywide forecast addresses total volume."),
            p("Sensitivity to fitting choices", "h2"),
            tab(["Perturbation", "Mean ARI", "Median ARI", "Lowest ARI"],
                [[label, f"{stability.loc[key, 'mean']:.3f}", f"{stability.loc[key, 'median']:.3f}", f"{stability.loc[key, 'min']:.3f}"]
                 for key, label in (("different_seed", "Change random seed"),
                                    ("80_percent_zones", "Refit on 80% of zones"),
                                    ("standardized_hours", "Standardize each hour"))], [218, 91, 91, 91]),
            p(f"Table 6. Adjusted Rand index (ARI) compares assignments regardless of label order. "
              f"Each perturbation is repeated {config['stability_replicates']} times. The subsample fits assign all zones to their nearest fitted centers before comparison.", "caption"),
            p("Seed changes reproduce the same assignments in these runs. Hour-wise standardization changes a few assignments. "
              "The lowest subsample ARI signals a possible shift in the split when coverage changes."),
        ],
        [
            p("Candidate anomalies and limitations", "h1"),
            p(f"Expected demand is the training median for that hour and day type. The training residual MAD is "
              f"{data['demand_residual_mad']:g} trips. Centering by the training median residual and scaling by 1.4826 times MAD "
              f"flags {len(anomalies)} development hours with |score| at least {config['anomaly_score_threshold']:g} "
              f"({sum(anomalies['split'] == 'train')} training, {sum(anomalies['split'] == 'validation')} validation)."),
            *graphic("04_demand_anomalies.png", "Figure 4. Demand screening across the training and validation windows. The shading marks validation."),
            p(f"The largest signal was {worst.pickup_hour_start[:16]}, with {worst.n_trips:,} pickups versus "
              f"{worst.expected:,.0f} expected. Several high counts around midnight on January 1 are consistent with a "
              "New Year's pattern. It is a plausible event interpretation. January 26 at 23:00 also exceeds the reference. "
              "A single weekday/weekend baseline can fail to represent Friday nights."),
            tab(["Trip-level candidate before cleaning", "Train", "Validation"],
                [[r.candidate, f"{r.train_count:,}", f"{r.validation_count:,}"] for r in flags.itertuples()],
                [299, 96, 96]),
            p("Table 7. Flags may overlap. High speed and invalid duration suggest checking recording quality. "
              "A high positive fare may reflect a real long or special-rate trip and warrants separate review.", "caption"),
            p("Coverage and responsible use", "h2"),
            p("Findings describe recorded, retained yellow taxi trips across one winter month. Coverage excludes other modes and "
              "unserved demand. The outputs publish aggregate time counts and zone profiles without linking individual trips or riders. "
              "The analysis supports exploratory fleet planning within this coverage."),
            p('Sources: <link href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page" color="#126468">NYC TLC Trip Record Data</link> · '
              '<link href="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv" color="#126468">TLC taxi zone lookup</link> · Course project brief, Phase 2.', "caption"),
        ],
    ]
    story = []
    for page in pages:
        if story:
            story.append(PageBreak())
        story.extend(page)
    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build()
