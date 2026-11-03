"""Builds reports/phase1.pdf from the same analysis pipeline used by the
notebook, so the two deliverables never diverge. Running this script also
(re-)runs the full pipeline, so figures/ and data/processed/ are always
regenerated from the raw parquet before the PDF is written.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, ListFlowable, ListItem, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT_PATH = ROOT / "reports" / "phase1.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.5, textColor=colors.grey))
styles.add(ParagraphStyle("Quote", parent=styles["BodyText"], leftIndent=18, textColor=colors.HexColor("#1b3a5c"),
                           borderPadding=6, backColor=colors.HexColor("#eef3f8")))


def df_to_table(df, col_widths=None, font_size=8):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=7.3, leading=9))


def wrapped_table(df, text_cols, col_widths, font_size=7.3):
    """Like df_to_table, but columns listed in text_cols are wrapped as
    Paragraph flowables so long strings break within their column instead
    of overflowing into the next cell."""
    header = list(df.columns)
    data = [header]
    for _, row in df.iterrows():
        cells = []
        for c in header:
            val = str(row[c])
            if c in text_cols:
                cells.append(Paragraph(val, styles["Cell"]))
            else:
                cells.append(val)
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("ALIGN", (1, 0), (-2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def fig(name, width=15 * cm):
    path = FIG_DIR / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, styles["Body"]), leftIndent=6) for t in items],
        bulletType="bullet", start="•",
    )


def build():
    out = A.run_pipeline()
    struct, qa, uni = out["struct"], out["qa"], out["uni"]
    fc = out["filter_counts"]
    borough, zones = out["borough_counts"], out["zone_counts"]
    anomalies = out["anomalies"]

    story = []

    # ---------------- title page ----------------
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Data Analysis Course &ndash; Fall 1405", styles["Title"]))
    story.append(Paragraph("Phase 1: Problem Understanding, Data Understanding, and Exploratory Analysis", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Path A &ndash; NYC TLC Yellow Taxi Trip Records (January 2024)", styles["Heading2"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("<b>Student:</b> Aida Abbasi", styles["Body"]))
    story.append(Paragraph("<b>Student ID:</b> 401170897", styles["Body"]))
    story.append(Paragraph("<b>Repository tag:</b> phase-1", styles["Body"]))
    story.append(PageBreak())

    # ---------------- 1. Problem framing ----------------
    story.append(Paragraph("1. Problem Framing", styles["H1c"]))
    story.append(Paragraph(
        "NYC Yellow Taxi dispatch and fleet-positioning decisions are made ahead of demand: "
        "a dispatcher or a fleet-rebalancing system needs to know, before each hour starts, "
        "roughly how many pickups to expect in order to allocate cabs across the city. This "
        "project turns that operational need into an analyzable question: <b>can short-horizon "
        "citywide pickup demand be forecast from time-of-day, day-of-week, and recent history "
        "alone</b> &mdash; i.e. from information that is genuinely available before the hour in "
        "question, with no information from inside or after the forecast window.", styles["Body"]))
    story.append(Paragraph(
        "The target output is a predicted trip count for a given (date, hour) cell, evaluated "
        "against the actual realized count for that cell. This is deliberately not a vague "
        "“best model” question: the output variable, time granularity, and evaluation window are "
        "fixed up front (Section 14).", styles["Body"]))

    # ---------------- 2. Three candidate questions ----------------
    story.append(Paragraph("2. Candidate Questions and Selected Main Question", styles["H1c"]))
    story.append(Paragraph("Three candidate analytical questions were considered for this dataset:", styles["Body"]))
    story.append(bullets([
        "<b>Q1 (selected &mdash; time-series):</b> Can hourly citywide pickup demand be forecast "
        "from time-of-day, day-of-week and recent history, to support dispatch/supply planning?",
        "<b>Q2 (regression):</b> Can the fare amount of a trip be predicted from trip distance, "
        "pickup time and pickup/dropoff zone, using only information known before or at trip start?",
        "<b>Q3 (unsupervised):</b> Do pickup/dropoff zones cluster into interpretable demand "
        "“regimes” (e.g. commuter corridors vs. nightlife vs. airport) based on their hourly "
        "demand profile shape?",
    ]))
    story.append(Paragraph(
        "<b>Q1 was selected as the main question.</b> Justification: the course syllabus "
        "explicitly designates Path A (NYC TLC) as the project's time-series track, requiring a "
        "serious time-dependent analysis such as demand forecasting (see the “Why only one "
        "path?” note in the project brief). Q1 is also the most operationally direct framing of the dataset: trip-level "
        "fare prediction (Q2) and zone clustering (Q3) remain available as secondary/exploratory "
        "lines of analysis in later phases and are kept as the regression and unsupervised "
        "components of this same project, but Q1 is the question the whole pipeline (baseline, "
        "split, metrics) is built around.", styles["Body"]))

    # ---------------- 3. Data provenance ----------------
    story.append(Paragraph("3. Data Provenance", styles["H1c"]))
    story.append(Paragraph(
        "Source: NYC Taxi &amp; Limousine Commission (TLC) Trip Record Data, official public "
        "dataset (<i>https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page</i>). File used: "
        "<i>yellow_tripdata_2024-01.parquet</i> (Yellow Taxi, January 2024), retrieved "
        "2026-11-03 directly from the TLC's CloudFront distribution "
        "(<i>https://d37ci6vzurychx.cloudfront.net/trip-data/</i>). A companion static reference "
        "table, <i>taxi_zone_lookup.csv</i>, maps the integer location IDs to borough/zone names.",
        styles["Body"]))
    story.append(Paragraph(
        "Unit of observation: one completed Yellow Taxi trip, as recorded by the vehicle's "
        "taxi meter and reported by the technology providers under TLC contract. Collection is "
        "a byproduct of the metering/payment system, not a designed survey &mdash; so values "
        "like distance and fare are meter-derived, while timestamps are logged automatically. "
        "The data is a full administrative census of metered Yellow Taxi trips for the month, "
        "not a sample.", styles["Body"]))
    story.append(Paragraph(
        "<b>Known limitations of the source itself:</b> it covers only Yellow (street-hail) "
        "taxis &mdash; it excludes green boros taxis, for-hire vehicles (Uber/Lyft/Via), and all "
        "non-metered trips, so citywide ground-transportation demand is undercounted, especially "
        "outside Manhattan. Passenger counts are entered by the driver and are known in TLC "
        "documentation to be unreliable. Pickup/dropoff location is reported at the zone level "
        "(265 zones), not exact coordinates.", styles["Body"]))

    # ---------------- 4. Structural audit ----------------
    story.append(Paragraph("4. Structural Audit", styles["H1c"]))
    story.append(Paragraph(
        f"Raw file: <b>{out['raw_shape'][0]:,} rows &times; {out['raw_shape'][1]} columns</b>. "
        "Each row is one trip; there is no natural grouping/repetition within the file beyond "
        "that. The table below lists every column, its dtype, missingness and cardinality.",
        styles["Body"]))
    story.append(df_to_table(struct, font_size=7))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Relationship between files: <i>yellow_tripdata_2024-01.parquet</i> is a single flat "
        "table; <i>taxi_zone_lookup.csv</i> is joined to it via <tt>PULocationID</tt>/"
        "<tt>DOLocationID</tt> &rarr; <tt>LocationID</tt> (many trips to one zone).", styles["Body"]))

    # ---------------- 5. Quality audit ----------------
    story.append(PageBreak())
    story.append(Paragraph("5. Quality Audit", styles["H1c"]))
    story.append(Paragraph(
        "Each issue below is quantified against the raw row count, with the Phase 1 decision "
        "(kept / dropped / flagged) stated explicitly and justified.", styles["Body"]))
    story.append(wrapped_table(
        qa, text_cols=["issue", "initial_decision"],
        col_widths=[5.8 * cm, 1.6 * cm, 1.7 * cm, 7.9 * cm],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>Cleaning funnel applied (ordered):</b>", styles["Body"]))
    funnel_rows = [["step", "rows removed", "rows remaining"]]
    remaining = fc["raw_rows"]
    for k in ["excl_duplicate_rows", "excl_outside_jan2024", "excl_nonpositive_duration",
              "excl_duration_over_4h", "excl_nonpositive_distance", "excl_nonpositive_fare",
              "excl_speed_over_80mph"]:
        remaining -= fc[k]
        funnel_rows.append([k, f"{fc[k]:,}", f"{remaining:,}"])
    import pandas as pd
    story.append(df_to_table(pd.DataFrame(funnel_rows[1:], columns=funnel_rows[0])))
    story.append(Paragraph(
        f"<b>Final Phase-1 analytic population: {fc['analytic_rows']:,} trips</b> "
        f"({100*fc['analytic_rows']/fc['raw_rows']:.1f}% of raw rows).", styles["Body"]))

    # ---------------- 6. Univariate EDA ----------------
    story.append(PageBreak())
    story.append(Paragraph("6. Univariate Exploratory Analysis", styles["H1c"]))
    story.append(df_to_table(uni))
    story.append(Spacer(1, 0.3 * cm))
    for cap, name in [
        ("Trip distance is heavily right-skewed (median 1.7 mi, p95 14.0 mi) &mdash; typical of "
         "an urban taxi market dominated by short Manhattan hops.", "fig1_hist_trip_distance.png"),
        ("Fare amount mirrors the distance skew; a long right tail reflects airport/long-haul "
         "trips.", "fig2_hist_fare_amount.png"),
        ("Trip duration is right-skewed around a ~12-minute median trip.", "fig3_hist_trip_duration.png"),
        ("Implied speed (distance/duration) clusters around 8&ndash;12 mph, consistent with dense "
         "Manhattan traffic; the hard 80&nbsp;mph cutoff used in cleaning is visible at the right "
         "edge.", "fig4_hist_speed.png"),
    ]:
        story.append(fig(name))
        story.append(Paragraph(cap, styles["Small"]))
        story.append(Spacer(1, 0.4 * cm))

    # ---------------- 7. Multivariate EDA ----------------
    story.append(PageBreak())
    story.append(Paragraph("7. Multivariate Exploratory Analysis", styles["H1c"]))
    story.append(fig("fig5_scatter_distance_fare.png"))
    story.append(Paragraph(
        "Distance and fare are strongly, almost linearly related (as expected from the metered "
        "fare formula), which is exactly the relationship a Phase-3 regression baseline for fare "
        "prediction would need to beat.", styles["Small"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(fig("fig6_box_duration_by_hour.png"))
    story.append(Paragraph(
        "Trip duration for a given trip varies systematically with pickup hour &mdash; rush-hour "
        "pickups (8&ndash;9 and 16&ndash;18) take noticeably longer than overnight trips, evidence "
        "of a congestion effect that any duration/fare model should account for.", styles["Small"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(fig("fig10_speed_by_daytype.png"))
    story.append(Paragraph(
        "Implied speed is slightly higher on weekends than weekdays, consistent with lighter "
        "weekend road traffic.", styles["Small"]))

    # ---------------- 8. Path A: trip-time audit ----------------
    story.append(PageBreak())
    story.append(Paragraph("8. Path A &mdash; Trip-Time Audit", styles["H1c"]))
    story.append(fig("fig7_hourly_demand_weekday_weekend.png"))
    story.append(Paragraph(
        "Weekdays show a sharp bimodal commute pattern (peaks near hour 8 and a dominant peak "
        "at hour 18), while weekends show a flatter, later-starting profile with relatively "
        "higher demand in the very early morning (0&ndash;3h, nightlife traffic). This weekday/"
        "weekend shape difference is the core structure the Phase-3 forecasting model needs to "
        "capture.", styles["Body"]))
    story.append(fig("fig8_daily_timeseries.png"))
    story.append(Paragraph(
        "Daily volume follows a clear 7-day cycle (lower on Mondays) with one sharp outlier: "
        "<b>January 8, 2024</b> drops to ~65k trips, far below the surrounding days (~90&ndash;100k). "
        "This date coincides with a winter storm that affected the NYC area in early January 2024; "
        "we note this as a plausible explanation, not a confirmed causal claim (Section 13), and "
        "flag it as a candidate weather-driven anomaly for the time-series model to either exclude "
        "or explicitly model in Phase 3.", styles["Body"]))

    # ---------------- 9. Path A: variable consistency ----------------
    story.append(Paragraph("9. Path A &mdash; Variable Consistency", styles["H1c"]))
    story.append(Paragraph(
        "Distance, duration and fare are cross-checked via implied speed "
        "(<tt>speed_mph = trip_distance / (trip_duration_min/60)</tt>) and fare-per-mile "
        "(<tt>fare_amount / trip_distance</tt>). Implausible-speed records "
        f"(&gt;80&nbsp;mph, n={fc['excl_speed_over_80mph']:,}) were excluded from the main analytic "
        "population as data-entry errors; records with extreme fare-per-mile "
        f"(top 0.1%, n={int(anomalies.loc[anomalies.anomaly_type=='very_high_fare_per_mile','n_flagged'].iloc[0]):,}) "
        "were kept but flagged as anomaly candidates (Section 12) rather than dropped, since very "
        "short, high-surcharge trips (e.g. airport minimum fares) can legitimately produce a high "
        "$/mile ratio.", styles["Body"]))

    # ---------------- 10. Path A: spatial snapshot ----------------
    story.append(Paragraph("10. Path A &mdash; Initial Spatial Snapshot", styles["H1c"]))
    story.append(df_to_table(borough.rename(columns={"n_trips": "n_trips", "pct": "pct_of_trips"})))
    story.append(fig("fig9_borough_share.png"))
    story.append(Paragraph(
        "Pickups are extremely concentrated: Manhattan alone accounts for ~90% of all trips in "
        "the cleaned sample, Queens (largely JFK/LGA airport traffic) for ~9%, and the remaining "
        "boroughs together for &lt;1%. This is a strong class/zone imbalance: any model or metric "
        "that reports only a citywide aggregate will be dominated by Manhattan behavior and could "
        "silently hide poor performance in the outer boroughs &mdash; a concrete instance of the "
        "zone-level evaluation requirement for this path.", styles["Body"]))
    story.append(Paragraph("Top 10 individual pickup zones:", styles["Body"]))
    story.append(df_to_table(zones, font_size=7))

    # ---------------- 11. Time-series question & split ----------------
    story.append(PageBreak())
    story.append(Paragraph("11. Time-Series Question Definition", styles["H1c"]))
    story.append(Paragraph(
        "<b>Question:</b> forecast <tt>n_trips(date, hour)</tt> &mdash; the citywide pickup count "
        "for each hour &mdash; using only information available strictly before that hour begins "
        "(calendar features: hour-of-day, day-of-week, holiday flag; and lagged demand from prior "
        "hours/days). <b>Planned temporal split (Phase 3):</b> train on the first three weeks of "
        "January 2024 (Jan 1&ndash;21), validate on Jan 22&ndash;28, and hold out the final days of "
        "the month as test &mdash; chronological, with no shuffling, so no future hour is ever used "
        "to predict an earlier one.", styles["Body"]))

    # ---------------- 12. Anomaly candidates ----------------
    story.append(Paragraph("12. Anomaly Candidates", styles["H1c"]))
    story.append(df_to_table(anomalies))
    story.append(Paragraph(
        "These are retained in the data (not dropped) because each has a plausible benign "
        "explanation (idle meter while waiting for a passenger, a legitimate minimum/flat fare, "
        "an unreported passenger count) as well as a plausible error explanation; distinguishing "
        "the two needs more structure than Phase 1 EDA alone provides, so they are logged here as "
        "the starting point for Phase 4's error-taxonomy work.", styles["Body"]))

    # ---------------- 13. Leakage risk ----------------
    story.append(Paragraph("13. Information Leakage Risk", styles["H1c"]))
    story.append(bullets([
        "<b>total_amount, tip_amount, tolls_amount, improvement_surcharge, congestion_surcharge, "
        "Airport_fee</b> are all computed at or after trip completion (several are deterministic "
        "functions of fare_amount and the trip's characteristics). None of these may be used as "
        "predictors for a pre-trip fare or demand forecast &mdash; only post-hoc as the prediction "
        "target itself or for descriptive analysis.",
        "<b>tpep_dropoff_datetime</b> and any duration/speed feature derived from it leak "
        "directly into a duration- or arrival-time prediction task, since the dropoff time is "
        "only known once the trip is already over.",
        "For the selected <b>demand-forecasting</b> question specifically, the main leakage risk "
        "is using same-hour or future lagged-demand features (e.g. an aggregate that accidentally "
        "includes trips from the hour being predicted) &mdash; all lag features must be built from "
        "strictly past hours relative to the forecast hour.",
    ]))

    # ---------------- 14. Baseline ----------------
    story.append(Paragraph("14. Baseline Definition", styles["H1c"]))
    story.append(Paragraph(
        "<b>Baseline for the main (demand-forecasting) question:</b> predict next hour's trip "
        "count as the mean trip count observed in that same (day-of-week, hour-of-day) cell over "
        "the training weeks (a seasonal-naive baseline). It uses no model fitting, is fully "
        "interpretable, and must be beaten by any Phase-3 model for that model to be considered "
        "useful. <b>Secondary baseline for the fare-regression side question (Q2):</b> predict "
        "fare_amount as a simple linear function of trip_distance alone (median $/mile from the "
        "training data &times; distance), ignoring time, location and all other features.", styles["Body"]))

    # ---------------- 15. Proposed evaluation metrics ----------------
    story.append(Paragraph("15. Proposed Evaluation Metrics", styles["H1c"]))
    story.append(bullets([
        "Demand forecasting (main question): MAE and RMSE of hourly trip counts; MAPE reported "
        "per zone/borough as well as citywide, to avoid Manhattan dominating the error average.",
        "Fare regression (secondary): RMSE and R² against the linear-distance baseline.",
        "Unsupervised zone clustering (secondary): qualitative interpretability of clusters plus "
        "silhouette score as a sanity check, not as the primary success criterion.",
    ]))

    # ---------------- 16. Preliminary training/eval split plan ----------------
    story.append(Paragraph("16. Preliminary Training / Validation / Test Split Plan", styles["H1c"]))
    story.append(Paragraph(
        "Chronological split only (see Section 11): train = Jan 1&ndash;21, validation = Jan "
        "22&ndash;28, test = Jan 29&ndash;31. The same chronological boundaries will be reused for "
        "the fare-regression side question so that no trip after the training cutoff ever "
        "influences training in any analysis in this project.", styles["Body"]))

    # ---------------- 17. Top data-leakage risks (summary) & ethics ----------------
    story.append(Paragraph("17. Privacy, Ethics and Governance (preliminary)", styles["H1c"]))
    story.append(Paragraph(
        "The dataset contains no passenger names, payment-card numbers, or exact GPS "
        "coordinates &mdash; locations are reported at the 265-zone level specifically to limit "
        "re-identification risk, and this project does not attempt any join, enrichment or "
        "inference intended to re-identify a driver or passenger from the trip records. Pickup/"
        "dropoff zone and time, in combination, carry some residual re-identification risk for "
        "very rare origin-destination pairs at unusual hours (e.g. outer-borough late-night "
        "trips with only one or two monthly occurrences); any zone-level reporting in later "
        "phases will aggregate small cells rather than publish raw rare-pair counts.", styles["Body"]))

    # ---------------- 18. Limitations & next steps ----------------
    story.append(Paragraph("18. Limitations and Next Steps", styles["H1c"]))
    story.append(bullets([
        "Single month (January 2024, winter seasonality only) &mdash; findings about demand "
        "shape may not generalize to summer months; this is a scope limitation to be revisited "
        "if a multi-month extension is pursued.",
        "Only Yellow Taxi trips are covered; citywide ground-transportation demand (including "
        "For-Hire Vehicles) is necessarily undercounted.",
        "Passenger_count is self/driver-reported and known to be unreliable per TLC "
        "documentation; it is not used as a predictor anywhere in this project.",
        "Phase 2 will turn the cleaning/feature decisions above into a reproducible "
        "preprocessing pipeline and add the lag/calendar features needed for the Phase-3 "
        "forecasting model.",
    ]))

    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Phase 1 Report - NYC TLC - Aida Abbasi 401170897",
        author="Aida Abbasi",
        subject="Data Analysis Course, Fall 1405 - Phase 1 Report",
    )
    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
