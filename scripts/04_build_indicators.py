"""
Mobility Feature Builder — leakage-safe raw-feature version
===========================================================
Builds individual-level mobility features inspired by Schultheiss & Kaufmann
(2025), with the Schneider et al. (2013) daily-motif pipeline.

Important separation of responsibilities
----------------------------------------
This script ONLY builds and exports raw mobility features. It does not fit
MinMaxScaler, StandardScaler, skewness rules, variance filters, rare-motif
filters, or correlation filters on the complete population.

All data-dependent preprocessing must be fitted inside each training fold.
The bottom of this file therefore provides sklearn-compatible transformers and
`make_model_pipeline(...)`, which can be used inside cross-validation without
leaking information from validation or test participants.

Main corrections relative to the previous version
-------------------------------------------------
1. Convex-hull area is now the true geometric convex-hull area, not covariance-
   ellipse area.
2. Visit-frequency weighting is performed once at the unique-location level;
   repeated observations are no longer accidentally squared in weight.
3. k-radius of gyration uses the top-k locations ranked by visit frequency.
4. Travel features and graph edges use distinct consecutive locations and
   geodesic leg distances.
5. Travel rhythm uses transition departure times rather than stay-start counts.
6. Land-use ratios and home/work profiles are duration-weighted, as documented.
7. Work hours are computed from stay duration, not the number of occupied hour
   bins.
8. Graph betweenness uses inverse trip frequency as path cost; frequent edges
   are therefore easier, not harder, to traverse.
9. Rare-motif, near-zero-variance, skewness, correlation, and scaling decisions
   are fitted only on training data through the supplied sklearn pipeline.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import json
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import skew as scipy_skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('default')



# =============================================================================
# <<< section: config >>>
# =============================================================================

trace_file = Path('/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv')

# ─────────────────────────────────────────────────────────────────────────────
# DATA WINDOW SWITCH
#   True  → keep only the first 4 weeks of each user's trajectory
#   False → use the full trajectory (all available data)
# ─────────────────────────────────────────────────────────────────────────────
USE_4WEEK_FILTER: bool = True

# Output directory is chosen automatically based on the switch above.
# 4-week run  → .../11_mobility_features_4weeks
# Full-data run → .../11_mobility_features_full
_out_dir_4weeks = Path('/data/baliu/thesis/09_indicators/2_mobility_features_4weeks')
_out_dir_full   = Path('/data/baliu/thesis/09_indicators/2_mobility_features_full')
out_dir = _out_dir_4weeks if USE_4WEEK_FILTER else _out_dir_full
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'figures').mkdir(exist_ok=True)

user_col      = 'user_id'
timestamp_col = 'started_at'

# MOBIS timestamps describe activities in Switzerland. Input files may mix
# timezone-naive values with ISO timestamps carrying UTC/offset information.
# All timestamps are therefore converted to Europe/Zurich wall-clock time and
# stored as timezone-naive datetime64[ns]. This keeps calendar-day and
# time-of-day features correct while preventing aware/naive comparison errors.
LOCAL_TIMEZONE = 'Europe/Zurich'

# Machine-learning preprocessing defaults. These are NOT applied while
# building the raw feature matrix; they are fitted inside each training fold.
ML_CORR_THRESHOLD       = 0.75
ML_SKEW_THRESHOLD       = 2.0
ML_NZV_FREQ_RATIO       = 19.0   # 95 / 5, analogous to caret::nearZeroVar
ML_NZV_UNIQUE_PCT       = 10.0
ML_MOTIF_MIN_FREQ_PCT   = 5.0

# Motif pipeline settings
motif_max_leg_km = 300.0  # filter (3b): discard diary if any leg > 300 km

# Motif type names — defines column order in output, change the node to purporse, instead of location id
motif_types = [
    'motif_3',   # stay-home all day  (filter 3c)
    'motif_1',   # H→A→H              n=2
    'motif_2',   # H→A→B→H            n=3  triangle
    'motif_4',   # H→A→B→C→H          n=4  three-activity chain
    'motif_5',   # H→A→H→B→H          n=3  star-2, home interior
    'motif_6',   # H→A→B→A→H          n=3  return chain
    'motif_7',   # H→A→B→A→B→H        n=3  no home interior
    'motif_8',   # H→A→H→A→B→H        n=3  home interior + chain (covers motif_9)
    'motif_9',   # H→B→H→B→A→H        n=3  binary-equiv to motif_8, always 0
    'motif_99',  # catchall
]

# Swiss federal zoning → land-use category mapping
land_use_map = {
    'wohnzonen'                            : 'residential',
    'mischzonen'                           : 'mixed',
    'zentrumszonen'                        : 'central',
    'arbeitszonen'                         : 'working',
    'zonen für öffentliche nutzungen'      : 'public',
    'eingeschränkte bauzonen'              : 'restricted',
    'verkehrszonen innerhalb der bauzonen' : 'transport',
    'tourismus- und freizeitzonen'         : 'leisure',
    'weitere bauzonen'                     : 'other',
}

land_use_cats = [
    'residential', 'transport', 'mixed', 'other',
    'central', 'working', 'public', 'restricted', 'leisure',
]

# Alias pairs: if both present in final feature set, drop the second
alias_groups = [
    ['travel_rhythm_entropy', 'rhythm_q8_slot0'],
]

# Human-readable labels for heatmap axes
feature_labels = {
    # 1st-order point level
    'stay_point_count'          : 'Number of stay points N',
    'unique_stay_locations'     : 'Number of unique stay points',
    'stay_entropy'              : 'Stay-point entropy',
    'stay_radius_of_gyration'   : 'Radius of gyration',
    'stay_convex_hull_diameter' : 'Diameter of convex hull (km)',
    'stay_area_km2'             : 'Area of convex hull (km²)',
    'stay_eccentricity'         : 'Eccentricity ε',
    'stay_direction_deg'        : 'Direction θ (°)',
    # 2nd-order line level
    'n_travels'                 : 'Number of travels',
    'total_travel_length_km'    : 'Total travel length (km)',
    'mean_travel_length_km'     : 'Mean travel length (km)',
    'od_entropy'                : 'Travel entropy (OD pairs)',
    # Motif ratios
    'motif_3_ratio'             : 'Motif 3 – stay-home all day',
    'motif_1_ratio'             : 'Motif 1 – H→A→H',
    'motif_2_ratio'             : 'Motif 2 – H→A→B→H (triangle)',
    'motif_4_ratio'             : 'Motif 4 – H→A→B→C→H (chain)',
    'motif_5_ratio'             : 'Motif 5 – H→A→H→B→H (star-2)',
    'motif_6_ratio'             : 'Motif 6 – H→A→B→A→H (return)',
    'motif_7_ratio'             : 'Motif 7 – H→A→B→A→B→H',
    'motif_8_ratio'             : 'Motif 8 – H→A→H→A→B→H',
    'motif_99_ratio'            : 'Motif 99 – catchall',
    'motif_stayhome_days'       : 'Motif stay-home days',
    # Temporal
    'time_fragmented'           : 'Time fragmentation (SD hours)',
    'travel_rhythm_entropy'     : 'Entropy of travel rhythm',
    'rhythm_morning'            : 'Rhythm – morning (06–12 h)',
    'rhythm_afternoon'          : 'Rhythm – afternoon (12–18 h)',
    'rhythm_evening'            : 'Rhythm – evening (18–24 h)',
    # Space-time integrated
    'k_rog_ratio_2'             : 'k-RoG ratio (k=2)',
    'top1_visit_frequency'      : 'Top-1 location visit frequency',
    'top2_visit_frequency'      : 'Top-2 location visit frequency',
    # Semantic – purpose
    'purpose_work_ratio'        : 'Work time %',
    'purpose_home_ratio'        : 'Home time %',
    'purpose_leisure_ratio'     : 'Leisure time %',
    'purpose_entropy'           : 'Purpose entropy',
    # Semantic – top-location summaries
    'top1_dur_ratio'            : 'Top-1 location time share',
    'top2_dur_ratio'            : 'Top-2 location time share',
    'top1_top2_dur_ratio'       : 'Top-1 / Top-2 dominance ratio',
    'top1_purpose_home'         : 'Top-1 location is home (fraction)',
    # Semantic – land use
    'landuse_residential_ratio' : 'Land use – residential %',
    'landuse_working_ratio'     : 'Land use – working zones %',
    'landuse_mixed_ratio'       : 'Land use – mixed zones %',
    'landuse_entropy'           : 'Land use entropy',
    'home_work_lu_contrast'     : 'Home vs. work land-use contrast',
    # Semantic – work / SES proxy
    'commute_dist_km'           : 'Commute distance (km)',
    'dist_per_work_trip'        : 'Mean work-trip distance (km)',
    'work_peak_ratio'           : 'Work trips in peak hours %',
    'work_travel_intensity'     : 'Work-trip distance / total distance',
    'work_hour_dist_product'    : 'Work hours × daily distance',
    # Multi-day directed graph
    'n_nodes'                   : 'Graph – unique locations (nodes)',
    'n_edges'                   : 'Graph – unique OD pairs (edges)',
    'graph_density'             : 'Graph – density',
    'n_weakly_connected'        : 'Graph – weakly connected components',
    'n_strongly_connected'      : 'Graph – strongly connected components',
    'home_in_degree'            : 'Graph – home in-degree',
    'home_out_degree'           : 'Graph – home out-degree',
    'home_betweenness'          : 'Graph – home betweenness centrality',
    'mean_betweenness'          : 'Graph – mean betweenness centrality',
    'home_pagerank'             : 'Graph – home PageRank',
    'max_pagerank'              : 'Graph – max PageRank',
    'mean_edge_weight'          : 'Graph – mean trip frequency',
    'max_edge_weight'           : 'Graph – max trip frequency',
    'edge_weight_entropy'       : 'Graph – trip frequency entropy',
    'mean_edge_dist_km'         : 'Graph – mean trip distance (km)',
    'total_edge_dist_km'        : 'Graph – total weighted distance (km)',
    'reciprocity'               : 'Graph – reciprocity (A↔B fraction)',
}

# <<< end: config >>>


# =============================================================================
# <<< section: geometry_functions >>>
# =============================================================================

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(
        lambda x: np.radians(np.asarray(x, dtype=float)),
        [lat1, lon1, lat2, lon2],
    )
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_local_naive_timestamp(value) -> pd.Timestamp:
    """Convert one timestamp to Europe/Zurich local wall time without tz info.

    Rules
    -----
    * timezone-naive inputs are assumed to already represent Swiss local time;
    * timezone-aware inputs are converted to Europe/Zurich first;
    * invalid or missing values become ``pd.NaT``.

    Returning one consistent representation is essential because pandas cannot
    sort timezone-aware and timezone-naive ``Timestamp`` objects together.
    """
    if value is None or value is pd.NaT:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass

    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT

    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert(LOCAL_TIMEZONE).tz_localize(None)


def _normalise_datetime_series(values: pd.Series) -> pd.Series:
    """Return a homogeneous timezone-naive local datetime Series.

    The fast paths avoid element-wise conversion when a column has already
    been normalised. Object columns are handled value by value so files that
    mix naive strings, ``Z`` timestamps and explicit UTC offsets remain valid.
    """
    series = pd.Series(values, index=values.index, name=values.name, copy=False)

    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)

    if pd.api.types.is_datetime64_dtype(series.dtype):
        return pd.to_datetime(series, errors='coerce')

    converted = series.map(_to_local_naive_timestamp)
    return pd.to_datetime(converted, errors='coerce')


def _safe_entropy(probabilities: np.ndarray, base: float = np.e) -> float:
    """Shannon entropy for a probability vector, ignoring zero entries."""
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    logs = np.log(p)
    if base != np.e:
        logs = logs / np.log(base)
    return float(-(p * logs).sum())


def _resolve_purpose(grp: pd.DataFrame) -> pd.Series:
    """Prefer labelled purpose when available; otherwise use imputed purpose."""
    if 'act_imputed_purpose' in grp.columns:
        purpose = grp['act_imputed_purpose'].astype('string').str.lower().fillna('unknown')
    else:
        purpose = pd.Series('unknown', index=grp.index, dtype='string')

    if 'act_labeled_purpose' in grp.columns:
        labelled = grp['act_labeled_purpose'].astype('string').str.lower()
        valid = labelled.notna() & (labelled != '') & (labelled != 'unknown')
        purpose = labelled.where(valid, purpose)
    return purpose.fillna('unknown').astype(str)


def _compute_stay_durations_seconds(grp: pd.DataFrame) -> pd.Series:
    """
    Return non-negative stay duration in seconds.

    `finished_at - started_at` is preferred. Missing durations are estimated
    from the next stay's start time within the same user, while the final
    missing duration is set to the median observed positive duration (or zero).
    """
    grp = grp.sort_values(timestamp_col)
    started = pd.to_datetime(grp[timestamp_col], errors='coerce')

    if 'finished_at' in grp.columns:
        finished = pd.to_datetime(grp['finished_at'], errors='coerce')
        duration = (finished - started).dt.total_seconds()
    else:
        duration = pd.Series(np.nan, index=grp.index, dtype=float)

    next_gap = (started.shift(-1) - started).dt.total_seconds()
    duration = duration.where(duration.notna(), next_gap)
    duration = duration.where(duration >= 0)

    positive = duration[(duration > 0) & np.isfinite(duration)]
    fallback = float(positive.median()) if not positive.empty else 0.0
    duration = duration.fillna(fallback).clip(lower=0)
    return duration.reindex(grp.index).astype(float)


def _prepare_user_group(grp: pd.DataFrame) -> pd.DataFrame:
    """Sort one user and add internally consistent purpose/duration/land-use fields."""
    out = grp.copy()
    out[timestamp_col] = _normalise_datetime_series(out[timestamp_col])
    if 'finished_at' in out.columns:
        out['finished_at'] = _normalise_datetime_series(out['finished_at'])
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    out['_purpose'] = _resolve_purpose(out)
    out['_duration_s'] = _compute_stay_durations_seconds(out).to_numpy()

    if 'act_CH_BEZ_D' in out.columns:
        raw_lu = out['act_CH_BEZ_D'].astype('string').str.lower().str.strip()
        out['_lu'] = raw_lu.map(land_use_map).fillna('other').astype(str)
        out.loc[out['act_CH_BEZ_D'].isna(), '_lu'] = 'unknown'
    else:
        out['_lu'] = 'unknown'
    return out


def _location_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate visits to unique location IDs without double weighting."""
    prepared = _prepare_user_group(grp)
    summary = (
        prepared.groupby('location_id', dropna=False)
        .agg(
            lat=('lat', 'mean'),
            lon=('lon', 'mean'),
            visit_count=('location_id', 'size'),
            total_duration_s=('_duration_s', 'sum'),
        )
        .reset_index()
    )
    summary['location_id'] = summary['location_id'].astype(str)
    return summary


def _weighted_covariance_2d(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Population weighted covariance matrix for 2-D coordinates."""
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    dx = x - mx
    dy = y - my
    return np.array(
        [
            [np.sum(w * dx * dx), np.sum(w * dx * dy)],
            [np.sum(w * dx * dy), np.sum(w * dy * dy)],
        ],
        dtype=float,
    )


def _project_local_km(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection centred on the supplied coordinates."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    x = (lons - lon0) * 111.320 * np.cos(np.radians(lat0))
    y = (lats - lat0) * 110.574
    return x, y


def stay_point_entropy(locs: pd.Series) -> float:
    """Entropy of visit frequencies over unique stay locations."""
    p = locs.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
    return _safe_entropy(p)


def radius_of_gyration_from_summary(
    summary: pd.DataFrame,
    weight_col: str = 'visit_count',
) -> float:
    """Weighted radius of gyration over unique locations."""
    if len(summary) < 2:
        return 0.0
    weights = summary[weight_col].to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(summary), dtype=float)
    weights = weights / weights.sum()

    lats = summary['lat'].to_numpy(dtype=float)
    lons = summary['lon'].to_numpy(dtype=float)
    centre_lat = float(np.sum(weights * lats))
    centre_lon = float(np.sum(weights * lons))
    distances = haversine_km(lats, lons, centre_lat, centre_lon)
    return float(np.sqrt(np.sum(weights * distances ** 2)))


def radius_of_gyration(
    lats: np.ndarray,
    lons: np.ndarray,
    locs: pd.Series,
    weighted: bool = True,
) -> float:
    """Compatibility wrapper using one weight per unique location."""
    df = pd.DataFrame({'lat': lats, 'lon': lons, 'location_id': locs.astype(str)})
    summary = (
        df.groupby('location_id')
        .agg(lat=('lat', 'mean'), lon=('lon', 'mean'), visit_count=('location_id', 'size'))
        .reset_index()
    )
    if not weighted:
        summary['visit_count'] = 1.0
    return radius_of_gyration_from_summary(summary, 'visit_count')


def convex_hull_diameter(lats: np.ndarray, lons: np.ndarray) -> float:
    """Maximum pairwise great-circle distance among unique coordinates."""
    coords = pd.DataFrame({'lat': lats, 'lon': lons}).drop_duplicates().to_numpy(dtype=float)
    if len(coords) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(coords) - 1):
        d = haversine_km(
            coords[i, 0], coords[i, 1], coords[i + 1 :, 0], coords[i + 1 :, 1]
        )
        if d.size:
            max_dist = max(max_dist, float(np.max(d)))
    return max_dist


def convex_hull_area_km2(lats: np.ndarray, lons: np.ndarray) -> float:
    """True convex-hull area in square kilometres after local projection."""
    coords = pd.DataFrame({'lat': lats, 'lon': lons}).drop_duplicates()
    if len(coords) < 3:
        return 0.0
    x, y = _project_local_km(coords['lat'].to_numpy(), coords['lon'].to_numpy())
    points = np.column_stack([x, y])
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2-D scipy stores polygon area in `volume`.
    except QhullError:
        return 0.0


def compute_eccentricity_and_direction(
    summary: pd.DataFrame,
    weight_col: str = 'visit_count',
) -> tuple[float, float]:
    """
    Frequency-weighted eccentricity and principal-axis orientation.

    Direction is an undirected axis in [0, 180) degrees measured anticlockwise
    from the local east-west x-axis. The modulo removes arbitrary eigenvector
    sign flips.
    """
    if len(summary) < 2:
        return 0.0, 0.0
    x, y = _project_local_km(
        summary['lat'].to_numpy(dtype=float),
        summary['lon'].to_numpy(dtype=float),
    )
    cov = _weighted_covariance_2d(x, y, summary[weight_col].to_numpy(dtype=float))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    lam1 = float(eigvals[0])
    lam2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    if lam1 <= 1e-15:
        return 0.0, 0.0
    eccentricity = float(np.sqrt(max(0.0, 1.0 - lam2 / lam1)))
    major = eigvecs[:, 0]
    direction = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return eccentricity, direction


def k_radius_of_gyration(summary: pd.DataFrame, k: int = 2) -> float:
    """RoG of the k most frequently visited unique locations."""
    if len(summary) < 2:
        return 0.0
    top = summary.sort_values(['visit_count', 'total_duration_s'], ascending=False).head(k)
    return radius_of_gyration_from_summary(top, weight_col='visit_count')


def _build_transition_table(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per transition between distinct consecutive locations.

    Distance is calculated from stay-point coordinates, avoiding ambiguity in
    whether an input `length_km` field belongs to the preceding or following
    stay. Departure time is the source stay's finish time when available,
    otherwise the destination stay's start time.
    """
    g = _prepare_user_group(grp)
    if len(g) < 2:
        return pd.DataFrame(
            columns=['src', 'dst', 'distance_km', 'departure_time', 'arrival_time',
                     'destination_purpose', 'mode', 'travel_duration_h']
        )

    records: list[dict] = []
    for i in range(len(g) - 1):
        src_row = g.iloc[i]
        dst_row = g.iloc[i + 1]
        src = str(src_row['location_id'])
        dst = str(dst_row['location_id'])
        if src == dst:
            continue

        arrival = pd.to_datetime(dst_row[timestamp_col], errors='coerce')
        if 'finished_at' in g.columns and pd.notna(src_row.get('finished_at')):
            departure = pd.to_datetime(src_row['finished_at'], errors='coerce')
        else:
            departure = arrival
        if pd.isna(departure):
            departure = pd.to_datetime(src_row[timestamp_col], errors='coerce')

        travel_duration_h = 0.0
        if pd.notna(departure) and pd.notna(arrival):
            travel_duration_h = max((arrival - departure).total_seconds() / 3600.0, 0.0)

        records.append({
            'src': src,
            'dst': dst,
            'distance_km': float(haversine_km(
                src_row['lat'], src_row['lon'], dst_row['lat'], dst_row['lon']
            )),
            'departure_time': departure,
            'arrival_time': arrival,
            'destination_purpose': str(dst_row['_purpose']),
            'mode': str(dst_row['mode']) if 'mode' in g.columns and pd.notna(dst_row.get('mode')) else 'unknown',
            'travel_duration_h': float(travel_duration_h),
        })
    return pd.DataFrame.from_records(records)
# <<< end: geometry_functions >>>


# =============================================================================
# <<< section: motif_functions >>>
# =============================================================================
# Daily motif pipeline
# =============================================================================


def _make_ref_motifs() -> list[dict]:
    """Reference binary adjacency patterns using home as node index 0."""
    def _mat(*edges, n: int) -> tuple:
        matrix = np.zeros((n, n), dtype=np.int8)
        for i, j in edges:
            matrix[i, j] = 1
        return tuple(matrix.ravel().tolist())

    return [
        {'label': 'motif_1', 'n': 2, 'pattern': _mat((0, 1), (1, 0), n=2)},
        {'label': 'motif_2', 'n': 3, 'pattern': _mat((0, 1), (1, 2), (2, 0), n=3)},
        {'label': 'motif_5', 'n': 3, 'pattern': _mat((0, 1), (1, 0), (0, 2), (2, 0), n=3)},
        {'label': 'motif_6', 'n': 3, 'pattern': _mat((0, 1), (1, 2), (2, 1), (1, 0), n=3)},
        {'label': 'motif_7', 'n': 3, 'pattern': _mat((0, 1), (1, 2), (2, 1), (2, 0), n=3)},
        {'label': 'motif_8', 'n': 3, 'pattern': _mat((0, 1), (1, 0), (1, 2), (2, 0), n=3)},
        {'label': 'motif_4', 'n': 4, 'pattern': _mat((0, 1), (1, 2), (2, 3), (3, 0), n=4)},
    ]


_REF_MOTIFS = _make_ref_motifs()


def _identify_home_node(grp: pd.DataFrame) -> str:
    """Choose the dominant home-labelled location, prioritising total duration."""
    g = _prepare_user_group(grp)
    home_rows = g[g['_purpose'] == 'home']
    candidate = home_rows if not home_rows.empty else g
    duration_by_location = candidate.groupby('location_id')['_duration_s'].sum()
    if not duration_by_location.empty and duration_by_location.max() > 0:
        return str(duration_by_location.idxmax())
    return str(candidate['location_id'].astype(str).value_counts().idxmax())


def _build_diary_days(grp: pd.DataFrame, home_id: str) -> dict:
    """
    Assign stays to diary dates and duplicate midnight-spanning home stays into
    every covered calendar day. A synthetic `_diary_order` timestamp places the
    duplicated home at the beginning of subsequent days.
    """
    g = _prepare_user_group(grp)
    diaries: dict = {}

    for _, row in g.iterrows():
        started = pd.to_datetime(row[timestamp_col], errors='coerce')
        if pd.isna(started):
            continue
        finished = (
            pd.to_datetime(row['finished_at'], errors='coerce')
            if 'finished_at' in g.columns and pd.notna(row.get('finished_at'))
            else started
        )
        if pd.isna(finished) or finished < started:
            finished = started

        is_home = str(row['location_id']) == home_id
        row_dict = row.to_dict()

        if is_home and finished.date() > started.date():
            # Treat stays as half-open intervals [started, finished). A stay
            # ending exactly at 00:00 contributes no time to the finish date.
            end_date = finished.date()
            if finished == finished.normalize():
                end_date -= timedelta(days=1)

            current = started.date()
            while current <= end_date:
                copy = dict(row_dict)
                if current == started.date():
                    copy['_diary_order'] = started
                else:
                    # Use the same local-naive representation as the source
                    # timestamps. This avoids mixing tz-aware and tz-naive
                    # values inside `_diary_order`.
                    copy['_diary_order'] = pd.Timestamp(current)
                diaries.setdefault(current, []).append(copy)
                current += timedelta(days=1)
        else:
            row_dict['_diary_order'] = started
            diaries.setdefault(started.date(), []).append(row_dict)

    return diaries


def _collapse_consecutive_locations(day: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive duplicate location IDs because they are not travels."""
    if day.empty:
        return day
    day = day.copy()
    day['_diary_order'] = _normalise_datetime_series(day['_diary_order'])
    ordered = day.dropna(subset=['_diary_order']).sort_values('_diary_order').reset_index(drop=True)
    loc = ordered['location_id'].astype(str)
    return ordered.loc[loc.ne(loc.shift())].reset_index(drop=True)


def _apply_diary_filters(
    sequence: list[str],
    home_id: str,
    leg_distances: np.ndarray,
) -> str | None:
    if not sequence:
        return 'invalid_boundary'
    if set(sequence) <= {home_id}:
        return 'stay_home'
    if sequence[0] != home_id or sequence[-1] != home_id:
        return 'invalid_boundary'
    if leg_distances.size and np.nanmax(leg_distances) > motif_max_leg_km:
        return 'long_distance'
    return None


def _build_adjacency_matrix(sequence: list[str], home_id: str) -> tuple[tuple, int]:
    seen = {home_id: 0}
    for loc in sequence:
        if loc not in seen:
            seen[loc] = len(seen)
    n = len(seen)
    matrix = np.zeros((n, n), dtype=np.int8)
    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src != dst:
            matrix[seen[src], seen[dst]] = 1
    return tuple(matrix.ravel().tolist()), n


def _match_motif(flat_matrix: tuple, n_nodes: int) -> str:
    for ref in _REF_MOTIFS:
        if ref['n'] == n_nodes and ref['pattern'] == flat_matrix:
            return ref['label']
    return 'motif_99'


def compute_motif_features(grp: pd.DataFrame) -> dict:
    """Return all motif ratios; training-fold rare-motif filtering happens later."""
    home_id = _identify_home_node(grp)
    counts = {motif: 0 for motif in motif_types}
    valid_days = 0

    for date, rows in sorted(_build_diary_days(grp, home_id).items()):
        day = _collapse_consecutive_locations(pd.DataFrame(rows))
        sequence = day['location_id'].astype(str).tolist()

        if len(day) >= 2:
            leg_distances = haversine_km(
                day['lat'].to_numpy()[:-1], day['lon'].to_numpy()[:-1],
                day['lat'].to_numpy()[1:], day['lon'].to_numpy()[1:],
            )
        else:
            leg_distances = np.array([], dtype=float)

        status = _apply_diary_filters(sequence, home_id, leg_distances)
        if status == 'stay_home':
            counts['motif_3'] += 1
            valid_days += 1
            continue
        if status is not None:
            continue

        flat, n_nodes = _build_adjacency_matrix(sequence, home_id)
        counts[_match_motif(flat, n_nodes)] += 1
        valid_days += 1

    denominator = max(valid_days, 1)
    return {f'{motif}_ratio': counts[motif] / denominator for motif in motif_types}


def motif_frequency_report(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive report only; it never removes columns from the full dataset."""
    motif_cols = [c for c in feat_df.columns if c.startswith('motif_') and c.endswith('_ratio')]
    report = pd.DataFrame({
        'feature': motif_cols,
        'population_mean_ratio': [float(feat_df[c].mean()) for c in motif_cols],
    })
    report['population_mean_pct'] = 100.0 * report['population_mean_ratio']
    report['would_be_rare_at_default_threshold'] = (
        report['population_mean_pct'] < ML_MOTIF_MIN_FREQ_PCT
    )
    return report.sort_values('population_mean_pct', ascending=False)
# <<< end: motif_functions >>>


# =============================================================================
# <<< section: multiday_directed_graph_features >>>
# =============================================================================
# Multi-day directed graph features
# =============================================================================


def build_user_graph(grp: pd.DataFrame) -> nx.DiGraph:
    """Build a weighted directed graph from distinct consecutive locations."""
    g = _prepare_user_group(grp)
    graph = nx.DiGraph()
    home_id = _identify_home_node(g)

    for loc_id, loc_grp in g.groupby('location_id'):
        purpose_dur = loc_grp.groupby('_purpose')['_duration_s'].sum()
        lu_dur = loc_grp.groupby('_lu')['_duration_s'].sum()
        graph.add_node(
            str(loc_id),
            lat=float(loc_grp['lat'].mean()),
            lon=float(loc_grp['lon'].mean()),
            dominant_purpose=str(purpose_dur.idxmax()) if not purpose_dur.empty else 'unknown',
            landuse=str(lu_dur.idxmax()) if not lu_dur.empty else 'unknown',
            total_duration_h=float(loc_grp['_duration_s'].sum() / 3600.0),
            visit_count=int(len(loc_grp)),
            is_home=(str(loc_id) == home_id),
            dist_to_centre=(
                float(pd.to_numeric(loc_grp['dist_to_centre'], errors='coerce').mean())
                if 'dist_to_centre' in loc_grp.columns else 0.0
            ),
        )

    transitions = _build_transition_table(g)
    edge_data = defaultdict(lambda: {
        'count': 0,
        'distances': [],
        'durations': [],
        'modes': [],
    })
    for row in transitions.itertuples(index=False):
        edge = edge_data[(row.src, row.dst)]
        edge['count'] += 1
        edge['distances'].append(float(row.distance_km))
        edge['durations'].append(float(row.travel_duration_h))
        edge['modes'].append(str(row.mode))

    for (src, dst), edge in edge_data.items():
        weights = np.asarray(edge['distances'], dtype=float)
        mode_counts = pd.Series(edge['modes']).value_counts()
        graph.add_edge(
            src,
            dst,
            weight=int(edge['count']),
            cost=1.0 / float(edge['count']),
            mean_dist_km=float(weights.mean()) if weights.size else 0.0,
            total_dist_km=float(weights.sum()) if weights.size else 0.0,
            mean_duration_h=(
                float(np.mean(edge['durations'])) if edge['durations'] else 0.0
            ),
            dominant_mode=str(mode_counts.idxmax()) if not mode_counts.empty else 'unknown',
        )
    return graph


def extract_graph_features(graph: nx.DiGraph) -> dict:
    """Extract the 17 graph-level scalar features."""
    if graph.number_of_nodes() == 0:
        return _empty_graph_features()

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    result = {
        'n_nodes': int(n_nodes),
        'n_edges': int(n_edges),
        'graph_density': float(nx.density(graph)),
        'n_weakly_connected': int(nx.number_weakly_connected_components(graph)),
        'n_strongly_connected': int(nx.number_strongly_connected_components(graph)),
    }

    home_candidates = [node for node, data in graph.nodes(data=True) if data.get('is_home')]
    home_node = home_candidates[0] if home_candidates else max(
        graph.nodes,
        key=lambda node: graph.nodes[node].get('total_duration_h', 0.0),
    )
    result['home_in_degree'] = int(graph.in_degree(home_node))
    result['home_out_degree'] = int(graph.out_degree(home_node))

    if n_nodes >= 3 and n_edges > 0:
        # Higher trip frequency means a lower movement cost.
        betweenness = nx.betweenness_centrality(graph, weight='cost', normalized=True)
        result['home_betweenness'] = float(betweenness.get(home_node, 0.0))
        result['mean_betweenness'] = float(np.mean(list(betweenness.values())))
    else:
        result['home_betweenness'] = 0.0
        result['mean_betweenness'] = 0.0

    if n_nodes >= 2 and n_edges > 0:
        pagerank = nx.pagerank(graph, weight='weight')
        result['home_pagerank'] = float(pagerank.get(home_node, 0.0))
        result['max_pagerank'] = float(max(pagerank.values()))
    else:
        result['home_pagerank'] = 1.0
        result['max_pagerank'] = 1.0

    edge_rows = list(graph.edges(data=True))
    if edge_rows:
        frequencies = np.array([d['weight'] for _, _, d in edge_rows], dtype=float)
        mean_distances = np.array([d['mean_dist_km'] for _, _, d in edge_rows], dtype=float)
        total_distances = np.array([d['total_dist_km'] for _, _, d in edge_rows], dtype=float)
        probabilities = frequencies / frequencies.sum()
        result.update({
            'mean_edge_weight': float(frequencies.mean()),
            'max_edge_weight': float(frequencies.max()),
            'edge_weight_entropy': _safe_entropy(probabilities),
            'mean_edge_dist_km': float(np.average(mean_distances, weights=frequencies)),
            'total_edge_dist_km': float(total_distances.sum()),
        })
    else:
        result.update({
            'mean_edge_weight': 0.0,
            'max_edge_weight': 0.0,
            'edge_weight_entropy': 0.0,
            'mean_edge_dist_km': 0.0,
            'total_edge_dist_km': 0.0,
        })

    reciprocity = nx.reciprocity(graph) if n_edges > 0 else 0.0
    result['reciprocity'] = float(reciprocity) if reciprocity is not None else 0.0
    return result


def _empty_graph_features() -> dict:
    return {
        'n_nodes': 0, 'n_edges': 0, 'graph_density': 0.0,
        'n_weakly_connected': 0, 'n_strongly_connected': 0,
        'home_in_degree': 0, 'home_out_degree': 0,
        'home_betweenness': 0.0, 'mean_betweenness': 0.0,
        'home_pagerank': 0.0, 'max_pagerank': 0.0,
        'mean_edge_weight': 0.0, 'max_edge_weight': 0.0,
        'edge_weight_entropy': 0.0, 'mean_edge_dist_km': 0.0,
        'total_edge_dist_km': 0.0, 'reciprocity': 0.0,
    }


def compute_graph_features(grp: pd.DataFrame) -> dict:
    """Build and summarise one user's multi-day graph."""
    return extract_graph_features(build_user_graph(grp))
# <<< end: multiday_directed_graph_features >>>


# =============================================================================
# <<< section: feature_functions >>>
# =============================================================================
# Feature-group functions
# =============================================================================

def compute_point_level_features(grp: pd.DataFrame) -> dict:
    """First-order spatial features and top-1/top-2 visit frequencies."""
    summary = _location_summary(grp)
    locs = grp['location_id'].astype(str)
    eccentricity, direction = compute_eccentricity_and_direction(summary)

    visit_counts = summary['visit_count'].sort_values(ascending=False).to_numpy(dtype=float)
    total_visits = max(float(visit_counts.sum()), 1.0)
    top1_frequency = float(visit_counts[0] / total_visits) if len(visit_counts) >= 1 else 0.0
    top2_frequency = float(visit_counts[1] / total_visits) if len(visit_counts) >= 2 else 0.0

    return {
        'stay_point_count': int(len(grp)),
        'unique_stay_locations': int(len(summary)),
        'stay_entropy': stay_point_entropy(locs),
        'stay_radius_of_gyration': radius_of_gyration_from_summary(summary),
        'stay_convex_hull_diameter': convex_hull_diameter(summary['lat'], summary['lon']),
        'stay_area_km2': convex_hull_area_km2(summary['lat'], summary['lon']),
        'stay_eccentricity': eccentricity,
        'stay_direction_deg': direction,
        'top1_visit_frequency': top1_frequency,
        'top2_visit_frequency': top2_frequency,
    }

def compute_line_level_features(grp: pd.DataFrame) -> dict:
    """Travel count, distance, and OD entropy from distinct transitions."""
    transitions = _build_transition_table(grp)
    if transitions.empty:
        return {
            'n_travels': 0,
            'total_travel_length_km': 0.0,
            'mean_travel_length_km': 0.0,
            'od_entropy': 0.0,
        }

    od_pairs = transitions['src'].astype(str) + '→' + transitions['dst'].astype(str)
    probabilities = od_pairs.value_counts(normalize=True).to_numpy(dtype=float)
    distances = transitions['distance_km'].to_numpy(dtype=float)
    return {
        'n_travels': int(len(transitions)),
        'total_travel_length_km': float(distances.sum()),
        'mean_travel_length_km': float(distances.mean()),
        'od_entropy': _safe_entropy(probabilities),
    }

def compute_temporal_features(grp: pd.DataFrame) -> dict:
    """
    Duration fragmentation and travel-departure rhythm.

    `time_fragmented` is the population standard deviation of stay durations in
    hours. Rhythm proportions use transition departure times in 3-hour bins.
    """
    g = _prepare_user_group(grp)
    duration_hours = g['_duration_s'].to_numpy(dtype=float) / 3600.0
    time_fragmented = float(np.std(duration_hours, ddof=0)) if len(duration_hours) else 0.0

    transitions = _build_transition_table(g)
    if transitions.empty:
        rhythm = np.zeros(8, dtype=float)
    else:
        departure = pd.to_datetime(transitions['departure_time'], errors='coerce')
        departure = departure.dropna()
        slots = (departure.dt.hour.to_numpy() // 3).clip(0, 7)
        counts = np.bincount(slots, minlength=8).astype(float)
        rhythm = counts / counts.sum() if counts.sum() > 0 else np.zeros(8, dtype=float)

    return {
        'time_fragmented': time_fragmented,
        'travel_rhythm_entropy': _safe_entropy(rhythm, base=2.0),
        'rhythm_morning': float(rhythm[2] + rhythm[3]),
        'rhythm_afternoon': float(rhythm[4] + rhythm[5]),
        'rhythm_evening': float(rhythm[6] + rhythm[7]),
    }

def compute_spacetime_features(grp: pd.DataFrame) -> dict:
    """Ratio between top-2-location RoG and full visit-frequency-weighted RoG."""
    summary = _location_summary(grp)
    full_rog = radius_of_gyration_from_summary(summary)
    top2_rog = k_radius_of_gyration(summary, k=2)
    return {'k_rog_ratio_2': float(top2_rog / full_rog) if full_rog > 0 else 0.0}

def _duration_weighted_centroid(rows: pd.DataFrame) -> tuple[float, float]:
    weights = rows['_duration_s'].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(rows), dtype=float)
    return (
        float(np.average(rows['lat'].to_numpy(dtype=float), weights=weights)),
        float(np.average(rows['lon'].to_numpy(dtype=float), weights=weights)),
    )

def _duration_weighted_category_distribution(
    rows: pd.DataFrame,
    category_col: str,
) -> pd.Series:
    weighted = rows.groupby(category_col)['_duration_s'].sum().astype(float)
    weighted = weighted[weighted > 0]
    total = float(weighted.sum())
    return weighted / total if total > 0 else pd.Series(dtype=float)

def compute_semantic_features(grp: pd.DataFrame) -> dict:
    """Duration-weighted purpose/land-use summaries and work-mobility features."""
    g = _prepare_user_group(grp)
    result: dict[str, float] = {}

    total_duration = max(float(g['_duration_s'].sum()), 1.0)
    purpose_duration = g.groupby('_purpose')['_duration_s'].sum().astype(float)
    purpose_prob = purpose_duration / max(float(purpose_duration.sum()), 1.0)
    for purpose in ['work', 'home', 'leisure']:
        result[f'purpose_{purpose}_ratio'] = float(purpose_prob.get(purpose, 0.0))
    result['purpose_entropy'] = _safe_entropy(purpose_prob.to_numpy(dtype=float))

    location_duration = g.groupby('location_id')['_duration_s'].sum().sort_values(ascending=False)
    top1_id = location_duration.index[0] if len(location_duration) >= 1 else None
    top2_id = location_duration.index[1] if len(location_duration) >= 2 else None
    top1_duration = float(location_duration.iloc[0]) if top1_id is not None else 0.0
    top2_duration = float(location_duration.iloc[1]) if top2_id is not None else 0.0

    result['top1_dur_ratio'] = top1_duration / total_duration
    result['top2_dur_ratio'] = top2_duration / total_duration
    result['top1_top2_dur_ratio'] = (
        float(top1_duration / top2_duration) if top2_duration > 0 else 0.0
    )
    if top1_id is not None:
        top1_rows = g[g['location_id'] == top1_id]
        top1_home_duration = float(
            top1_rows.loc[top1_rows['_purpose'] == 'home', '_duration_s'].sum()
        )
        result['top1_purpose_home'] = top1_home_duration / max(top1_duration, 1.0)
    else:
        result['top1_purpose_home'] = 0.0

    known_lu = g[g['_lu'] != 'unknown']
    lu_prob = _duration_weighted_category_distribution(known_lu, '_lu')
    result['landuse_residential_ratio'] = float(lu_prob.get('residential', 0.0))
    result['landuse_working_ratio'] = float(lu_prob.get('working', 0.0))
    result['landuse_mixed_ratio'] = float(lu_prob.get('mixed', 0.0))
    result['landuse_entropy'] = _safe_entropy(lu_prob.to_numpy(dtype=float))

    home_rows = g[g['_purpose'] == 'home']
    work_rows = g[g['_purpose'] == 'work']
    if not home_rows.empty and not work_rows.empty:
        home_lu = _duration_weighted_category_distribution(
            home_rows[home_rows['_lu'] != 'unknown'], '_lu'
        )
        work_lu = _duration_weighted_category_distribution(
            work_rows[work_rows['_lu'] != 'unknown'], '_lu'
        )
        home_vector = np.array([float(home_lu.get(cat, 0.0)) for cat in land_use_cats])
        work_vector = np.array([float(work_lu.get(cat, 0.0)) for cat in land_use_cats])
        denominator = np.linalg.norm(home_vector) * np.linalg.norm(work_vector)
        result['home_work_lu_contrast'] = (
            float(1.0 - np.dot(home_vector, work_vector) / denominator)
            if denominator > 0 else 0.0
        )

        home_lat, home_lon = _duration_weighted_centroid(home_rows)
        work_lat, work_lon = _duration_weighted_centroid(work_rows)
        result['commute_dist_km'] = float(haversine_km(
            home_lat, home_lon, work_lat, work_lon
        ))
    else:
        result['home_work_lu_contrast'] = 0.0
        result['commute_dist_km'] = 0.0

    transitions = _build_transition_table(g)
    total_transition_distance = float(transitions['distance_km'].sum()) if not transitions.empty else 0.0
    work_transitions = transitions[transitions['destination_purpose'] == 'work'] if not transitions.empty else transitions
    result['dist_per_work_trip'] = (
        float(work_transitions['distance_km'].mean()) if not work_transitions.empty else 0.0
    )
    result['work_travel_intensity'] = (
        float(work_transitions['distance_km'].sum() / total_transition_distance)
        if total_transition_distance > 0 else 0.0
    )

    if not work_transitions.empty:
        departure_hours = pd.to_datetime(
            work_transitions['departure_time'], errors='coerce'
        ).dt.hour
        peak = departure_hours.between(7, 9) | departure_hours.between(17, 19)
        result['work_peak_ratio'] = float(peak.mean())
    else:
        result['work_peak_ratio'] = 0.0

    observed_days = max(g[timestamp_col].dt.date.nunique(), 1)
    mean_daily_distance = total_transition_distance / observed_days
    mean_daily_work_hours = float(work_rows['_duration_s'].sum() / 3600.0) / observed_days
    result['work_hour_dist_product'] = mean_daily_work_hours * mean_daily_distance
    return result
# <<< end: feature_functions >>>

# =============================================================================
# <<< section: build_feature_matrix >>>
# =============================================================================

def build_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute every raw feature for every user without population-level fitting."""
    rows: list[dict] = []
    failed: list[dict] = []
    grouped = raw.groupby(user_col, sort=True)
    total_users = grouped.ngroups

    for index, (uid, grp) in enumerate(grouped, start=1):
        try:
            features = {user_col: str(uid)}
            features.update(compute_point_level_features(grp))
            features.update(compute_line_level_features(grp))
            features.update(compute_motif_features(grp))
            features.update(compute_temporal_features(grp))
            features.update(compute_spacetime_features(grp))
            features.update(compute_semantic_features(grp))
            features.update(compute_graph_features(grp))
            rows.append(features)
        except Exception as exc:
            failed.append({user_col: str(uid), 'error': repr(exc)})
            raise RuntimeError(f'Feature construction failed for user {uid}: {exc}') from exc

        if index % 50 == 0 or index == total_users:
            print(f'  processed {index:,}/{total_users:,} users')

    feature_df = pd.DataFrame(rows)
    numeric_cols = [c for c in feature_df.columns if c != user_col]
    feature_df[numeric_cols] = (
        feature_df[numeric_cols]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    print(
        f'  built: {feature_df.shape[0]:,} users × '
        f'{feature_df.shape[1] - 1:,} raw features'
    )
    return feature_df
# <<< end: build_feature_matrix >>>

# =============================================================================
# <<< section: preprocess >>>
# =============================================================================
# Descriptive diagnostics and leakage-safe sklearn preprocessing
# =============================================================================

def _plot_corr_heatmap(corr: pd.DataFrame, filename: str, title: str) -> None:
    """Save a descriptive full-population heatmap; never use it for CV selection."""
    if corr.empty:
        return
    n = len(corr)
    cell_size = max(0.35, min(0.7, 20 / max(n, 1)))
    fig_size = max(8, n * cell_size)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    labels = [feature_labels.get(feature, feature) for feature in corr.columns]
    mask = np.triu(np.ones(corr.shape, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='vlag',
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3 if n < 25 else 0,
        annot=(n <= 20),
        fmt='.2f',
        annot_kws={'size': 7},
        ax=ax,
        cbar_kws={'shrink': 0.6, 'label': 'Pearson r'},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=max(6, 9 - n // 10))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=max(6, 9 - n // 10))
    ax.set_title(title, fontweight='bold', fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'figures' / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_raw_diagnostics(feature_df: pd.DataFrame) -> None:
    """Save descriptive statistics only; no columns are transformed or removed."""
    feature_cols = [c for c in feature_df.columns if c != user_col]
    x = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    stats = x.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats.columns = ['mean', 'std', 'min', 'p25', 'median', 'p75', 'max']
    stats.index.name = 'feature'
    stats.to_csv(out_dir / 'feature_stats_raw.csv')

    skewness = x.apply(
        lambda column: float(scipy_skew(column.dropna(), bias=False))
        if column.dropna().nunique() > 1 else 0.0
    )
    pd.DataFrame({
        'feature': skewness.index,
        'population_skewness_descriptive_only': skewness.values,
    }).sort_values('population_skewness_descriptive_only', ascending=False).to_csv(
        out_dir / 'feature_skewness_descriptive.csv', index=False
    )

    corr = x.corr(method='pearson')
    corr.to_csv(out_dir / 'correlation_matrix_raw_descriptive.csv')
    _plot_corr_heatmap(
        corr,
        'correlation_heatmap_raw_descriptive.png',
        f'Descriptive Pearson correlation — all {len(feature_cols)} raw features',
    )

    motif_frequency_report(feature_df).to_csv(
        out_dir / 'motif_frequency_descriptive.csv', index=False
    )

def _ensure_dataframe(X, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = np.asarray(X)
    if columns is None:
        columns = [f'x{i}' for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)

class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Replace non-finite values and learn per-column medians from training data."""
    def fit(self, X, y=None):
        frame = _ensure_dataframe(X).replace([np.inf, -np.inf], np.nan)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.medians_ = frame.median(numeric_only=True).reindex(frame.columns).fillna(0.0)
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame.fillna(self.medians_)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

class RareMotifFilter(BaseEstimator, TransformerMixin):
    """Drop rare motif ratios using the TRAINING-fold population mean only."""
    def __init__(self, min_mean_pct: float = ML_MOTIF_MIN_FREQ_PCT):
        self.min_mean_pct = min_mean_pct

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        motif_cols = [
            c for c in frame.columns
            if c.startswith('motif_') and c.endswith('_ratio')
        ]
        self.dropped_features_ = [
            c for c in motif_cols
            if float(frame[c].mean()) * 100.0 < self.min_mean_pct
        ]
        self.kept_features_ = [c for c in frame.columns if c not in self.dropped_features_]
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        return frame.loc[:, self.kept_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_features_, dtype=object)


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """Scale-independent constant/near-zero-variance filter fitted on training data."""
    def __init__(
        self,
        freq_ratio_threshold: float = ML_NZV_FREQ_RATIO,
        unique_pct_threshold: float = ML_NZV_UNIQUE_PCT,
    ):
        self.freq_ratio_threshold = freq_ratio_threshold
        self.unique_pct_threshold = unique_pct_threshold

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        dropped: list[str] = []
        reasons: dict[str, str] = {}

        for column in frame.columns:
            counts = frame[column].value_counts(dropna=False)
            if len(counts) <= 1:
                dropped.append(column)
                reasons[column] = 'constant'
                continue
            unique_pct = 100.0 * len(counts) / max(len(frame), 1)
            freq_ratio = float(counts.iloc[0] / counts.iloc[1]) if counts.iloc[1] > 0 else np.inf
            if unique_pct <= self.unique_pct_threshold and freq_ratio >= self.freq_ratio_threshold:
                dropped.append(column)
                reasons[column] = (
                    f'near-zero variance: unique_pct={unique_pct:.3f}, '
                    f'freq_ratio={freq_ratio:.3f}'
                )

        self.dropped_features_ = dropped
        self.drop_reasons_ = reasons
        self.kept_features_ = [c for c in frame.columns if c not in dropped]
        if not self.kept_features_:
            raise ValueError('NearZeroVarianceFilter removed every feature.')
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        return frame.loc[:, self.kept_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_features_, dtype=object)

class FixedAliasDropper(BaseEstimator, TransformerMixin):
    """Remove deterministic aliases without looking at validation/test values."""
    def __init__(self, groups: list[list[str]] | None = None):
        self.groups = groups

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        groups = alias_groups if self.groups is None else self.groups
        drops: list[str] = []
        for group in groups:
            present = [c for c in group if c in frame.columns]
            drops.extend(present[1:])
        self.dropped_features_ = list(dict.fromkeys(drops))
        self.kept_features_ = [c for c in frame.columns if c not in self.dropped_features_]
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        return frame.loc[:, self.kept_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_features_, dtype=object)

class SkewedLog1pTransformer(BaseEstimator, TransformerMixin):
    """Learn non-negative right-skewed columns from training data and apply log1p."""
    def __init__(self, skew_threshold: float = ML_SKEW_THRESHOLD):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        selected: list[str] = []
        skewness: dict[str, float] = {}
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors='coerce').dropna()
            if values.empty or values.nunique() <= 1 or float(values.min()) < 0:
                skewness[column] = 0.0
                continue
            value = float(scipy_skew(values, bias=False))
            if not np.isfinite(value):
                value = 0.0
            skewness[column] = value
            if value > self.skew_threshold:
                selected.append(column)
        self.skewness_ = skewness
        self.log_features_ = selected
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        transformed = frame.copy()
        for column in self.log_features_:
            transformed[column] = np.log1p(transformed[column].clip(lower=0))
        return transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

class GreedyCorrelationFilter(BaseEstimator, TransformerMixin):
    """Greedy Pearson filter fitted solely on the training fold."""
    def __init__(self, threshold: float = ML_CORR_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        corr = frame.corr(method='pearson').abs().fillna(0.0)
        drops: list[str] = []
        registry: list[dict] = []

        while corr.shape[1] > 1:
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            max_value = float(upper.max().max())
            if not np.isfinite(max_value) or max_value < self.threshold:
                break
            col_j = upper.max().idxmax()
            col_i = upper[col_j].idxmax()
            mean_i = float(corr[col_i].drop(index=col_i).mean())
            mean_j = float(corr[col_j].drop(index=col_j).mean())
            drop = col_i if mean_i >= mean_j else col_j
            keep = col_j if drop == col_i else col_i
            drops.append(drop)
            registry.append({'dropped': drop, 'kept_instead': keep, 'abs_r': max_value})
            corr = corr.drop(index=drop, columns=drop)

        self.dropped_features_ = drops
        self.drop_registry_ = registry
        self.kept_features_ = [c for c in frame.columns if c not in drops]
        return self

    def transform(self, X):
        frame = _ensure_dataframe(X, list(self.feature_names_in_))
        return frame.loc[:, self.kept_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_features_, dtype=object)

def make_feature_preprocessor(
    scaling: str = 'standard',
    corr_threshold: float = ML_CORR_THRESHOLD,
    skew_threshold: float = ML_SKEW_THRESHOLD,
    motif_min_mean_pct: float = ML_MOTIF_MIN_FREQ_PCT,
) -> Pipeline:
    """
    Build a leakage-safe feature preprocessor.

    scaling:
        'standard' -> recommended for logistic regression and SVM
        'minmax'   -> useful for kNN or neural networks; train values map to [0,1]
        'none'     -> recommended for Random Forest and XGBoost
    """
    scaling = scaling.lower()
    if scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    elif scaling in {'none', 'passthrough'}:
        scaler = 'passthrough'
    else:
        raise ValueError("scaling must be 'standard', 'minmax', or 'none'.")

    return Pipeline([
        ('impute', DataFrameMedianImputer()),
        ('rare_motifs', RareMotifFilter(min_mean_pct=motif_min_mean_pct)),
        ('near_zero_variance', NearZeroVarianceFilter()),
        ('aliases', FixedAliasDropper()),
        ('log1p', SkewedLog1pTransformer(skew_threshold=skew_threshold)),
        ('correlation', GreedyCorrelationFilter(threshold=corr_threshold)),
        ('scaler', scaler),
    ])

def make_model_pipeline(estimator, scaling: str = 'standard', **preprocess_kwargs) -> Pipeline:
    """Combine leakage-safe preprocessing and an sklearn-compatible estimator."""
    return Pipeline([
        ('preprocess', make_feature_preprocessor(scaling=scaling, **preprocess_kwargs)),
        ('model', estimator),
    ])

def selected_feature_names(fitted_model_pipeline: Pipeline) -> list[str]:
    """Return feature names retained by a fitted model pipeline."""
    preprocess_pipeline = fitted_model_pipeline.named_steps['preprocess']
    names = preprocess_pipeline[:-1].get_feature_names_out()
    return [str(name) for name in names]
# <<< end: preprocess >>>

# =============================================================================
# <<< section: main >>>
# =============================================================================

def _validate_input(raw: pd.DataFrame) -> None:
    required = {user_col, timestamp_col, 'location_id', 'lat', 'lon'}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f'Missing required input columns: {missing}')

def _load_raw_trajectory() -> pd.DataFrame:
    if not trace_file.exists():
        raise FileNotFoundError(f'Input file not found: {trace_file}')

    raw = pd.read_csv(trace_file, dtype={user_col: str}, low_memory=False)
    _validate_input(raw)

    raw[user_col] = raw[user_col].astype(str).str.strip()
    raw['location_id'] = raw['location_id'].astype('string')
    raw[timestamp_col] = _normalise_datetime_series(raw[timestamp_col])
    if 'finished_at' in raw.columns:
        raw['finished_at'] = _normalise_datetime_series(raw['finished_at'])
    raw['lat'] = pd.to_numeric(raw['lat'], errors='coerce')
    raw['lon'] = pd.to_numeric(raw['lon'], errors='coerce')
    if 'length_km' in raw.columns:
        raw['length_km'] = pd.to_numeric(raw['length_km'], errors='coerce')

    raw = raw.dropna(subset=[user_col, timestamp_col, 'location_id', 'lat', 'lon'])
    raw = raw[raw['location_id'].astype(str).str.len() > 0]
    raw = raw.sort_values([user_col, timestamp_col]).reset_index(drop=True)

    if USE_4WEEK_FILTER:
        user_start = raw.groupby(user_col)[timestamp_col].transform('min')
        raw = raw[(raw[timestamp_col] - user_start) < pd.Timedelta(days=28)].copy()
    return raw

def _write_preprocessing_readme() -> None:
    text = """Leakage-safe model usage
========================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

feature_df = pd.read_csv('feature_matrix_raw.csv')
X = feature_df.drop(columns=['user_id'])
y = labels

model = make_model_pipeline(
    LogisticRegression(max_iter=5000, class_weight='balanced'),
    scaling='standard',
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
result = cross_validate(model, X, y, cv=cv, scoring=['balanced_accuracy', 'f1'])

Recommended scaling
-------------------
Logistic regression / SVM: standard
kNN / neural network:      standard or minmax
Random Forest / XGBoost:   none

Every data-dependent decision is fitted separately inside each training fold.
Do not preprocess the complete feature matrix before cross-validation.
"""
    (out_dir / 'ML_PREPROCESSING_README.txt').write_text(text, encoding='utf-8')

def main() -> None:
    window_label = 'first 4 weeks' if USE_4WEEK_FILTER else 'full trajectory'
    print('=' * 72)
    print('Mobility Feature Builder — raw features + leakage-safe ML utilities')
    print(f'  Data window : {window_label}')
    print(f'  Output dir  : {out_dir}')
    print('=' * 72)

    raw_cache = out_dir / 'feature_matrix_raw.csv'
    window_tag = '4weeks' if USE_4WEEK_FILTER else 'full'
    cache_version = f'v2_raw_only_geometry_graph_semantic_{window_tag}'
    version_file = out_dir / 'feature_matrix_raw.version'

    cache_valid = (
        raw_cache.exists()
        and version_file.exists()
        and version_file.read_text(encoding='utf-8').strip() == cache_version
    )

    if cache_valid:
        print(f'  Loading cached raw matrix ({cache_version})')
        feature_df = pd.read_csv(raw_cache, dtype={user_col: str})
    else:
        print(f'  Reading {trace_file}')
        raw = _load_raw_trajectory()
        print(f'  Rows: {len(raw):,}   users: {raw[user_col].nunique():,}')
        feature_df = build_feature_matrix(raw)
        feature_df.to_csv(raw_cache, index=False)
        version_file.write_text(cache_version, encoding='utf-8')
        print(f'  Saved raw matrix → {raw_cache}')

    save_raw_diagnostics(feature_df)
    _write_preprocessing_readme()

    metadata = {
        'cache_version': cache_version,
        'data_window': window_label,
        'n_users': int(len(feature_df)),
        'n_raw_features': int(feature_df.shape[1] - 1),
        'normalisation_applied': False,
        'feature_selection_applied': False,
        'note': 'Fit make_model_pipeline(...) inside each CV training fold.',
    }
    (out_dir / 'feature_build_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    print('\nNo global normalisation or correlation filtering was applied.')
    print('Use make_model_pipeline(...) during machine learning.')
    print('=' * 72)
    print('Done')
    print('=' * 72)

if __name__ == '__main__':
    main()
# <<< end: main >>>
