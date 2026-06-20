"""
ParkWatch: Parking-Induced Congestion Hotspot & Impact Intelligence
=====================================================================
Full data pipeline: load, clean, score, and aggregate violation data
into a Congestion Impact Score per geo-location.
"""

import pandas as pd
import numpy as np
import ast
import json

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: Load & Clean
# ────────────────────────────────────────────────────────────────────────────

def load_and_clean(path):
    df = pd.read_csv(path)

    # Parse datetime
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime', 'latitude', 'longitude'])

    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.day_name()
    df['date'] = df['created_datetime'].dt.date
    df['is_weekend'] = df['created_datetime'].dt.dayofweek.isin([5, 6])

    # Parse violation_type list column
    df['violations_list'] = df['violation_type'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    return df


# ────────────────────────────────────────────────────────────────────────────
# STEP 2: Capacity-Loss Model (Path 3 — traffic engineering based)
# ────────────────────────────────────────────────────────────────────────────
# Rationale: Highway Capacity Manual-style research shows that a parked
# vehicle obstructing a lane on an urban road reduces the effective capacity
# of that road segment. The reduction is more severe on narrower /
# fewer-lane roads (a 2-lane road loses ~50% of its capacity if one lane is
# blocked, vs. a 4+ lane road which can partially reroute traffic into
# remaining lanes, ~15-20% loss).
#
# Since we don't have a direct lane-count field, we use a proxy: violation
# type + location context (main road / road crossing / footpath) as an
# indicator of how disruptive the obstruction is, and combine it with road
# classification implied by violation phrasing. This is an estimate, and is
# explicitly documented as such in the concept note.

CAPACITY_LOSS_PCT = {
    'PARKING IN A MAIN ROAD': 0.45,             # arterial road, single lane blocked = major loss
    'PARKING NEAR ROAD CROSSING': 0.40,         # blocks crossing visibility/flow
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 0.35,
    'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 0.35,
    'DOUBLE PARKING': 0.35,                     # blocks two effective lanes
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 0.30,  # narrows road from both sides
    'WRONG PARKING': 0.20,
    'NO PARKING': 0.15,
    'PARKING ON FOOTPATH': 0.05,                # doesn't block vehicle lane directly, pushes pedestrians onto road
    'PARKING OTHER THAN BUS STOP': 0.15,
}
DEFAULT_CAPACITY_LOSS = 0.10  # fallback for non-parking / minor violations

def capacity_loss_score(violations):
    """Take the MAX capacity loss among violations in a single record
    (the worst single obstruction governs the road's bottleneck, not the sum)."""
    losses = [CAPACITY_LOSS_PCT.get(v, DEFAULT_CAPACITY_LOSS) for v in violations]
    return max(losses) if losses else DEFAULT_CAPACITY_LOSS


# ────────────────────────────────────────────────────────────────────────────
# STEP 3: Severity Weight (for prioritization layer, Path 1)
# ────────────────────────────────────────────────────────────────────────────
SEVERITY_WEIGHT = {
    'PARKING IN A MAIN ROAD': 3,
    'PARKING NEAR ROAD CROSSING': 3,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 3,
    'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 3,
    'DOUBLE PARKING': 2,
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 2,
    'WRONG PARKING': 2,
    'NO PARKING': 1,
    'PARKING ON FOOTPATH': 1,
    'PARKING OTHER THAN BUS STOP': 1,
}
def severity_score(violations):
    return sum(SEVERITY_WEIGHT.get(v, 1) for v in violations) if violations else 1


# ────────────────────────────────────────────────────────────────────────────
# STEP 4: Geo-cluster assignment (handles the 49.5% "No Junction" gap AND
# fixes the junction-splitting problem from naive grid rounding)
# ────────────────────────────────────────────────────────────────────────────
# PROBLEM WITH THE OLD APPROACH: rounding lat/lon to a fixed grid (e.g. 3
# decimal places, ~111m cells) chops up large, sprawling junctions into many
# separate "hotspots". Validation showed KR Market Junction alone was split
# into 25 different grid cells spanning ~800m, making it look like 25 smaller
# problems instead of 1 big one, and inflating the apparent hotspot count.
#
# FIX: Use DBSCAN, a density-based spatial clustering algorithm, directly on
# lat/lon. It groups points that are close together into one cluster
# regardless of grid boundaries, and doesn't force a fixed cell size -- a
# sprawling junction naturally forms one cluster, while two genuinely
# distinct nearby streets stay separate if there's a gap between them.
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

def assign_geo_clusters(df, eps_meters=120, min_samples=3, max_bucket_size=4000, bucket_size_deg=0.01):
    """
    DBSCAN on lat/lon directly groups points into clusters regardless of grid
    boundaries -- fixing the old grid-rounding approach where one sprawling
    junction got split into dozens of cells.

    MEMORY CONSTRAINT: testing showed DBSCAN's ball_tree + haversine metric
    blows up in memory (>3.5GB) when a single dense region has 10,000+
    points within eps of each other -- this dataset has several such regions
    (one junction alone has 24,000+ tightly packed violations). Running
    DBSCAN on the full 298k points at once is not memory-safe here.

    FIX: pre-bucket points into overlapping coarse grid cells (with a buffer
    equal to eps so true clusters spanning a bucket boundary aren't split),
    run DBSCAN independently within each bucket, then stitch bucket-local
    cluster labels into globally unique cluster IDs using vectorized numpy
    array writes (not row-by-row .loc, which doesn't scale).
    """
    eps_deg = eps_meters / 111320
    buffer_deg = eps_deg * 1.5
    eps_rad = eps_meters / 6371000

    df = df.reset_index(drop=True)
    n = len(df)
    lat = df['latitude'].values
    lon = df['longitude'].values

    # Output arrays, filled in per-bucket
    cluster_id = np.full(n, -1, dtype=np.int64)   # -1 = unassigned/noise
    global_label = np.empty(n, dtype=object)

    bucket_lat = np.round(lat / bucket_size_deg).astype(int)
    bucket_lon = np.round(lon / bucket_size_deg).astype(int)
    bucket_keys = np.unique(np.stack([bucket_lat, bucket_lon], axis=1), axis=0)

    cluster_counter = 0

    for b_lat, b_lon in bucket_keys:
        lat_center = b_lat * bucket_size_deg
        lon_center = b_lon * bucket_size_deg

        in_buffer = (
            (lat >= lat_center - bucket_size_deg/2 - buffer_deg) &
            (lat <  lat_center + bucket_size_deg/2 + buffer_deg) &
            (lon >= lon_center - bucket_size_deg/2 - buffer_deg) &
            (lon <  lon_center + bucket_size_deg/2 + buffer_deg)
        )
        idx_buffer = np.where(in_buffer)[0]
        if len(idx_buffer) == 0:
            continue

        in_core = (
            (lat[idx_buffer] >= lat_center - bucket_size_deg/2) &
            (lat[idx_buffer] <  lat_center + bucket_size_deg/2) &
            (lon[idx_buffer] >= lon_center - bucket_size_deg/2) &
            (lon[idx_buffer] <  lon_center + bucket_size_deg/2)
        )
        # Skip buckets with no core points (pure buffer overlap from a neighbor)
        if not in_core.any():
            continue

        coords_rad = np.radians(np.stack([lat[idx_buffer], lon[idx_buffer]], axis=1)).astype(np.float32)

        if len(idx_buffer) > max_bucket_size:
            # Dense region: fit DBSCAN on a random subsample, then assign the
            # rest to the nearest sampled point's cluster if within eps.
            # This is a standard scalable-DBSCAN technique that trades a
            # small amount of boundary precision for memory safety.
            rng = np.random.default_rng(42)
            sample_pos = rng.choice(len(idx_buffer), size=max_bucket_size, replace=False)
            fit_coords = coords_rad[sample_pos]
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree',
                        metric='haversine', n_jobs=1)
            fit_labels = db.fit_predict(fit_coords)

            local_labels = np.full(len(idx_buffer), -1)
            local_labels[sample_pos] = fit_labels

            remaining_pos = np.setdiff1d(np.arange(len(idx_buffer)), sample_pos)
            if len(remaining_pos) > 0:
                tree = BallTree(fit_coords, metric='haversine')
                dist, ind = tree.query(coords_rad[remaining_pos], k=1)
                within = dist[:, 0] <= eps_rad
                local_labels[remaining_pos[within]] = fit_labels[ind[within, 0]]
        else:
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree',
                        metric='haversine', n_jobs=1)
            local_labels = db.fit_predict(coords_rad)

        # Commit only the core points (avoids double-assignment from overlap)
        core_global_idx = idx_buffer[in_core]
        core_local_labels = local_labels[in_core]

        has_cluster = core_local_labels != -1
        if has_cluster.any():
            unique_local = np.unique(core_local_labels[has_cluster])
            label_map = {loc_lbl: f'cluster_{cluster_counter + i}' for i, loc_lbl in enumerate(unique_local)}
            cluster_counter += len(unique_local)
            mapped = np.array([label_map[l] for l in core_local_labels[has_cluster]])
            global_label[core_global_idx[has_cluster]] = mapped

        noise_idx = core_global_idx[~has_cluster]
        if len(noise_idx) > 0:
            noise_labels = [f'noise_{round(lat[i],4)}_{round(lon[i],4)}' for i in noise_idx]
            global_label[noise_idx] = noise_labels

    # Safety net: any point somehow not covered by a bucket (shouldn't happen,
    # but guards against edge-case float boundary gaps) falls back to its own
    # rounded-coordinate singleton.
    uncovered = pd.isna(pd.Series(global_label))
    if uncovered.any():
        idxs = np.where(uncovered)[0]
        global_label[idxs] = [f'noise_{round(lat[i],4)}_{round(lon[i],4)}' for i in idxs]

    df['geo_cell'] = global_label
    return df


# ────────────────────────────────────────────────────────────────────────────
# STEP 5: Build per-record scores, then aggregate to geo-cell level
# ────────────────────────────────────────────────────────────────────────────
def build_scores(df):
    df['capacity_loss'] = df['violations_list'].apply(capacity_loss_score)
    df['severity'] = df['violations_list'].apply(severity_score)
    return df


def peak_concentration(hours):
    """% of violations falling in the busiest 3 hours for this location."""
    counts = hours.value_counts()
    if len(counts) == 0:
        return 0.0
    return counts.nlargest(3).sum() / len(hours)


def aggregate_geo_cells(df):
    grouped = df.groupby('geo_cell')

    agg = grouped.agg(
        violation_count=('id', 'count'),
        avg_capacity_loss=('capacity_loss', 'mean'),
        max_capacity_loss=('capacity_loss', 'max'),
        avg_severity=('severity', 'mean'),
        total_severity=('severity', 'sum'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
        unique_vehicles=('vehicle_number', 'nunique'),
        junction_name=('junction_name', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        police_station=('police_station', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        location=('location', lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else 'Unknown'),
    ).reset_index()

    peak_conc = grouped['hour'].apply(peak_concentration).rename('peak_hour_concentration')
    agg = agg.merge(peak_conc, on='geo_cell')

    weekend_ratio = grouped['is_weekend'].mean().rename('weekend_ratio')
    agg = agg.merge(weekend_ratio, on='geo_cell')

    # Most common hour (mode) for "worst time" reporting
    mode_hour = grouped['hour'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else -1).rename('peak_hour')
    agg = agg.merge(mode_hour, on='geo_cell')

    # Repeat-vehicle ratio: what fraction of violations come from vehicles
    # that were caught more than once at this location. This is a genuinely
    # independent signal from capacity_loss/severity (which describe HOW BAD
    # a violation type is) -- it instead tells us whether the same offenders
    # keep coming back, which indicates an entrenched, ongoing problem rather
    # than one-off incidents, and is useful for targeted enforcement (repeat
    # offenders are a natural first target).
    def repeat_vehicle_ratio(vehicle_numbers):
        counts = vehicle_numbers.value_counts()
        repeat_records = counts[counts > 1].sum()
        return repeat_records / len(vehicle_numbers) if len(vehicle_numbers) > 0 else 0.0

    repeat_ratio = grouped['vehicle_number'].apply(repeat_vehicle_ratio).rename('repeat_vehicle_ratio')
    agg = agg.merge(repeat_ratio, on='geo_cell')

    # Build a clean display_name: use junction_name when it's a real named
    # junction, otherwise fall back to a shortened version of the street
    # address (location field) so every hotspot has a human-readable label
    # for the dashboard/report instead of a blank "No Junction".
    def make_display_name(row):
        if row['junction_name'] and row['junction_name'] != 'No Junction' and row['junction_name'] != 'Unknown':
            return row['junction_name']
        loc = row['location']
        if isinstance(loc, str) and loc != 'Unknown':
            # location strings look like "MBT Road, Devasandra Junction, KR Puram, Bengaluru, Karnataka. Pin-560036 (India)"
            # keep the first 2-3 comma-separated parts, which is the specific street/area, drop city/state/pin boilerplate
            parts = [p.strip() for p in loc.split(',')]
            short = ', '.join(parts[:3])
            return short
        return f"Unnamed location near {row['police_station']}"

    agg['display_name'] = agg.apply(make_display_name, axis=1)

    return agg


# ────────────────────────────────────────────────────────────────────────────
# STEP 6: Congestion Impact Score (combine Path 1 + Path 3)
# ────────────────────────────────────────────────────────────────────────────
def compute_impact_score(agg):
    # Normalize each component to 0-1 before combining
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Violation count is heavily right-skewed (a few cells have 1000s of
    # violations, most have a handful) -- use log scale so the ranking
    # reflects "more violations = more important" without a few extreme
    # outlier junctions collapsing every other cell's normalized value near 0.
    norm_violation_count = normalize(np.log1p(agg['violation_count']))
    norm_capacity_loss = normalize(agg['avg_capacity_loss'])
    norm_repeat_ratio = normalize(agg['repeat_vehicle_ratio'])

    # NOTE on avg_severity: validation showed avg_severity and
    # avg_capacity_loss are 92.6% correlated -- they're measuring nearly the
    # same underlying thing (how disruptive the violation types are), so
    # including both in the score double-counts that signal without adding
    # real information. avg_severity is kept in the output as a descriptive
    # stat but excluded from scoring. In its place, repeat_vehicle_ratio adds
    # a genuinely independent dimension: whether the same offenders return
    # repeatedly, indicating an entrenched problem location.
    #
    # NOTE on peak_hour_concentration: this metric is only meaningful at
    # higher sample sizes (a cell with 1-2 violations trivially shows 100%
    # concentration). Kept as a descriptive stat, excluded from scoring.

    # Weights: violation density remains the primary, most statistically
    # reliable signal. Capacity loss adds the engineering-based "how
    # disruptive" layer. Repeat-vehicle ratio adds the "is this an
    # entrenched, ongoing problem" layer.
    agg['congestion_impact_score'] = (
        0.50 * norm_violation_count +
        0.30 * norm_capacity_loss +
        0.20 * norm_repeat_ratio
    ) * 100  # scale to 0-100 for readability

    return agg.sort_values('congestion_impact_score', ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    df = load_and_clean('data/jan to may police violation_anonymized791b166.csv')
    print(f"Loaded {len(df):,} clean rows")

    print("Assigning geo-clusters (DBSCAN)...")
    df = assign_geo_clusters(df)
    print(f"Found {df['geo_cell'].nunique():,} distinct geo-clusters")

    print("Computing per-record scores...")
    df = build_scores(df)

    print("Aggregating to geo-cell level...")
    agg = aggregate_geo_cells(df)

    print("Computing Congestion Impact Score...")
    agg = compute_impact_score(agg)

    # Filter out low-volume cells for the main hotspot list -- testing showed
    # cells with very few violations (under ~20) produce statistically
    # unreliable rankings (e.g. a single severe violation type can spike
    # their average capacity-loss score despite not being a real recurring
    # problem location). They're kept in the full geo_cell_scores.csv for
    # completeness/dashboard filtering, but excluded from the headline
    # "priority enforcement" hotspot list.
    hotspots = agg[agg['violation_count'] >= 20].copy()

    print(f"\nTotal geo-cells: {len(agg):,}")
    print(f"Hotspots (>=20 violations): {len(hotspots):,}")
    print("\nTop 15 hotspots by Congestion Impact Score:")
    print(hotspots[['display_name', 'police_station', 'violation_count',
                     'avg_capacity_loss', 'peak_hour', 'congestion_impact_score']].head(15).to_string())

    # Save outputs for dashboard
    agg.to_csv('geo_cell_scores.csv', index=False)
    df.to_csv('cleaned_violations.csv', index=False)

    print("\nSaved geo_cell_scores.csv and cleaned_violations.csv")