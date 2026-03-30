# ---------------------------------------------------------------------------
# Step 2: POI context lookup
# POI context lookup: given staypoints + POI GeoDataFrame,
# find the k nearest POIs for each location and format as prompt text.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path


# --------------------------------------
# Load POI frame (handles CRS)
# --------------------------------------
def load_poi_frame(path: Path) -> gpd.GeoDataFrame:
    """
    Load POI GeoPackage and ensure it is in from WGS84 to EPSG:2056 (Swiss LV95).
    Handles missing / wrong CRS
    """
    poi = gpd.read_file(path)

    if poi.crs is not None and poi.crs.to_epsg() == 2056:
        return poi

    if poi.crs is not None and poi.crs.to_epsg() == 4326:
        return poi.to_crs(2056)

    # Infer from coordinate magnitude
    x_med = poi.geometry.x.median()
    if 2_000_000 < x_med < 3_000_000:
        # Looks like LV95
        return poi.set_crs(epsg=2056)
    if -180 < x_med < 180:
        # Looks like WGS84
        return poi.set_crs(epsg=4326).to_crs(2056)

    # Fallback
    return poi.set_crs(epsg=2056, allow_override=True)

# --------------------------------------
# compute compass direction from dx, dy
# --------------------------------------
def _bearing_to_direction(dx: float, dy: float) -> str:
    angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dirs = ["North", "North-East", "East", "South-East",
            "South", "South-West", "West", "North-West"]
    return dirs[int((angle + 22.5) / 45) % 8]

# --------------------------------------
# Text cleaning
# --------------------------------------
def _clean(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip()
    return None if s.lower() in {"none", "nan", "-", ""} else s

# --------------------------------------
# Core KDTree and join logic to find nearby POIs for each location_id in sp
# --------------------------------------
def get_poi_context(
    sp: pd.DataFrame,
    poi: gpd.GeoDataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    For each unique location in sp, find the top_k nearest POIs.
    Parameters
    ----------
    sp  : DataFrame with columns [location_id, lon, lat]
    poi : GeoDataFrame in EPSG:2056 with columns [name, category]
    top_k : number of nearest POIs per location

    Returns
    -------
    DataFrame with columns:
        [location_id, name, category, addr_poi_dist_km, direction]
    """
    # Project staypoints to LV95
    loc = gpd.GeoDataFrame(
        sp[["location_id", "lon", "lat"]].drop_duplicates("location_id").copy(),
        geometry=gpd.points_from_xy(
            sp.drop_duplicates("location_id")["lon"].astype(float),
            sp.drop_duplicates("location_id")["lat"].astype(float),
        ),
        crs="EPSG:4326",
    ).to_crs(2056)

    poi_proj = poi.copy()
    if poi_proj.crs is None or poi_proj.crs.to_epsg() != 2056:
        poi_proj = poi_proj.set_crs(epsg=2056, allow_override=True)

    # KDTree
    poi_xy = np.c_[poi_proj.geometry.x.values, poi_proj.geometry.y.values]
    loc_xy = np.c_[loc.geometry.x.values,      loc.geometry.y.values]
    tree = cKDTree(poi_xy)
    dists, idxs = tree.query(loc_xy, k=top_k * 3)

    rows = []
    for i, (ds, js) in enumerate(zip(dists, idxs)):
        for d, j in zip(ds, js):
            rows.append({
                "location_id":     str(loc.iloc[i]["location_id"]),
                "name":            poi_proj.iloc[j]["name"],
                "category":        poi_proj.iloc[j]["category"],
                "addr_poi_dist_m": d,
                "dx": poi_proj.geometry.iloc[j].x - loc.geometry.iloc[i].x,
                "dy": poi_proj.geometry.iloc[j].y - loc.geometry.iloc[i].y,
            })

    joined = pd.DataFrame(rows)

    # Filter useless categories
    joined = joined[
        joined["category"].notna() &
        (~joined["category"].str.lower().isin(["unknown", "others"]))
    ]

    joined["addr_poi_dist_km"] = (joined["addr_poi_dist_m"] / 1000).round(3)
    joined["direction"] = [
        _bearing_to_direction(dx, dy)
        for dx, dy in zip(joined["dx"], joined["dy"])
    ]

    return (
        joined
        .sort_values("addr_poi_dist_m")
        .groupby("location_id", group_keys=False)
        .head(top_k)
        [["location_id", "name", "category", "addr_poi_dist_km", "direction"]]
    )

# --------------------------------------
# Format as prompt text
# --------------------------------------
def format_poi_text(df: pd.DataFrame) -> str:
    """
    Convert a per-location POI table row into a compact string.
    e.g. '0.012km NE Shopping Coop; 0.034km S Transportation Bus Stop'
    """
    out = []
    for _, r in df.iterrows():
        name     = _clean(r.get("name"))
        category = _clean(r.get("category"))
        if name is None and category is None:
            continue
        label = " ".join(filter(None, [category, name]))
        out.append(f"{r['addr_poi_dist_km']}km {r['direction']} {label}")
    return "; ".join(out)

# --------------------------------------
# Attach nearby_places column to sp dataframe
# --------------------------------------
def attach_poi_context(
    sp: pd.DataFrame,
    poi: gpd.GeoDataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Add a 'nearby_places' column to sp with formatted POI text per location.
    Parameters

    sp  : DataFrame with [location_id, lon, lat]
    poi : GeoDataFrame (will be loaded/projected if needed)

    Returns
    sp with new column 'nearby_places'
    """
    sp = sp.copy()
    sp["location_id"] = sp["location_id"].astype(str)

    poi_ctx = (
        get_poi_context(sp, poi, top_k=top_k)
        .groupby("location_id", group_keys=False)
        .apply(format_poi_text, include_groups=False)
        .reset_index(name="nearby_places")
    )
    poi_ctx["location_id"] = poi_ctx["location_id"].astype(str)

    sp = sp.merge(poi_ctx, on="location_id", how="left")
    filled = sp["nearby_places"].notna().sum()
    print(f"nearby_places: {filled}/{len(sp)} rows filled")
    return sp


# --------------------------------------
# CLI entry point: test POI loading
# --------------------------------------
if __name__ == "__main__":
    import sys
    poi_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/data/baliu/python_code/data/version2/data/final_pois_nob.gpkg")

    print(f"Loading POI from {poi_path}")
    poi = load_poi_frame(poi_path)
    print(f"POI shape: {poi.shape}")
    print(f"CRS: {poi.crs}")
    print(f"Category counts:\n{poi['category'].value_counts()}")
    print(f"X range: {poi.geometry.x.min():.0f} – {poi.geometry.x.max():.0f}")
    print(f"Y range: {poi.geometry.y.min():.0f} – {poi.geometry.y.max():.0f}")
