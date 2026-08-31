"""
Geocontext Enrichment 
=============================================================================
Input  : sp_enriched.csv  (coordinates + mode + purpose + sociodemographics)
Adds   :
  1. Reverse geocoding  - address, neighbourhood, city  (Nominatim cache)
  2. POI context        - nearby place names + category counts within radius
Output : sp_final_llms.csv    — one row per staypoint, ready for LLM inference

Each row will contain everything the LLMs needs:
  Who    age, gender, income, education
  Where  geometry, address, neighbourhood, city
  What   act_imputed_purpose, land-use zone (act_CH_BEZ_D)
  How    mode, length, duration
  Around nearby_places (top POI names), poi_counts per category
=============================================================================
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import sys
import os

# ── Add notebook_utils to path (same location as your notebooks) ─────────────
notebook_dir = Path('/data/baliu/python_code')
sys.path.insert(0, str(notebook_dir))

#from notebook_utils.geocontext import load_poi_frame, get_poi_context_fast

# ============================================================================
# Config — adjust paths and parameters
# ============================================================================
data_dir      = Path('/data/baliu/thesis/00data')
sp_enriched  = data_dir / 'sp_enriched.csv'
poi_path     = Path('/data/baliu/python_code/data/version2/data/final_pois_nob.gpkg')
nominatim_cache = Path('/data/baliu/python_code/data/version2/data/nominatim_cache.gpkg')

out_file     = data_dir / 'sp_final_llms.csv'

poi_radius_m  = 300    # metres — search radius for nearby POIs
poi_top_n    = 10     # max number of POI names to include in text field

# POI categories from your assign_category() function
POI_Categories = [
    'Shopping', 'Entertainment', 'Residential',
    'Transportation', 'Services', 'Schools', 'Civic', 'Others',
]

# ============================================================================
# 1.  Load sp_enriched and convert to Geodataframe
# ============================================================================

def load_sp(path: Path) -> gpd.GeoDataFrame:
    print("=" * 60)
    print("Loading sp_enriched")
    print("=" * 60)

    df = pd.read_csv(path)
    df['user_id'] = df['user_id'].astype(str).str.strip()
    print(f"  Rows    : {len(df):,}")
    print(f"  Users   : {df['user_id'].nunique():,}")
    print(f"  Columns : {list(df.columns)}")

    # Parse geometry column  "Point (lon lat)"
    def parse_point(s):
        try:
            s = str(s).strip()
            # handle both "Point (x y)" and "Point(x y)"
            coords = s.replace('Point', '').replace('(', '').replace(')', '').strip()
            lon, lat = map(float, coords.split())
            return Point(lon, lat)
        except Exception:
            return None

    df['geometry'] = df['geometry'].apply(parse_point)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

    # Drop rows with no geometry
    missing_geom = gdf['geometry'].isna().sum()
    if missing_geom:
        print(f" Dropped {missing_geom} rows with unparseable geometry")
        gdf = gdf[gdf['geometry'].notna()].reset_index(drop=True)

    print(f" Geodataframe ready  ({len(gdf):,} rows, EPSG:4326)")
    return gdf

# ============================================================================
# 2.  Reverse geocoding via nominatim cache
# ============================================================================

def add_reverse_geocoding(gdf: gpd.GeoDataFrame,
                           cache_path: Path) -> gpd.GeoDataFrame:
    print("\n" + "=" * 60)
    print("Step 1 — Reverse geocoding  (nominatim cache)")
    print("=" * 60)

    if not cache_path.exists():
        print(f" Cache not found: {cache_path}")
        print(f" Skipping reverse geocoding — address columns will be NaN")
        gdf['address']       = np.nan
        gdf['neighbourhood'] = np.nan
        gdf['city']          = np.nan
        gdf['postcode']      = np.nan
        return gdf

    cache = gpd.read_file(cache_path).to_crs('EPSG:4326')
    print(f"  Cache loaded: {len(cache):,} cached locations")
    print(f"  Cache columns: {list(cache.columns)}")

    # Spatial join: for each staypoint find the nearest cached address
    # Use nearest join (requires geopandas >= 0.10)
    gdf_proj   = gdf.to_crs('EPSG:3857')
    cache_proj = cache.to_crs('EPSG:3857')

    joined = gpd.sjoin_nearest(
        gdf_proj,
        cache_proj[['geometry'] + [c for c in cache_proj.columns
                                   if c not in ('geometry', 'index_right')]],
        how='left',
        distance_col='nominatim_dist_m',
    )
    joined = joined.to_crs('EPSG:4326')

    # Rename common Nominatim fields if they exist
    rename_map = {
        'display_name' : 'address',
        'road'         : 'street',
        'suburb'       : 'neighbourhood',
        'city'         : 'city',
        'town'         : 'town',
        'postcode'     : 'postcode',
        'country'      : 'country',
    }
    for old, new in rename_map.items():
        if old in joined.columns and new not in joined.columns:
            joined = joined.rename(columns={old: new})

    # Keep only one row per original index (nearest join can duplicate)
    joined = joined[~joined.index.duplicated(keep='first')]

    matched = joined['address'].notna().sum() if 'address' in joined.columns else 0
    print(f" Matched : {matched:,} / {len(gdf):,} rows with address")

    return joined.reset_index(drop=True)

# ============================================================================
# 3.  POI context  (radius-based count + top names)
# ============================================================================
def add_poi_context(gdf: gpd.GeoDataFrame,
                    poi_path: Path,
                    radius_m: int = poi_radius_m,
                    top_n: int = poi_top_n) -> gpd.GeoDataFrame:
    print("\n" + "=" * 60)
    print(f"Step 2 — POI context  (radius = {radius_m} m)")
    print("=" * 60)

    if not poi_path.exists():
        print(f" POI file not found: {poi_path}")
        print(f" Skipping POI enrichment")
        for cat in POI_Categories:
            gdf[f'poi_{cat.lower()}'] = 0
        gdf['nearby_places'] = ''
        return gdf

    # Load POIs using pois (returns EPSG:3857)
    print(f"  Loading POIs from {poi_path.name} …")
    pois = load_poi_frame(poi_path, epsg=3857)
    print(f"  POIs loaded: {len(pois):,}")

    # Project staypoints to EPSG:3857 for metric distance
    gdf_proj = gdf.to_crs('EPSG:3857').copy()
    gdf_proj = gdf_proj.reset_index(drop=True)

    poi_rows = []
    print(f"  Processing {len(gdf_proj):,} staypoints …")

    for i, row in gdf_proj.iterrows():
        if i % 5000 == 0:
            print(f"    {i:,} / {len(gdf_proj):,}")

        ctx = get_poi_context_fast(
            point    = row['geometry'],
            pois_gdf = pois,
            radius_m = radius_m,
            top_n    = top_n,
        )
        poi_rows.append(ctx)

    poi_df = pd.DataFrame(poi_rows, index=gdf_proj.index)

    # Expected output from get_poi_context_fast:
    #   nearby_places : comma separated POI names
    #   poi_shopping, poi_entertainment, … : counts per category
    # Rename to ensure consistent column names
    rename = {}
    for cat in POI_Categories:
        for candidate in [cat, cat.lower(), f'poi_{cat.lower()}',
                          f'count_{cat.lower()}']:
            if candidate in poi_df.columns:
                rename[candidate] = f'poi_{cat.lower()}'
                break

    poi_df = poi_df.rename(columns=rename)

    # Fill any missing category columns with 0
    for cat in POI_Categories:
        col = f'poi_{cat.lower()}'
        if col not in poi_df.columns:
            poi_df[col] = 0

    gdf = gdf.join(poi_df, how='left')

    # Summary
    if 'nearby_places' in gdf.columns:
        has_pois = gdf['nearby_places'].notna() & (gdf['nearby_places'] != '')
        print(f"  Staypoints with ≥1 nearby POI: {has_pois.sum():,} / {len(gdf):,}")

    return gdf

# ============================================================================
# 4.  Build LLM ready text summary column  — one row per staypoint with all context for LLM input
# ============================================================================
def build_llm_context(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single 'llm_context' column that summarises all info
    as a short natural-language paragraph for the LLM prompt.
    """
    print("\n" + "=" * 60)
    print("Step 3 — Building LLM context column")
    print("=" * 60)

    def row_to_text(r):
        parts = []

        # Who
        who = []
        if pd.notna(r.get('age')):        who.append(f"age {r['age']}")
        if pd.notna(r.get('gender')):     who.append(str(r['gender']))
        if pd.notna(r.get('income')):     who.append(f"income: {r['income']}")
        if pd.notna(r.get('education')): who.append(str(r['education']))
        if who: parts.append("Person: " + ", ".join(who))

        # When
        if pd.notna(r.get('started_at')):
            parts.append(f"Time: {r['started_at']} → {r.get('finished_at','?')}")

        # Where
        loc = []
        if pd.notna(r.get('address')):       loc.append(str(r['address']))
        elif pd.notna(r.get('city')):        loc.append(str(r['city']))
        if pd.notna(r.get('act_CH_BEZ_D')): loc.append(f"zone: {r['act_CH_BEZ_D']}")
        if loc: parts.append("Location: " + "; ".join(loc))

        # How
        trip = []
        if pd.notna(r.get('mode')):   trip.append(f"mode: {r['mode']}")
        if pd.notna(r.get('length')): trip.append(f"distance: {float(r['length']):.0f} m")
        if trip: parts.append("Trip: " + ", ".join(trip))

        # Why
        purpose = r.get('act_imputed_purpose')
        if pd.notna(purpose):
            parts.append(f"Activity purpose: {purpose}")

        # Around
        nearby = r.get('nearby_places')
        if pd.notna(nearby) and str(nearby).strip():
            parts.append(f"Nearby places: {nearby}")

        # POI counts
        cat_counts = []
        for cat in POI_Categories:
            col = f'poi_{cat.lower()}'
            v = r.get(col, 0)
            if pd.notna(v) and int(v) > 0:
                cat_counts.append(f"{cat}: {int(v)}")
        if cat_counts:
            parts.append("POI counts ({}m): ".format(poi_radius_m) +
                         ", ".join(cat_counts))

        return " | ".join(parts)

    gdf['llm_context'] = gdf.apply(row_to_text, axis=1)
    print(f"  ✓ llm_context column created")
    print(f"\n  Example (row 0):\n  {gdf['llm_context'].iloc[0]}")
    return gdf

# ============================================================================
# 5.  Save
# ============================================================================

def save(gdf: gpd.GeoDataFrame, path: Path):
    # Drop geometry column before saving to CSV (keep as WKT in 'geometry')
    df_out = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    if hasattr(gdf, 'geometry') and gdf.geometry is not None:
        df_out['geometry'] = gdf.geometry.apply(
            lambda g: f"POINT ({g.x} {g.y})" if g else np.nan
        )

    df_out.to_csv(path, index=False)
    print(f"\n  ✓ Saved: {path}")
    print(f"    Rows    : {len(df_out):,}")
    print(f"    Columns ({len(df_out.columns)}): {list(df_out.columns)}")


# ============================================================================
# 6.  Main
# ============================================================================

def main():
    print("=" * 60)
    print("Geocontext enrichment")
    print("=" * 60)

    # Load
    gdf = load_sp(sp_enriched)

    # Step 1: reverse geocoding
    gdf = add_reverse_geocoding(gdf, nominatim_cache)

    # Step 2: POI context
    gdf = add_poi_context(gdf, poi_path, radius_m=poi_radius_m, top_n=poi_top_n)

    # Step 3: build LLM context text
    gdf = build_llm_context(gdf)

    # Save
    print("\n" + "=" * 60)
    print("Saving sp_final_llms.csv")
    print("=" * 60)
    save(gdf, out_file)

    # Preview
    print("\n" + "=" * 60)
    print("Done — Column overview")
    print("=" * 60)
    preview_cols = [c for c in [
        'user_id', 'started_at', 'mode', 'length',
        'act_imputed_purpose', 'act_CH_BEZ_D',
        'address', 'city',
        'poi_shopping', 'poi_entertainment', 'poi_transportation',
        'nearby_places',
        'age', 'gender', 'income',
        'llm_context',
    ] if c in gdf.columns]
    print(gdf[preview_cols].head(2).to_string())


if __name__ == '__main__':
    main()
