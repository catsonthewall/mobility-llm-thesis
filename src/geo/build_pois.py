# ---------------------------------------------------------------------------
# Step 1 build_pois.py
# Build POI dataset from raw OSM shapefiles → final_pois_nob.gpkg
# ---------------------------------------------------------------------------

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

# --------------------------------------
# Config
# --------------------------------------
# path to raw OSM shapefiles (downloaded from https://download.geofabrik.de/europe/switzerland.html)
RAW_DIR  = Path("/data/baliu/python_code/data/switzerland-251226-free copy")
OUT_DIR  = Path("/data/baliu/python_code/data/version2/data")
OUT_GPKG = OUT_DIR / "final_pois_nob.gpkg"

# OSM fclass → category mapping
FCLASS_TO_CATEGORY = {
    # Civic
    "church": "Civic", "place_of_worship": "Civic", "mosque": "Civic",
    "synagogue": "Civic", "hindu_temple": "Civic", "wayside_cross": "Civic",
    "wayside_shrine": "Civic",
    # Entertainment
    "park": "Entertainment", "nature_reserve": "Entertainment",
    "playground": "Entertainment", "stadium": "Entertainment",
    "theatre": "Entertainment", "cinema": "Entertainment",
    "arts_centre": "Entertainment", "museum": "Entertainment",
    "zoo": "Entertainment", "theme_park": "Entertainment",
    "swimming_pool": "Entertainment", "sports_centre": "Entertainment",
    "pitch": "Entertainment", "beach": "Entertainment",
    # Shopping
    "supermarket": "Shopping", "convenience": "Shopping",
    "mall": "Shopping", "department_store": "Shopping",
    "clothes": "Shopping", "shoes": "Shopping",
    "bakery": "Shopping", "butcher": "Shopping",
    "pharmacy": "Shopping", "florist": "Shopping",
    "books": "Shopping", "electronics": "Shopping",
    # Transportation
    "railway_station": "Transportation", "bus_station": "Transportation",
    "tram_stop": "Transportation", "subway_entrance": "Transportation",
    "ferry_terminal": "Transportation", "airport": "Transportation",
    "bus_stop": "Transportation", "taxi": "Transportation",
    # Services
    "hospital": "Services", "clinic": "Services", "doctors": "Services",
    "dentist": "Services", "bank": "Services", "atm": "Services",
    "post_office": "Services", "police": "Services",
    "fire_station": "Services", "library": "Services",
    # Schools
    "school": "Schools", "university": "Schools", "college": "Schools",
    "kindergarten": "Schools",
    # Residential
    "hotel": "Residential", "hostel": "Residential", "motel": "Residential",
    "guest_house": "Residential", "apartment": "Residential",
    "restaurant": "Residential", "cafe": "Residential",
    "fast_food": "Residential", "bar": "Residential", "pub": "Residential",
}


def assign_category(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map OSM fclass to human-readable category."""
    gdf = gdf.copy()
    if "fclass" in gdf.columns:
        gdf["category"] = gdf["fclass"].map(FCLASS_TO_CATEGORY).fillna("Others")
    elif "type" in gdf.columns:
        gdf["category"] = gdf["type"].map(FCLASS_TO_CATEGORY).fillna("Others")
    else:
        gdf["category"] = "Unknown"
    return gdf


def read_poi_layer(path: Path, layer_name: str) -> gpd.GeoDataFrame | None:
    """Safely read one OSM shapefile layer."""
    if not path.exists():
        print(f"Missing: {path.name}")
        return None
    print(f"Reading {layer_name}: {path.name}")
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna()].copy()
    # Keep only Point geometries
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    return gdf


def build_final_pois(all_pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign categories, reproject to LV95 (EPSG:2056), select final columns."""
    all_pois = assign_category(all_pois)
    all_pois = all_pois.to_crs("EPSG:2056")

    # Keep only useful columns
    keep = ["name", "code", "category", "geometry"]
    keep = [c for c in keep if c in all_pois.columns]
    all_pois = all_pois[keep].copy()

    # Reset index as id
    all_pois = all_pois.reset_index(drop=True)
    all_pois.insert(0, "id", all_pois.index)

    # Drop unknown and others for cleaner prompts
    # (keep them — they still carry distance/direction signal)
    print(f"Category distribution:\n{all_pois['category'].value_counts()}")
    return all_pois


def _read_all_layers() -> gpd.GeoDataFrame:
    """Read and concatenate all relevant OSM layers."""
    gdfs = []

    # Places of worship
    p = read_poi_layer(RAW_DIR / "gis_osm_pofw_free_1.shp", "pofw")
    if p is not None:
        gdfs.append(p)

    # Transport stops
    p = read_poi_layer(RAW_DIR / "gis_osm_transport_free_1.shp", "transport")
    if p is not None:
        gdfs.append(p)

    # Natural features (skip trees)
    p = read_poi_layer(RAW_DIR / "gis_osm_natural_free_1.shp", "natural")
    if p is not None and "fclass" in p.columns:
        p = p[p["fclass"] != "tree"]
        gdfs.append(p)

    # Natural areas → beach centroids
    p = gpd.read_file(RAW_DIR / "gis_osm_natural_a_free_1.shp") \
        if (RAW_DIR / "gis_osm_natural_a_free_1.shp").exists() else None
    if p is not None and "fclass" in p.columns:
        p = p[p["fclass"] == "beach"].copy()
        p["geometry"] = p.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")
        p = p[p.geometry.notna()]
        gdfs.append(p)

    # POIs (main layer)
    p = read_poi_layer(RAW_DIR / "gis_osm_pois_free_1.shp", "pois")
    if p is not None:
        gdfs.append(p)

    # Traffic (parking only)
    PARKING_CLASSES = {
        "parking", "parking_bicycle",
        "parking_underground", "parking_multistorey"
    }
    p = read_poi_layer(RAW_DIR / "gis_osm_traffic_free_1.shp", "traffic")
    if p is not None and "fclass" in p.columns:
        p = p[p["fclass"].isin(PARKING_CLASSES)]
        gdfs.append(p)

    # Traffic areas → parking centroids
    p_path = RAW_DIR / "gis_osm_traffic_a_free_1.shp"
    if p_path.exists():
        p = gpd.read_file(p_path)
        if "fclass" in p.columns:
            p = p[p["fclass"].isin(PARKING_CLASSES)].copy()
            p["geometry"] = p.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")
            p = p[p.geometry.notna()]
            gdfs.append(p)

    if not gdfs:
        raise RuntimeError("No POI layers could be loaded — check RAW_DIR path.")

    combined = pd.concat(gdfs, ignore_index=True)
    print(f"Total POIs loaded: {len(combined)}")
    return combined


def build(force: bool = False) -> None:
    """Main entry point. Builds final_pois_nob.gpkg."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_GPKG.exists() and not force:
        raise RuntimeError(
            f"{OUT_GPKG} already exists.\n"
            "Pass force=True or delete manually to rebuild."
        )

    all_pois = _read_all_layers()
    all_pois = build_final_pois(all_pois)
    all_pois.to_file(OUT_GPKG, driver="GPKG")
    print(f"final_pois_nob.gpkg written: {len(all_pois)} POIs → {OUT_GPKG}")
    print(all_pois.head())


if __name__ == "__main__":
    build()
