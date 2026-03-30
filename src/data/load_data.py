# ---------------------------------------------------------------------------
# About Data load_data.py
# Load and clean raw staypoint data
# ---------------------------------------------------------------------------
import re
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------
# wkt Point → (lon, lat)
# --------------------------------------
WKT_RE = re.compile(
    r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
    re.I,
)

def _extract_lonlat(s: str) -> tuple[float, float]:
    m = WKT_RE.search(str(s))
    if not m:
        return (np.nan, np.nan)
    return float(m.group(1)), float(m.group(2))

# --------------------------------------
# Load raw staypoints csv
# --------------------------------------
def load_staypoints(path: Path, coord_prec: int = 4) -> pd.DataFrame:
    """
    Load staypoints CSV, parse geometry, add time and mobility features.

    Returns cleaned DataFrame with columns:
        user_id, location_id, started_at, finished_at,
        lon, lat, mode, act_duration, act_duration_h,
        length_km, date, dow, hour_bin
    """
    print(f"Loading staypoints from {path} ...")
    sp = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")
    print(f"Raw shape: {sp.shape}")

    # Datetimes
    sp["started_at"]  = pd.to_datetime(sp["started_at"],  errors="coerce", utc=True)
    sp["finished_at"] = pd.to_datetime(sp.get("finished_at", pd.Series(dtype=str)),
                                        errors="coerce", utc=True)

    # Drop rows with missing essentials
    sp = sp.dropna(subset=["user_id", "started_at", "location_id"]).copy()
    sp["user_id"]     = sp["user_id"].astype(str)
    sp["location_id"] = sp["location_id"].astype(str)

    # Numeric fields
    sp["act_duration"]   = pd.to_numeric(sp.get("act_duration"),   errors="coerce")
    sp["act_duration_h"] = (sp["act_duration"] / 60.0).round(1)
    sp["length_km"]      = (pd.to_numeric(sp.get("length"), errors="coerce") / 1000.0).round(1)

    # Time features
    sp["date"]     = sp["started_at"].dt.date
    sp["dow"]      = sp["started_at"].dt.dayofweek.astype(int)   # 0 = Monday
    sp["hour_bin"] = sp["started_at"].dt.hour.astype(int)

    # Coordinates from wkt geometry
    if "lon" not in sp.columns or sp["lon"].isna().all():
        if "geometry" in sp.columns:
            lonlat = sp["geometry"].astype(str).apply(_extract_lonlat)
            sp[["lon", "lat"]] = pd.DataFrame(lonlat.tolist(), index=sp.index)
        else:
            sp["lon"] = np.nan
            sp["lat"]  = np.nan

    sp["lon"] = pd.to_numeric(sp["lon"], errors="coerce").round(coord_prec)
    sp["lat"]  = pd.to_numeric(sp["lat"], errors="coerce").round(coord_prec)

    # Mode
    if "mode" not in sp.columns:
        sp["mode"] = "unknown"
    sp["mode"] = sp["mode"].fillna("unknown").astype(str).str.lower()

    print(f"Cleaned shape: {sp.shape}")
    return sp


# --------------------------------------
# CLI test
# --------------------------------------
if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/data/baliu/python_code/data/sp_all copy.csv")
    sp = load_staypoints(path)
    print(sp.dtypes)
    print(sp.head(3))
