# ---------------------------------------------------------------------------
# Build compact prompts for sociodemographic prediction
# Fixed: 28 Mar 2026
# Fix: POI CRS handling — matches notebook's load_poi_frame behaviour
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

# --------------------------------------
# Paths
# --------------------------------------
SP_PATH   = Path("/data/baliu/python_code/data/sp_sampled2_with_geocontext.csv")
POI_PATH  = Path("/data/baliu/python_code/data/version2/data/final_pois_nob.gpkg")
PROMPTS_OUT = Path("/data/baliu/python_code/data/prompts_v4_compact_28Mar2026.txt")

# --------------------------------------
# Load data
# --------------------------------------
print("Loading sp_sampled2...")
sp_sampled2 = pd.read_csv(SP_PATH)
sp_sampled2["location_id"] = sp_sampled2["location_id"].astype(str)
print(f"sp_sampled2: {sp_sampled2.shape}")
print(f"columns: {sp_sampled2.columns.tolist()}")

print("\nLoading POI...")
poi = gpd.read_file(POI_PATH)
print(f"poi: {poi.shape}")
print(f"poi crs (raw): {poi.crs}")
print(f"poi columns: {poi.columns.tolist()}")
print(f"poi x sample: {poi.geometry.x[:3].tolist()}")
print(f"poi y sample: {poi.geometry.y[:3].tolist()}")

# --------------------------------------
# CRS fix — matches notebook's load_poi_frame
# The POI file coordinates are in LV95 (EPSG:2056)
# but CRS metadata may be missing or wrong
# --------------------------------------
def load_poi_frame(path: Path) -> gpd.GeoDataFrame:
    poi = gpd.read_file(path)
    # If CRS is already 2056, keep it
    if poi.crs is not None and poi.crs.to_epsg() == 2056:
        print("✅ POI already in EPSG:2056")
        return poi
    # If CRS is 4326 (WGS84), reproject to 2056
    if poi.crs is not None and poi.crs.to_epsg() == 4326:
        print("🔄 Reprojecting POI from EPSG:4326 → EPSG:2056")
        return poi.to_crs(2056)
    # If CRS is None or unknown, set it as 2056 (coordinates look like Swiss)
    x_sample = poi.geometry.x.median()
    if 2_000_000 < x_sample < 3_000_000:
        print("✅ Coordinates look like LV95 (2056), setting CRS")
        return poi.set_crs(epsg=2056)
    # Coordinates look like WGS84 (lon/lat)
    if -180 < x_sample < 180:
        print("🔄 Coordinates look like WGS84, setting 4326 then reprojecting to 2056")
        return poi.set_crs(epsg=4326).to_crs(2056)
    # Fallback — just set as 2056 (matches notebook behaviour)
    print("⚠️  Unknown CRS, forcing EPSG:2056 (matches notebook)")
    return poi.set_crs(epsg=2056, allow_override=True)

poi = load_poi_frame(POI_PATH)
print(f"\nPOI after fix — crs: {poi.crs}")
print(f"POI x range: {poi.geometry.x.min():.0f} to {poi.geometry.x.max():.0f}")
print(f"POI y range: {poi.geometry.y.min():.0f} to {poi.geometry.y.max():.0f}")

# --------------------------------------
# Helper functions
# --------------------------------------
def clean_addr_part(s):
    if pd.isna(s): return None
    s = str(s).strip()
    return None if s.lower() in ["none", "nan", "-", ""] else s

def clean_text_part(s):
    if pd.isna(s): return None
    s = str(s).strip()
    return None if s.lower() in ["none", "nan", "-", ""] else s

dow_names = {0:"Sun", 1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat"}

def bearing_to_direction(dx, dy):
    angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((angle + 22.5) / 45) % 8]

# --------------------------------------
# POI context builder
# --------------------------------------
def get_poi_context_for_prompt(sp_sampled, poi, top_k=5):
    loc = gpd.GeoDataFrame(
        sp_sampled.copy(),
        geometry=gpd.points_from_xy(
            sp_sampled["lon"].astype(float),
            sp_sampled["lat"].astype(float)
        ),
        crs="EPSG:4326"
    ).to_crs(2056)

    # POI already in 2056 from load_poi_frame
    poi_proj = poi.copy()
    if poi_proj.crs is None or poi_proj.crs.to_epsg() != 2056:
        poi_proj = poi_proj.set_crs(epsg=2056, allow_override=True)

    # Sanity check distances
    loc_x_med = loc.geometry.x.median()
    poi_x_med = poi_proj.geometry.x.median()
    print(f"  loc x median: {loc_x_med:.0f} | poi x median: {poi_x_med:.0f}")
    if abs(loc_x_med - poi_x_med) > 500_000:
        print("  ⚠️  Large coordinate gap — CRS mismatch likely!")

    poi_xy = np.c_[poi_proj.geometry.x.values, poi_proj.geometry.y.values]
    tree   = cKDTree(poi_xy)
    loc_xy = np.c_[loc.geometry.x.values, loc.geometry.y.values]
    dists, idxs = tree.query(loc_xy, k=top_k * 3)

    print(f"  nearest distance sample (m): {dists[:3, 0].tolist()}")

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
    joined = joined[
        joined["category"].notna() &
        (~joined["category"].str.lower().isin(["unknown", "others"]))
    ]
    joined["addr_poi_dist_km"] = (joined["addr_poi_dist_m"] / 1000).round(3)
    joined["direction"] = [
        bearing_to_direction(dx, dy)
        for dx, dy in zip(joined.dx, joined.dy)
    ]
    joined = (
        joined.sort_values("addr_poi_dist_m")
        .groupby("location_id", group_keys=False)
        .head(top_k)
    )
    return joined[["location_id","name","category","addr_poi_dist_km","direction"]]


def build_poi_prompt_text(df):
    out = []
    for _, r in df.iterrows():
        name     = clean_text_part(r.get("name"))
        category = clean_text_part(r.get("category"))
        if name is None and category is None:
            continue
        label = " ".join(filter(None, [category, name]))
        out.append(f"{r['addr_poi_dist_km']}km {r['direction']} {label}")
    return "; ".join(out)

# --------------------------------------
# Build POI context and merge
# --------------------------------------
if "nearby_places" in sp_sampled2.columns and sp_sampled2["nearby_places"].notna().sum() > 0:
    print(f"\n✅ nearby_places already exists ({sp_sampled2['nearby_places'].notna().sum()} rows), skipping POI computation")
else:
    print("\n🔧 Computing POI context...")
    poi_prompt_df = (
        get_poi_context_for_prompt(sp_sampled2, poi, top_k=5)
        .groupby("location_id", group_keys=False)
        .apply(build_poi_prompt_text, include_groups=False)
        .reset_index(name="nearby_places")
    )
    poi_prompt_df["location_id"] = poi_prompt_df["location_id"].astype(str)

    print(f"poi_prompt_df shape: {poi_prompt_df.shape}")
    print(f"poi_prompt_df sample:\n{poi_prompt_df.head(3)}")

    sp_sampled2 = sp_sampled2.merge(poi_prompt_df, on="location_id", how="left")

    print(f"\nnearby_places in columns: {'nearby_places' in sp_sampled2.columns}")
    print(f"nearby_places filled: {sp_sampled2['nearby_places'].notna().sum()} / {len(sp_sampled2)}")

    sp_sampled2.to_csv(SP_PATH, index=False)
    print(f"✅ Saved updated sp_sampled2 → {SP_PATH}")

# --------------------------------------
# Compact token builder
# --------------------------------------
def tokens_compact_1week(df_u: pd.DataFrame, max_events: int = 40, prec: int = 4) -> list[str]:
    duration_col = None
    for c in df_u.columns:
        if c.lower() in ["duration","duration_min","act_duration","act_duration_min","dur_min"]:
            duration_col = c
            break

    use_cols = [
        "started_at","dow","hour_bin","location_id",
        "lon","lat","mode","city","neighbourhood",
        "road","nearby_places","postcode"
    ]
    if duration_col:
        use_cols.append(duration_col)

    df = df_u.loc[:, [c for c in use_cols if c in df_u.columns]].copy()
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"]).sort_values("started_at")
    if len(df) > max_events:
        df = df.head(max_events)

    toks = []
    current_date = None

    for r in df.itertuples(index=False):
        t = r.started_at
        date_str = t.date().isoformat()
        if date_str != current_date:
            toks.append(f"[{date_str}]")
            current_date = date_str

        hhmm = t.strftime("%H:%M")
        dow_label = dow_names.get(int(getattr(r, "dow", 0)), "?")

        addr_parts = [
            clean_addr_part(getattr(r, "road", None)),
            clean_addr_part(getattr(r, "city", None)),
        ]
        addr_parts = [p for p in addr_parts if p]
        postcode = clean_addr_part(getattr(r, "postcode", None))
        if postcode:
            addr_parts.append(postcode)
        addr = " ".join(addr_parts) if addr_parts else "unknown"

        mode = str(getattr(r, "mode", "?")).lower()
        dur_val = getattr(r, duration_col, 0) if duration_col else 0
        dur = int(round(float(dur_val))) if pd.notna(dur_val) else 0

        nearby_raw = getattr(r, "nearby_places", None)
        if isinstance(nearby_raw, list):
            nearby_str = "; ".join(nearby_raw[:3])
        elif isinstance(nearby_raw, str) and nearby_raw.strip():
            parts = [p.strip() for p in nearby_raw.split(";")][:3]
            nearby_str = "; ".join(parts)
        else:
            nearby_str = ""

        line = f"{hhmm} {dow_label} | {addr} | {mode} {dur}min"
        if nearby_str:
            line += f" | {nearby_str}"
        toks.append(line)

    return toks

# --------------------------------------
# Build prompts
# --------------------------------------
MAX_EVENTS = 40
PREC = 4
SEP = "=" * 80

sp_prompt = sp_sampled2.copy()
sp_prompt["date_str"] = (
    pd.to_datetime(sp_prompt["started_at"], errors="coerce")
    .dt.tz_localize(None)
    .dt.date
    .astype(str)
)

user_dates = (
    sp_prompt
    .dropna(subset=["started_at"])
    .loc[:, ["user_id","date_str"]]
    .drop_duplicates()
    .groupby("user_id")["date_str"]
    .apply(list)
)
print(f"\nTotal users: {len(user_dates)}")

rows_prompts = []
for user_id, dates in user_dates.items():
    df_u = sp_prompt[
        (sp_prompt["user_id"] == user_id) &
        (sp_prompt["date_str"].isin(dates))
    ]
    toks = tokens_compact_1week(df_u, max_events=MAX_EVENTS, prec=PREC)
    if not toks:
        continue

    prompt = (
        f"User: {user_id}\n\n"
        f"Mobility evidence:\n"
        + "\n".join(toks)
    )
    rows_prompts.append({"user_id": user_id, "date": dates, "prompt": prompt})

# --------------------------------------
# Save
# --------------------------------------
PROMPTS_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(PROMPTS_OUT, "w", encoding="utf-8") as f:
    for r in rows_prompts:
        f.write(r["prompt"])
        f.write("\n\n" + SEP + "\n\n")

print(f"\n✅ Saved {len(rows_prompts)} prompts → {PROMPTS_OUT}")

if rows_prompts:
    sample = rows_prompts[0]["prompt"]
    approx_tokens = len(sample.split()) * 1.3
    print(f"📊 Sample word count: {len(sample.split())}")
    print(f"📊 Approx tokens: {approx_tokens:.0f}")
    print(f"\n--- Sample (first 600 chars) ---")
    print(sample[:600])
    print("...")
