# ---------------------------------------------------------------------------
# scripts/run_pipeline.py
# Full data pipeline: load → clean → sample → attach POI → build prompts
#
# Usage:
#   python scripts/run_pipeline.py
#   python scripts/run_pipeline.py --force-poi   # recompute POI even if exists
# ---------------------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── src imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_data   import load_staypoints
from src.data.sample      import sample_one_week_per_user
from src.geo.poi_context  import load_poi_frame, attach_poi_context
from src.utils.io         import save_csv_snapshot

# ── Config ───────────────────────────────────────────────────────────────────
DATA_SP      = Path("/data/baliu/python_code/data/sp_all copy.csv")
POI_PATH     = Path("/data/baliu/python_code/data/version2/data/final_pois_nob.gpkg")
NOM_CACHE    = Path("/data/baliu/python_code/cache/nominatim_cache.parquet")
SP_OUT       = Path("/data/baliu/python_code/data/sp_sampled2_with_geocontext.csv")
PROMPTS_OUT  = Path("/data/baliu/python_code/data/prompts_v4_compact_28Mar2026.txt")

RANDOM_SEED  = 42
MAX_EVENTS   = 40
PREC         = 4
SEP          = "=" * 80

DOW_NAMES = {0:"Sun", 1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat"}


# ── Token builder ─────────────────────────────────────────────────────────────
def _clean(s) -> str | None:
    if pd.isna(s): return None
    s = str(s).strip()
    return None if s.lower() in {"none","nan","-",""} else s


def tokens_compact_1week(df_u: pd.DataFrame, max_events: int = 40, prec: int = 4) -> list[str]:
    """Build compact mobility tokens — one line per staypoint."""
    duration_col = next(
        (c for c in df_u.columns
         if c.lower() in {"duration","duration_min","act_duration","act_duration_min","dur_min"}),
        None
    )

    use_cols = ["started_at","dow","location_id","lon","lat","mode",
                "city","road","nearby_places","postcode"]
    if duration_col:
        use_cols.append(duration_col)

    df = df_u[[c for c in use_cols if c in df_u.columns]].copy()
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"]).sort_values("started_at").head(max_events)

    toks, current_date = [], None

    for r in df.itertuples(index=False):
        t = r.started_at
        date_str = t.date().isoformat()
        if date_str != current_date:
            toks.append(f"[{date_str}]")
            current_date = date_str

        dow_label = DOW_NAMES.get(int(getattr(r, "dow", 0)), "?")
        hhmm = t.strftime("%H:%M")

        addr_parts = [_clean(getattr(r, "road", None)), _clean(getattr(r, "city", None))]
        addr_parts = [p for p in addr_parts if p]
        pc = _clean(getattr(r, "postcode", None))
        if pc: addr_parts.append(pc)
        addr = " ".join(addr_parts) or "unknown"

        mode = str(getattr(r, "mode", "?")).lower()
        dur_val = getattr(r, duration_col, 0) if duration_col else 0
        dur = int(round(float(dur_val))) if pd.notna(dur_val) else 0

        nearby_raw = getattr(r, "nearby_places", None)
        if isinstance(nearby_raw, list):
            nearby_str = "; ".join(nearby_raw[:3])
        elif isinstance(nearby_raw, str) and nearby_raw.strip():
            nearby_str = "; ".join(p.strip() for p in nearby_raw.split(";"))[:3]
        else:
            nearby_str = ""

        line = f"{hhmm} {dow_label} | {addr} | {mode} {dur}min"
        if nearby_str:
            line += f" | {nearby_str}"
        toks.append(line)

    return toks


# ── Nominatim attach ──────────────────────────────────────────────────────────
def attach_nominatim(sp: pd.DataFrame, nom_path: Path) -> pd.DataFrame:
    """Attach reverse-geocoded address fields (road, city, postcode) from cache."""
    if not nom_path.exists():
        print(f"⚠️  Nominatim cache not found at {nom_path}, skipping address attachment")
        return sp

    nom = pd.read_parquet(nom_path)
    nom["location_id"] = nom["location_id"].astype(str)
    sp["location_id"]  = sp["location_id"].astype(str)

    addr_cols = [c for c in ["road","neighbourhood","city","postcode","country"]
                 if c in nom.columns]
    nom_sub = nom[["location_id"] + addr_cols].drop_duplicates("location_id")

    sp = sp.merge(nom_sub, on="location_id", how="left")
    print(f"✅ Nominatim attached: {sp['city'].notna().sum()} rows have city")
    return sp


# ── Build prompts ─────────────────────────────────────────────────────────────
def build_prompts(sp: pd.DataFrame) -> list[dict]:
    sp = sp.copy()
    sp["date_str"] = (
        pd.to_datetime(sp["started_at"], errors="coerce")
        .dt.tz_localize(None)
        .dt.date.astype(str)
    )

    user_dates = (
        sp.dropna(subset=["started_at"])
        .loc[:, ["user_id","date_str"]]
        .drop_duplicates()
        .groupby("user_id")["date_str"]
        .apply(list)
    )

    rows = []
    for user_id, dates in user_dates.items():
        df_u = sp[(sp["user_id"] == user_id) & (sp["date_str"].isin(dates))]
        toks = tokens_compact_1week(df_u, max_events=MAX_EVENTS, prec=PREC)
        if not toks:
            continue
        prompt = f"User: {user_id}\n\nMobility evidence:\n" + "\n".join(toks)
        rows.append({"user_id": user_id, "date": dates, "prompt": prompt})

    print(f"✅ Built {len(rows)} prompts")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main(force_poi: bool = False) -> None:

    # 1. Load + clean
    sp = load_staypoints(DATA_SP, coord_prec=PREC)

    # 2. Sample one week per user
    sp_week = sample_one_week_per_user(sp, seed=RANDOM_SEED)

    # 3. Attach address (Nominatim)
    sp_week = attach_nominatim(sp_week, NOM_CACHE)

    # 4. Attach POI context
    if "nearby_places" in sp_week.columns and sp_week["nearby_places"].notna().sum() > 0 \
            and not force_poi:
        print("✅ nearby_places already present, skipping POI lookup")
    else:
        print("🔧 Loading POI and computing context...")
        poi = load_poi_frame(POI_PATH)
        sp_week = attach_poi_context(sp_week, poi, top_k=5)

    # 5. Save enriched staypoints
    sp_week.to_csv(SP_OUT, index=False)
    print(f"✅ Enriched staypoints saved → {SP_OUT}")

    # 6. Build prompts
    rows = build_prompts(sp_week)

    PROMPTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["prompt"])
            f.write("\n\n" + SEP + "\n\n")

    print(f"✅ Prompts saved → {PROMPTS_OUT} ({len(rows)} users)")

    # Sample token stats
    if rows:
        sample = rows[0]["prompt"]
        print(f"\n--- Sample prompt (first 500 chars) ---\n{sample[:500]}\n...")
        print(f"Approx tokens: {len(sample.split()) * 1.3:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-poi", action="store_true",
                        help="Recompute POI context even if already present")
    args = parser.parse_args()
    main(force_poi=args.force_poi)
