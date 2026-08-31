"""
Merge MOBIS datasets
=============================================================================
Data-flow
---------
  sp_all (2300 users — the filtered set of users who completed the survey)
    │
    │  Step 0: filter legs, activities, participants to sp_all users only
    │
    ├─[user_id + time overlap]── mobis_legs
    ├─[user_id + time overlap]── mobis_activities
    │           │
    │           ▼
    │     sp_spatial.csv          ← output 1
    │     (spatial + behavioural, no sociodemographics)
    │
    └─[user_id]────────────────── mobis_tracked_participants
                │
                ▼
          sp_enriched.csv         ← output 2
          (sp_spatial + sociodemographics)

Time-overlap rule
-----------------
  sp record [sp_start, sp_end] matches leg/activity [a_start, a_end] when:
      sp_start < a_end  and  a_start < sp_end   (intervals overlap)
  Among all overlapping candidates the one with the longest overlap is chosen.
=============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# Config
# ============================================================================
data_dir  = Path('/data/baliu/thesis/00data')
sp_file   = data_dir / '00_filter_user/sp_all.csv'
legs_file = data_dir / '01_survey/mobis_legs.csv'
act_file  = data_dir / '01_survey/mobis_activities.csv'
part_file = data_dir / '01_survey/mobis_tracked_participants.csv'

out_spatial  = data_dir / 'sp_spatial.csv'    # output 1 — no socio
out_enriched = data_dir / 'sp_enriched.csv'   # output 2 — with socio


# ============================================================================
# 1.  Load & filter all tables to sp_all users only
# ============================================================================

def load_tables():
    print("=" * 60)
    print("Loading & filtering tables to sp_all users")
    print("=" * 60)

    # sp_all — defines the reference user set
    sp = pd.read_csv(sp_file)
    sp['started_at']  = pd.to_datetime(sp['started_at'],  utc=True, errors='coerce')
    sp['finished_at'] = pd.to_datetime(sp['finished_at'], utc=True, errors='coerce')
    sp['user_id']     = sp['user_id'].astype(str).str.strip()
    sp_users          = set(sp['user_id'].unique())
    print(f"  sp_all                    : {len(sp):>7,} rows  |  "
          f"{len(sp_users)} users  ← reference population")

    # mobis_legs — filter to sp_all users
    legs = pd.read_csv(legs_file)
    legs['started_at']  = pd.to_datetime(legs['started_at'],  utc=True, errors='coerce')
    legs['finished_at'] = pd.to_datetime(legs['finished_at'], utc=True, errors='coerce')
    legs['user_id']     = legs['user_id'].astype(str).str.strip()
    n_before = len(legs)
    legs = legs[legs['user_id'].isin(sp_users)].reset_index(drop=True)
    print(f"  mobis_legs                : {n_before:>7,} total  →  "
          f"{len(legs):,} kept  ({legs['user_id'].nunique()} sp users)")

    # mobis_activities — filter to sp_all users
    act = pd.read_csv(act_file)
    act['started_at']  = pd.to_datetime(act['started_at'],  utc=True, errors='coerce')
    act['finished_at'] = pd.to_datetime(act['finished_at'], utc=True, errors='coerce')
    act['user_id']     = act['user_id'].astype(str).str.strip()
    n_before = len(act)
    act = act[act['user_id'].isin(sp_users)].reset_index(drop=True)
    print(f"  mobis_activities          : {n_before:>7,} total  →  "
          f"{len(act):,} kept  ({act['user_id'].nunique()} sp users)")

    # mobis_tracked_participants — filter to sp_all users
    part = pd.read_csv(part_file)
    part['user_id'] = part['user_id'].astype(str).str.strip()
    n_before = len(part)
    part = part[part['user_id'].isin(sp_users)].reset_index(drop=True)
    print(f"  mobis_tracked_participants: {n_before:>7,} total  →  "
          f"{len(part):,} kept  ({part['user_id'].nunique()} sp users)")

    return sp, legs, act, part


# ============================================================================
# 2.  Time-overlap merge step
# ============================================================================

def _time_overlap_merge(sp: pd.DataFrame,
                         other: pd.DataFrame,
                         other_cols: list,
                         prefix: str) -> pd.DataFrame:
    """
    For every sp row find the row in `other` (same user) with the greatest
    time overlap, and attach `other_cols` with the given prefix.
    """
    results = []
    common  = sorted(set(sp['user_id'].unique()) & set(other['user_id'].unique()))
    print(f"    Users with records in both tables: {len(common)}")

    for uid in common:
        sp_u    = sp[sp['user_id'] == uid]
        other_u = other[other['user_id'] == uid]

        for sp_idx, sp_row in sp_u.iterrows():
            sp_s = sp_row['started_at']
            sp_e = sp_row['finished_at']

            # Vectorised overlap in seconds
            overlap = (
                np.minimum(sp_e, other_u['finished_at']) -
                np.maximum(sp_s, other_u['started_at'])
            ).dt.total_seconds()

            if overlap.max() > 0:
                best_idx = overlap.idxmax()
                row_dict = other_u.loc[best_idx, other_cols].to_dict()
                row_dict['_overlap_sec'] = float(overlap[best_idx])
            else:
                row_dict = {c: np.nan for c in other_cols}
                row_dict['_overlap_sec'] = 0.0

            row_dict['_sp_index'] = sp_idx
            results.append(row_dict)

    if not results:
        for c in other_cols:
            sp[f'{prefix}{c}'] = np.nan
        sp[f'{prefix}overlap_sec'] = 0.0
        return sp

    match_df = (
        pd.DataFrame(results)
          .set_index('_sp_index')
          .rename(columns={c: f'{prefix}{c}' for c in other_cols})
          .rename(columns={'_overlap_sec': f'{prefix}overlap_sec'})
    )
    return sp.join(match_df, how='left')


# ============================================================================
# 3.  Merge mobis_activities  →  adds act_* columns
# ============================================================================

def merge_activities(sp: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 1 — Merge mobis_activities  (time overlap)")
    print("=" * 60)

    act_cols = [c for c in [
        'imputed_purpose', 'labeled_purpose',
        'CH_CODE_HN', 'CH_BEZ_D',
        'zoning_id', 'dist_to_zoning',
        'in_switzerland',
    ] if c in act.columns]

    sp_out  = _time_overlap_merge(sp, act, act_cols, prefix='act_')
    matched = sp_out['act_imputed_purpose'].notna().sum()
    print(f"  Matched : {matched:,} / {len(sp):,} rows  "
          f"({matched / len(sp) * 100:.1f}%)")
    print(f"  Purpose distribution (imputed):")
    print(sp_out['act_imputed_purpose'].value_counts().to_string())
    return sp_out


# ============================================================================
# 4.  Merge mobis_legs  →  adds leg_* columns
# ============================================================================

def merge_legs(sp: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 2 — Merge mobis_legs  (time overlap)")
    print("=" * 60)

    leg_cols = [c for c in [
        'mode', 'length', 'duration',
        'labeled_purpose', 'imputed_purpose',
        'treatment', 'phase',
        'in_switzerland',
    ] if c in legs.columns]

    sp_out  = _time_overlap_merge(sp, legs, leg_cols, prefix='leg_')
    matched = sp_out['leg_mode'].notna().sum()
    print(f"  Matched : {matched:,} / {len(sp):,} rows  "
          f"({matched / len(sp) * 100:.1f}%)")
    print(f"  Mode distribution (legs):")
    print(sp_out['leg_mode'].value_counts().to_string())
    return sp_out


# ============================================================================
# 5.  Merge sociodemographics  →  adds socio columns
# ============================================================================

def merge_sociodemographics(sp: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 3 — Merge sociodemographics  (user_id join)")
    print("=" * 60)

    socio_cols = [c for c in [
        'user_id',
        'language', 'wave', 'postcode_home',
        'age', 'gender', 'education', 'main_employment', 'income',
        'household_size', 'citizen_1',
        'work_status_employed', 'work_status_student', 'work_status_retired',
        'workload_jobs_main', 'postcode_jobs_main',
        'own_vehicles_car', 'own_vehicles_bicycle',
        'car_fuel', 'car_size',
        'pt_pass_ga', 'pt_pass_half_fare',
        'freq_cardriver_own_car', 'freq_pt_train', 'freq_pt_local_pt',
        'freq_bike_own_bike',
        'gen_accessibility', 'oev_accessibility', 'miv_accessibility',
        'gen_access_quintile', 'finished_tracking_study',
    ] if c in part.columns]

    sp_out  = sp.merge(part[socio_cols], on='user_id', how='left')
    matched = sp_out['age'].notna().sum()
    print(f"  Rows with socio info : {matched:,} / {len(sp):,}  "
          f"({matched / len(sp) * 100:.1f}%)")
    print(f"  Rows without socio   : {len(sp) - matched:,}  "
          f"(in sp_all but not in participants file)")
    return sp_out


# ============================================================================
# 6.  Save 
# ============================================================================

def _save(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index=False)
    print(f"  ✓ {label}")
    print(f"    Path    : {path}")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns ({len(df.columns)}): {list(df.columns)}")


# ============================================================================
# 7.  Main
# ============================================================================

def main():
    print("=" * 60)
    print("MOBIS data merge")
    print("=" * 60)

    sp, legs, act, part = load_tables()

    # ── Output 1: sp_all + activities + legs  (no socio) ────────────────────
    sp_spatial = merge_activities(sp, act)
    sp_spatial = merge_legs(sp_spatial, legs)

    print("\n" + "=" * 60)
    print("Saving Output 1 — sp_spatial.csv")
    print("=" * 60)
    _save(sp_spatial, out_spatial,
          'sp_spatial.csv  (spatial + behavioural, no sociodemographics)')

    # ── Output 2: sp_spatial + sociodemographics ─────────────────────────────
    sp_enriched = merge_sociodemographics(sp_spatial, part)

    print("\n" + "=" * 60)
    print("Saving Output 2 — sp_enriched.csv")
    print("=" * 60)
    _save(sp_enriched, out_enriched,
          'sp_enriched.csv  (spatial + behavioural + sociodemographics)')

    # ── Quick preview ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Quick preview (first 3 rows)")
    print("=" * 60)
    preview_cols = [c for c in [
        'user_id', 'started_at', 'mode',
        'act_imputed_purpose', 'leg_mode', 'leg_length',
        'age', 'gender', 'income',
    ] if c in sp_enriched.columns]
    print(sp_enriched[preview_cols].head(3).to_string())


if __name__ == '__main__':
    main()
