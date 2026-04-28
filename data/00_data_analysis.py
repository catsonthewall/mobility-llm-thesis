import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================================================
# Config — 
# ==================================================================================
sp_all  = Path('/data/baliu/python_code/gt_data/sp_all.csv')
survey  = Path('/data/baliu/python_code/gt_data/introSurvey_complete.csv')
id_filtered  = Path('/data/baliu/python_code/gt_data/mobis_filtered.csv') # ID csv
output  = Path('/data/baliu/python_code/data/demographic_statistics.csv')

# ============================================================================
# 1. Load the id list
# ============================================================================
id_df = pd.read_csv(id_filtered)

# Auto-detect the user_id column name (handles 'mobis_filtered is user_id ', 'id', 'participant_ID' etc.)
id_col = id_df.columns[0]
target_ids = id_df[id_col].astype(str).str.strip().unique().tolist()
print(f"Target user IDs loaded: {len(target_ids)}")
print(f"Sample IDs: {target_ids[:5]}")

# ============================================================================
# 2. Load sp_all and filter to target users only
# ============================================================================
sp = pd.read_csv(sp_all, dtype=str, engine='python', on_bad_lines='skip')
sp['user_id'] = sp['user_id'].astype(str).str.strip()

sp_filtered = sp[sp['user_id'].isin(target_ids)].copy()
print(f"\nsp_all total rows:       {len(sp)}")
print(f"Rows after ID filter:    {len(sp_filtered)}")
print(f"Unique users matched:    {sp_filtered['user_id'].nunique()}")

# Check who/users was not found
found_ids   = set(sp_filtered['user_id'].unique())
missing_ids = set(target_ids) - found_ids
if missing_ids:
    print(f"IDs not found in sp_all: {missing_ids}")

# ============================================================================
# 3. Load survey and join demographic characteristics 
# ============================================================================
survey = pd.read_csv(survey, dtype=str)
survey['participant_ID'] = survey['participant_ID'].astype(str).str.strip()

# Keep only the demographic columns we need
# Adjust column names below to match your actual survey column names
DEMO_COLS = ['participant_ID', 'gender', 'income', 'household_size', 'age']

# Only keep columns that we need 
demo_cols_present = [c for c in DEMO_COLS if c in survey.columns]
missing_demo_cols = set(DEMO_COLS) - set(demo_cols_present)
if missing_demo_cols:
    print(f"Demo columns not found in survey: {missing_demo_cols}")
    print(f"Available survey columns: {survey.columns.tolist()}")

survey_demo = survey[demo_cols_present].copy()

# Merge: one row per unique user_id
unique_users = sp_filtered[['user_id']].drop_duplicates()
merged = unique_users.merge(
    survey_demo,
    left_on='user_id',
    right_on='participant_ID',
    how='left'
)
print(f"\nUsers with survey data:  {merged['participant_ID'].notna().sum()}")
print(f"Users missing survey:    {merged['participant_ID'].isna().sum()}")

# ============================================================================
# 4. Categorize each demographic variable
# ============================================================================

# ── Income ──────────────────────────────────────────────────────────────────
def categorize_income(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    if s == 'prefer not to say': return 'Prefer not to say'
    if '4 000 chf or less'  in s or '4000 chf or less' in s: return '<4,000 CHF'
    if '4 001 - 8 000'      in s: return '4,001 – 8,000 CHF'
    if '8 001 - 12 000'     in s: return '8,001 – 12,000 CHF'
    if '12 001 - 16 000'    in s: return '12,001 – 16,000 CHF'
    if 'more than 16 000'   in s: return '>16,000 CHF'
    return 'Unknown'

# ── Age group ────────────────────────────────────────────────────────────────
def categorize_age(v):
    if pd.isna(v): return 'Unknown'
    try:
        age = int(float(str(v).strip()))
        if age < 18:              return '<18'
        elif age <= 24:           return '18–24'
        elif age <= 34:           return '25–34'
        elif age <= 44:           return '35–44'
        elif age <= 54:           return '45–54'
        elif age <= 64:           return '55–64'
        else:                     return '65+'
    except:
        return str(v).strip()     # keep raw label if already a string category

# ── Household size ────────────────────────────────────────────────────────────
def categorize_hh(v):
    if pd.isna(v): return 'Unknown'
    try:
        n = int(float(str(v).strip()))
        if n == 1:   return '1 person'
        elif n == 2: return '2 persons'
        elif n == 3: return '3 persons'
        elif n == 4: return '4 persons'
        else:        return '5 + persons'
    except:
        return str(v).strip()

# ── Gender ───────────────────────────────────────────────────────────────────
def clean_gender(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    if s in ('male', 'm', 'man'):         return 'Male'
    if s in ('female', 'f', 'woman'):     return 'Female'
    if 'non' in s or 'diverse' in s:      return 'Non-binary / Diverse'
    if 'prefer' in s:                     return 'Prefer not to say'
    return str(v).strip()

# Apply categorizations
if 'income'         in merged.columns: merged['income_cat']  = merged['income'].apply(categorize_income)
if 'age'            in merged.columns: merged['age_group']   = merged['age'].apply(categorize_age)
if 'household_size' in merged.columns: merged['hh_cat']      = merged['household_size'].apply(categorize_hh)
if 'gender'         in merged.columns: merged['gender_cat']  = merged['gender'].apply(clean_gender)

# ============================================================================
# 5. Build the statistics table
# ============================================================================

def make_freq_table(series, label):
    """Count + percentage for one categorical variable."""
    counts = series.value_counts(dropna=False)
    pct    = (counts / counts.sum() * 100).round(1)
    df = pd.DataFrame({
        'Category'  : counts.index.astype(str),
        'Count'     : counts.values,
        'Percentage': pct.values
    })
    df.insert(0, 'Variable', label)
    return df

tables = []

if 'gender_cat'  in merged.columns: tables.append(make_freq_table(merged['gender_cat'],  'Gender'))
if 'age_group'   in merged.columns: tables.append(make_freq_table(merged['age_group'],   'Age Group'))
if 'income_cat'  in merged.columns: tables.append(make_freq_table(merged['income_cat'],  'Household Income'))
if 'hh_cat'      in merged.columns: tables.append(make_freq_table(merged['hh_cat'],      'Household Size'))

stats_table = pd.concat(tables, ignore_index=True)
stats_table['Percentage'] = stats_table['Percentage'].apply(lambda x: f"{x}%")

# ============================================================================
# 6. Display and save
# ============================================================================
print("\n" + "="*55)
print("DEMOGRAPHIC STATISTICS FOR SELECTED USERS")
print("="*55)
print(stats_table.to_string(index=False))
print("="*55)
print(f"Total users in analysis: {len(merged)}")

stats_table.to_csv(output, index=False)
print(f"\n✓ Saved to {output}")