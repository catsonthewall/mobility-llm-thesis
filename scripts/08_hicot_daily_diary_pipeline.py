#!/usr/bin/env python3
"""
Everyday-life diaries -> sociodemographic inference
===================================================
A .py version of the geocontext-prompt notebook
(`03_geocontext_prompt_v3_template_S8_0319_age_fine_tuning_gpt-oss-20b.ipynb`),
with the POI redundancy taken out.

What this does
--------------
Reads `sp_toponym_poi_purpose_demographics.csv`, takes a window of consecutive
days per participant, and writes their week as a diary a person could actually
read: a short list of the places in their life, then what they did each day and
how they got there. A language model reads the record and names a gender, an
age group and a household income band.

The diary describes; it does not conclude. No percentiles, no high/low labels,
no class frequencies, no statement of what each group tends to look like.

Where the redundancy was
------------------------
The notebook wrote one self contained sentence per staypoint, each ending in
the raw `nearby_places` string. Three problems compounded:

   The same POI recurs within one string. A real row reads
    `0.129 km ...: Transportation Waldegg; 0.129 km ...: Transportation
    Waldegg; 0.13 km ...: Transportation Waldegg; ...` -- one bus stop, three
    times, before any other category appears.
  `Unknown` and `Others` dominate the POI table (2.9M and 159k rows against
    2.9k schools) and carry nothing.
  Every place is re-described on every visit. Somebody who sleeps at home
    each night gets the same address and the same POI list seven times in a
    seven-day diary.

So POIs are deduplicated by name and category, `Unknown`/`Others` are dropped,
distant ones are dropped, at most MAX_POIS survive per place -- and each place
is described once in a legend, after which the diary just names it. A week of
diary costs roughly a third of what the flat form cost.

Leakage policy
--------------
The source table holds survey answers and tracking observations side by side.
Only tracking-derived and spatially joined columns may enter a diary. Survey
answers are read in a separate pass that cannot reach the prompt builder; the
separation is enforced at read time by `usecols`, not by filtering afterwards.

Stages
------
    build       merged CSV        -> one diary window per participant, CSV
    verbalize   diary windows     -> prompts JSONL
    predict     prompts JSONL     -> checkpointed predictions JSONL
    parse       predictions JSONL -> clean predictions CSV
    evaluate    clean preds       -> metrics, confusion, bootstrap CI

Examples
--------
    python 17_daily_diary_pipeline_cot_0408_v1.py build --window-days 7
    python 17_daily_diary_pipeline_cot_0408_v1.py verbalize --sample-size 200 --dry-run
    python 17_daily_diary_pipeline_cot_0408_v1.py verbalize --sample-size 3 --dry-run
    python 17_daily_diary_pipeline_cot_0408_v1.py all --sample-size 200
    python 17_daily_diary_pipeline_cot_0408_v1.py all --sample-size 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

# =============================================================================
# <<< section: paths and global config >>>
# =============================================================================

MERGED_CSV = Path(os.environ.get(
    'MOBILITY_MERGED_CSV',
    '/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv',
))
OUT_DIR = Path(os.environ.get(
    'MOBILITY_OUT_DIR', '/data/baliu/thesis/02_merged_data/'
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = 'diary_v1'
WINDOW_OUT = OUT_DIR / f'windows_{TAG}.csv'
PROMPTS_OUT = OUT_DIR / f'prompts_{TAG}.jsonl'
PREDS_OUT = OUT_DIR / f'preds_{TAG}.jsonl'
PREDS_CLEAN = OUT_DIR / f'preds_{TAG}_clean.csv'
MERGED_OUT = OUT_DIR / f'merged_{TAG}.csv'
METRICS_OUT = OUT_DIR / f'metrics_{TAG}.csv'

USER_COL = 'user_id'
SEED = 42

# --- diary shape -------------------------------------------------------------
WINDOW_DAYS = 7          # consecutive days per participant
MIN_STAYS = 5            # skip participants with almost no records
MAX_STAYS_PER_DAY = 12   # a day with more than this is GPS noise, not a life
MAX_POIS = 3             # per place, after deduplication
MAX_POI_KM = 0.4         # a POI further than this is not "nearby"
MIN_STAY_MIN = 10        # shorter stays are pass-throughs
MAX_TRIP_HOURS = 4       # a longer gap is missing data, so its times are not stated

# --- model -------------------------------------------------------------------
BACKEND = os.environ.get('LLM_BACKEND', 'ollama')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
MODEL_NAME = os.environ.get('LLM_MODEL', 'gpt-oss:20b')
HF_MODEL_DIR = os.environ.get('HF_MODEL_DIR', '')
NUM_CTX = 8192
NUM_PREDICT = 2048
REQ_TIMEOUT = 300
PREDICT_RETRIES = 3

# Both defaults below were set from `probe_ollama.py` against gpt-oss:20b on a
# 6.6k-character diary prompt. They are not arbitrary:
#
#   setting                   resp  think   eval  done_reason
#   plain, 4096                  0  20121   4096  length      <- reasons forever
#   think=False, 2048            0   7745   2048  length      <- False ignored
#   think=low, 1024            235    206    112  stop        <- works
#   schema, 4096                 0      0     70  stop        <- emits nothing
#
# `think='low'` is what caps the reasoning channel; omitting the parameter or
# passing False lets it run until the budget is gone and `response` is empty.
# JSON-schema decoding returns nothing at all for this model, so it is off by
# default; the parser reads free-text JSON and prose anyway.
THINK_LEVEL = os.environ.get('LLM_THINK', 'low')   # 'low'|'medium'|'high'|'default'
USE_SCHEMA = os.environ.get('LLM_SCHEMA', '0') != '0'

# --- role ---------------------------------------------------------------------
# Prepended to every prompt. Kept as a constant so the wording is easy to change
# and easy to report, and overridable per run so the persona itself can be
# ablated: `--role ""` sends the diary with no role line at all.
ROLE = os.environ.get(
    'LLM_ROLE',
    'You are a senior researcher in transport geography and travel-behaviour '
    'analysis. You have spent years reading GNSS mobility diaries alongside '
    'Swiss household travel survey data, and you are used to judging what a '
    "person's daily movement implies about who they are."
)

USE_LANGUAGE = True      # survey language ~ language region, recoverable from home
# =============================================================================
# <<< section: leakage control >>>
# =============================================================================

OBSERVED_COLS = [
    'user_id', 'started_at', 'finished_at', 'location_id', 'mode', 'length_km',
    'act_duration_h', 'date', 'lon', 'lat',
    'road', 'neighbourhood', 'city', 'nearby_places',
    'act_imputed_purpose', 'act_CH_BEZ_D', 'act_CH_CODE_HN',
]
if USE_LANGUAGE:
    OBSERVED_COLS.append('language')

BLOCKED_COLS = {
    'age', 'gender', 'income', 'education', 'main_employment',
    'household_size', 'citizen_1',
    'work_status_employed', 'work_status_student', 'work_status_retired',
    'workload_jobs_main', 'postcode_jobs_main', 'postcode_home',
    'own_vehicles_car', 'own_vehicles_bicycle', 'car_fuel', 'car_size',
    'pt_pass_ga', 'pt_pass_half_fare',
    'freq_cardriver_own_car', 'freq_pt_train', 'freq_pt_local_pt',
    'freq_bike_own_bike', 'finished_tracking_study',
}
assert not (set(OBSERVED_COLS) & BLOCKED_COLS), 'leakage: observed/blocked overlap'

# =============================================================================
# <<< section: label spaces >>>
# =============================================================================
# The feature-based chapter bins age as 18-24 / 25-44 / 45-65. The earlier LLM
# script used 45-66, which silently moves both the baseline and the matched
# count. 65 keeps the two comparable.
AGE_TOP = int(os.environ.get('AGE_TOP', '65'))

GENDER_CATS = ['male', 'female']
AGE_CATS = ['18-24', '25-44', f'45-{AGE_TOP}']
INCOME_CATS = ['<4000', '4001-8000', '8001-12000', '12001-16000', '>16000']

TARGETS = {
    'gender': GENDER_CATS,
    'age_group': AGE_CATS,
    'income_level': INCOME_CATS,
}
# Age and income are ordered scales, so a neighbouring miss is a better answer
# than a distant one and `ordinal_mae` is meaningful. Gender is not ordered.
ORDINAL_TARGETS = {'age_group', 'income_level'}

TARGET_QUESTION = {
    'gender': "the person's most likely gender",
    'age_group': "the person's most likely age group",
    'income_level': "the household's most likely gross monthly income, in CHF",
}

# =============================================================================
# <<< section: vocabularies >>>
# =============================================================================

# `walked 0.9 km` reads better than `traveled 0.9 km on foot`, so active modes
# get their own verb and everything else takes `traveled ... by ...`.
MODE_VERB = {'walk': 'walked', 'bike': 'cycled'}
MODE_BY = {'car': 'by car', 'rail': 'by public transport (train or tram)',
           'bus': 'by public transport (bus)', 'other': ''}

PURPOSE_PHRASE = {
    'home': 'their home', 'work': 'their workplace',
    'shopping': 'a shopping location', 'leisure': 'a leisure location',
    'education': 'an educational location', 'errand': 'an errand location',
}

CONNECTIVES = ['Afterwards,', 'Later,', 'After that,', 'Then']

# Published facts about Switzerland, not statistics from this study's labels.
# They exist because the model reasons "owns a car and commutes 30 km, therefore
# wealthy", which in Switzerland is a statement about almost everybody: it
# predicted the top income band for 44% of participants against a true 11%, and
# read "is employed" as evidence for 25-44 against a true 42%. Withhold with
# --no-context to measure how much of the calibration comes from this block.
# The two-source structure from the notebook's cell 22 ("use only the
# information explicitly provided in the mobility record" + "based on swiss
# census data"), made explicit. There is no route from a clock time to an
# income band without population knowledge, so the fix is not to forbid that
# knowledge but to separate it from the reading and make both auditable.

# Each step answers a failure seen in the free-form version:
#   1  the model latched onto one cue and ignored the rest of the week, and
#      different samples of the same person latched onto different cues
#   2  it invented facts ("drops off a child at a kindergarten") that the
#      record does not contain -- quoting the text makes that checkable
#   3  its priors were doing the work while staying unstated
#   4  "typical of a working adult" was offered as evidence for 25-44 when it
#      separates nothing, since 45-65 works too
#   5  it produced a confident label where it had said there was no signal
COT_BLOCK = """## How to work
Go through these steps in order, briefly, then give the JSON.

Step 1 - Read off the record. List four or five things that can point to in
the text: usual departure and return times, commute distance and mode, how
far they go at weekends, which kinds of places recur, whether the week is
regular or irregular. Quote the clock times and distances. Do not interpret
anything yet.

Step 2 - Take each fact in turn and say what it would imply, naming the
population regularity you are relying on ("in Switzerland, ..."). Keep this
separate from Step 1: Step 1 is what the record says, Step 2 is what you
know about the population.

Step 3 - Drop every fact that fits all the categories equally. Working
regular hours does not separate 25-44 from 45-65. Owning a car does not
separate income bands. Say which factsdropped.

Step 4 - Weigh what survives. If nothing survives, say so, and let the
probabilities stay close to even rather than picking a category anyway.

Then, on the final line and with nothing after it, the JSON object."""

# Asking the model to use "only the record" cannot be obeyed literally -- there
# is no route from a clock time to an income band without outside knowledge --
# but it is a real prompt-design condition worth measuring rather than assuming.
# It also lets the model decline to guess, which the default prompt forbids.
STRICT_BLOCK = '''## Restriction
Work only from what the record states. Do not fall back on general
associations about which sorts of people travel which ways. Where the record
does not separate two categories, say so in the evidence and divide the
probability between them rather than picking one on a hunch.'''

SWISS_CONTEXT = '''## Reference points
These describe Switzerland in general. They say nothing about this person.
- Median gross household income is roughly CHF 10,000 per month. Above
  CHF 16,000 is about the top tenth of households; most people are not there.
- Around four in five households own a car, and a 20-30 km commute is
  unremarkable. Neither is a sign of wealth.
- People hold full-time jobs across the whole 25-65 range, so working regular
  hours does not narrow the age down.
- Both men and women commute long distances by car, and both do the school
  run.'''
MODE_GROUPS = {
    'car': {'car', 'car_driver', 'car_passenger', 'motorbike', 'taxi', 'ecar'},
    'rail': {'train', 'rail', 'tram', 'lightrail', 'funicular', 'cablecar'},
    'bus': {'bus', 'coach', 'trolleybus'},
    'bike': {'bicycle', 'bike', 'ebicycle', 'ebike'},
    'walk': {'walk', 'walking', 'foot'},
}

ZONE_LABEL = {
    'wohnzonen': 'residential area',
    'arbeitszonen': 'industrial or office area',
    'mischzonen': 'mixed residential and commercial area',
    'zentrumszonen': 'town centre',
    'zonen fur offentliche nutzungen': 'public-facility area',
    'eingeschrankte bauzonen': 'restricted building area',
    'tourismus- und freizeitzonen': 'tourism and leisure area',
    'verkehrszonen innerhalb der bauzonen': 'transport area',
}
ZONE_CODE_LABEL = {
    11: 'residential area', 12: 'industrial or office area',
    13: 'mixed residential and commercial area', 14: 'town centre',
    15: 'public-facility area', 16: 'restricted building area',
    17: 'tourism and leisure area', 18: 'transport area',
}

# `Unknown` (2.9M rows) and `Others` (159k) crowd out everything informative,
# so they never reach a diary.
POI_PHRASE = {
    'Shopping': 'shop', 'Entertainment': 'cafe or restaurant',
    'Transportation': 'public transport stop', 'Services': 'office or bank',
    'Schools': 'school', 'Civic': 'civic building',
    'Residential': 'residential building',
}
POI_DROP = {'unknown', 'others', 'other', 'building', 'buildings'}

LANG_LABEL = {'DE': 'German-speaking', 'FR': 'French-speaking',
              'IT': 'Italian-speaking'}

DOW_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
             'Saturday', 'Sunday']


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = str(s).strip().lower()
    for a, b in (('ä', 'a'), ('ö', 'o'), ('ü', 'u'), ('é', 'e'), ('è', 'e'),
                 ('à', 'a'), ('ç', 'c'), ('ß', 'ss')):
        s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s)


def _mode_group(m) -> str:
    m = _norm(m).replace(' ', '_')
    for g, members in MODE_GROUPS.items():
        if m in members:
            return g
    return 'other'


def _zone_label(bez, code) -> str | None:
    z = ZONE_LABEL.get(_norm(bez))
    if z:
        return z
    try:
        return ZONE_CODE_LABEL.get(int(float(code)))
    except (TypeError, ValueError):
        return None


def _article(noun: str) -> str:
    return f'{"an" if noun[:1].lower() in "aeiou" else "a"} {noun}'


def _series(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def _clean(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ('nan', 'none', '-', '') else None


# =============================================================================
# <<< section: POI cleaning >>>
# =============================================================================
# `0.129 km to the south-east: Transportation Waldegg; 0.13 km to the
# south-east: Transportation Waldegg; ...`
_POI_ENTRY = re.compile(
    r'([\d.]+)\s*km\s+to\s+the\s+[\w-]+\s*:\s*([A-Za-z]+)\s*(.*)', re.I)


def clean_pois(raw, max_pois: int = MAX_POIS,
               max_km: float = MAX_POI_KM) -> list[str]:
    """Nearby POIs as a short list of distinct things, nearest first.

    The raw field repeats the same POI several times at near-identical
    distances and is dominated by Unknown/Others. This keeps one entry per
    named thing, one per unnamed category, and at most `max_pois` of them.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    seen_names: set[str] = set()
    seen_cats: set[str] = set()
    out: list[tuple[float, str]] = []

    for part in raw.split(';'):
        m = _POI_ENTRY.search(part.strip())
        if not m:
            continue
        try:
            dist = float(m.group(1))
        except ValueError:
            continue
        if dist > max_km:
            continue

        cat = m.group(2).strip().capitalize()
        if cat.lower() in POI_DROP or cat not in POI_PHRASE:
            continue
        name = _clean(m.group(3))
        if name:
            name = re.sub(r'\s*\([^)]*\)', '', name).strip()

        key = _norm(name) if name else ''
        if key and key in seen_names:          # same place listed again
            continue
        if not key and cat in seen_cats:       # second unnamed shop, etc.
            continue
        if key:
            seen_names.add(key)
        seen_cats.add(cat)

        bare = POI_PHRASE[cat]
        label = f'{name} ({bare})' if name else _article(bare)
        out.append((dist, label))

    out.sort(key=lambda t: t[0])
    return [label for _, label in out[:max_pois]]


# =============================================================================
# <<< section: ground truth (isolated ) >>>
# =============================================================================

def _age_to_group(age) -> str | None:
    age = pd.to_numeric(age, errors='coerce')
    if pd.isna(age) or age < 18 or age > AGE_TOP:
        return None
    if age <= 24:
        return '18-24'
    if age <= 44:
        return '25-44'
    return f'45-{AGE_TOP}'


def _income_to_level(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower().replace('chf', '').replace(',', '').replace(' ', '')
    if 'prefer' in s or 'notsay' in s or s in ('', 'nan'):
        return None
    if 'morethan' in s or s.startswith('>'):
        return '>16000'
    if 'less' in s or s.startswith('<'):
        return '<4000'
    nums = [int(n) for n in re.findall(r'\d+', s.replace('–', '-'))]
    if not nums:
        return None
    mid = sum(nums[:2]) / 2 if len(nums) >= 2 else nums[0]
    for cut, lab in ((4000, '<4000'), (8000, '4001-8000'), (12000, '8001-12000'),
                     (16000, '12001-16000')):
        if mid <= cut:
            return lab
    return '>16000'


def _gender_to_cat(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ('1', 'm', 'male', 'mann', 'homme'):
        return 'male'
    if s in ('2', 'f', 'female', 'frau', 'femme'):
        return 'female'
    return None


def _first_non_null(series: pd.Series):
    v = series.dropna()
    return v.iloc[0] if not v.empty else None


def load_ground_truth() -> pd.DataFrame:
    """Read only the survey target columns. Never called by the prompt builder."""
    if not MERGED_CSV.exists():
        sys.exit(f'Missing {MERGED_CSV}')

    header = pd.read_csv(MERGED_CSV, nrows=0)
    id_col = 'participant_ID' if 'participant_ID' in header.columns else USER_COL
    gcol = next((c for c in ('gender', 'sex') if c in header.columns), None)
    usecols = [id_col] + [c for c in ('age', 'income') if c in header.columns]
    if gcol:
        usecols.append(gcol)

    gt = pd.read_csv(MERGED_CSV, usecols=list(dict.fromkeys(usecols)),
                     dtype={id_col: str}, low_memory=False)

    out = pd.DataFrame({USER_COL: gt[id_col].astype(str).str.strip()})
    out['age_group'] = gt['age'].apply(_age_to_group) if 'age' in gt.columns else None
    out['income_level'] = gt['income'].apply(_income_to_level) if 'income' in gt.columns else None
    out['gender'] = gt[gcol].apply(_gender_to_cat) if gcol else None
    if gcol is None:
        print('  Warning: no gender/sex column found; gender will not be scored.')
    out = out.loc[out[USER_COL].ne('') & out[USER_COL].ne('nan')]

    n0 = len(out)
    out = out.groupby(USER_COL, as_index=False, sort=False)[list(TARGETS)].agg(_first_non_null)
    if n0 != len(out):
        print(f'  Ground truth collapsed {n0:,} rows -> {len(out):,} participants.')
    return out

# =============================================================================
# <<< section: stage 1 — pick one diary window per participant >>>
# =============================================================================

def _pick_window(dates: np.ndarray, n_days: int,
                 rng: np.random.Generator) -> set:
    """A random run of `n_days` consecutive tracked days, as the notebook did.

    Falls back to the densest available days when no unbroken run exists.
    """
    d = np.sort(pd.to_datetime(pd.Series(dates)).dt.normalize().unique())
    if len(d) == 0:
        return set()
    if len(d) <= n_days:
        return set(pd.Timestamp(x) for x in d)

    gaps = np.diff(d).astype('timedelta64[D]').astype(int)
    starts = []
    for i in range(len(d) - n_days + 1):
        if gaps[i:i + n_days - 1].max() == 1:
            starts.append(i)
    if starts:
        i = int(rng.choice(starts))
        return set(pd.Timestamp(x) for x in d[i:i + n_days])
    i = int(rng.integers(0, len(d) - n_days + 1))
    return set(pd.Timestamp(x) for x in d[i:i + n_days])


def stage_build(args) -> None:
    """Chunked pass over the merged table -> the diary window for each person."""
    print('=' * 70)
    print('Stage 1 — select diary windows')
    print('=' * 70)

    if not MERGED_CSV.exists():
        sys.exit(f'Missing merged table: {MERGED_CSV}')

    header = pd.read_csv(MERGED_CSV, nrows=0)
    present = [c for c in OBSERVED_COLS if c in header.columns]
    absent = [c for c in OBSERVED_COLS if c not in header.columns]
    if absent:
        print(f'  Note: {len(absent)} observed columns absent: {absent}')
    print(f'  Reading {len(present)} observed columns; withholding '
          f'{len(BLOCKED_COLS & set(header.columns))} survey columns.')

    parts: list[pd.DataFrame] = []
    n_rows = 0
    t0 = time.time()
    reader = pd.read_csv(MERGED_CSV, usecols=present,
                         dtype={USER_COL: str, 'location_id': str},
                         chunksize=args.chunksize, low_memory=False)
    for ci, chunk in enumerate(reader, start=1):
        n_rows += len(chunk)
        chunk[USER_COL] = chunk[USER_COL].astype(str).str.strip()
        chunk = chunk.loc[chunk[USER_COL].ne('') & chunk[USER_COL].ne('nan')]
        if chunk.empty:
            continue
        chunk['started_at'] = pd.to_datetime(chunk['started_at'], errors='coerce',
                                             utc=True)
        chunk['finished_at'] = pd.to_datetime(chunk.get('finished_at'),
                                              errors='coerce', utc=True)
        chunk = chunk.dropna(subset=['started_at'])
        for c in ('act_duration_h', 'length_km', 'lon', 'lat'):
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk['day'] = chunk['started_at'].dt.tz_localize(None).dt.normalize()
        parts.append(chunk)
        if ci % 4 == 0:
            print(f'  ...{n_rows:,} rows in {time.time() - t0:.0f}s')

    ev = pd.concat(parts, ignore_index=True)
    del parts
    print(f'  Read {n_rows:,} rows in {time.time() - t0:.0f}s; '
          f'{ev[USER_COL].nunique():,} participants.')

    rng = np.random.default_rng(args.seed)
    keep: list[pd.DataFrame] = []
    dropped = 0
    for uid, g in ev.groupby(USER_COL, sort=False):
        window = _pick_window(g['day'].to_numpy(), args.window_days, rng)
        sub = g.loc[g['day'].isin(window)]
        if len(sub) < MIN_STAYS:
            dropped += 1
            continue
        keep.append(sub)

    win = pd.concat(keep, ignore_index=True).sort_values([USER_COL, 'started_at'])
    win.to_csv(WINDOW_OUT, index=False)
    print(f'  {dropped} participants dropped for fewer than {MIN_STAYS} stays '
          f'in the window.')
    print(f'  Saved {len(win):,} stays for {win[USER_COL].nunique():,} '
          f'participants ({args.window_days}-day windows) -> {WINDOW_OUT}')

# =============================================================================
# <<< section: stage 2 — write the diary >>>
# =============================================================================

def _hhmm(t) -> str | None:
    if t is None or pd.isna(t):
        return None
    return pd.Timestamp(t).tz_localize(None).strftime('%H:%M')


def _travel_phrase(mode, km) -> str | None:
    """`walked 0.9 km` / `traveled 4.2 km by public transport`."""
    grp = _mode_group(mode)
    dist = f'{float(km):.1f} km' if pd.notna(km) and float(km) >= 0.1 else None
    verb = MODE_VERB.get(grp)
    if verb:                                   # walked / cycled
        return f'{verb} {dist}' if dist else verb
    by = MODE_BY.get(grp)
    if not by:
        return f'traveled {dist}' if dist else None
    return f'traveled {dist} {by}' if dist else f'traveled {by}'


def _place_phrase(row) -> str:
    """`their workplace in Oerlikon`, `a shopping location on Bahnhofstrasse`."""
    purpose = _norm(getattr(row, 'act_imputed_purpose', None))
    base = PURPOSE_PHRASE.get(purpose, 'a location')

    road = _clean(getattr(row, 'road', None))
    hood = _clean(getattr(row, 'neighbourhood', None))
    city = _clean(getattr(row, 'city', None))

    if road:
        where = f'on {road}'
        if city:
            where += f' in {city}'
    elif hood:
        where = f'in {hood}'
        if city and city != hood:
            where += f', {city}'
    elif city:
        where = f'in {city}'
    else:
        where = ''
    return f'{base} {where}'.strip()


def build_narrative(g: pd.DataFrame) -> str:
    """The week as continuous prose, in the manner of Prompt A.

    Every clock time is stated and nothing is aggregated: how much of the week
    goes to work, how far they range, how regular they are, all of it stays
    implicit in the timestamps for the model to read out for itself.

    A place is described in full the first time it appears, including what is
    nearby; on later visits it is named but not re-described, which is what
    keeps a seven-day narrative from repeating one address seven times.
    """
    g = g.sort_values('started_at').copy()
    g['prev_finished'] = g['finished_at'].shift(1)

    # short records are pass-throughs, dropped after the travel windows are
    # fixed so a discarded stay cannot stretch the trip that follows it
    dur_min = (g['finished_at'] - g['started_at']).dt.total_seconds() / 60
    g = g.loc[dur_min.isna() | (dur_min >= MIN_STAY_MIN)]

    described: set[str] = set()
    paragraphs: list[str] = []

    for day, dg in g.groupby('day', sort=True):
        dg = dg.head(MAX_STAYS_PER_DAY)
        ts = pd.Timestamp(day)
        opener = f'On {DOW_NAMES[ts.dayofweek]} {ts.strftime("%d %B")}'

        sentences: list[str] = []
        prev_loc = None
        for i, r in enumerate(dg.itertuples(index=False)):
            loc = getattr(r, 'location_id', None)
            if loc == prev_loc:
                continue
            prev_loc = loc

            # travel: from the end of the previous stay to the start of this one
            trip = _travel_phrase(getattr(r, 'mode', None),
                                  getattr(r, 'length_km', None))
            prev_fin = getattr(r, 'prev_finished', None)
            gap_h = ((r.started_at - prev_fin).total_seconds() / 3600
                     if pd.notna(prev_fin) else None)
            t_from, t_to = _hhmm(prev_fin), _hhmm(r.started_at)
            leg = ''
            if trip:
                leg = trip
                if (t_from and t_to and t_from != t_to
                        and gap_h is not None and 0 <= gap_h <= MAX_TRIP_HOURS):
                    leg += f' from {t_from} to {t_to}'

            # stay
            place = _place_phrase(r)
            s_from, s_to = _hhmm(r.started_at), _hhmm(getattr(r, 'finished_at', None))
            stay = f'stayed at {place}'
            if s_from and s_to:
                stay += f' from {s_from} to {s_to}'
            elif s_from:
                stay += f' from {s_from}'

            # nearby detail, first visit only
            if loc not in described:
                described.add(loc)
                pois = clean_pois(getattr(r, 'nearby_places', None))
                if pois:
                    stay += f', near {_series(pois)}'

            if not sentences:
                body = f'{opener}, the user ' + (f'{leg}, then {stay}' if leg else stay)
            else:
                lead = CONNECTIVES[(len(sentences) - 1) % len(CONNECTIVES)]
                body = f'{lead} the user ' + (f'{leg} and {stay}' if leg else stay)
            sentences.append(body + '.')

        if not sentences:
            sentences.append(f'{opener}, no movement was recorded.')
        paragraphs.append(' '.join(sentences))

    return '\n\n'.join(paragraphs)


def build_description(g: pd.DataFrame) -> str:
    region = None
    if 'language' in g.columns and g['language'].notna().any():
        region = LANG_LABEL.get(str(g['language'].dropna().iloc[0]).strip().upper())

    head = f'Record of {g["day"].nunique()} consecutive tracked days'
    head += f' for one user in {region} Switzerland.' if region else ' for one user.'
    return f'{head}\n\n{build_narrative(g)}'


def build_prompt(desc: str, targets: list[str], role: str = ROLE,
                 context: bool = True, strict: bool = False,
                 cot: bool = True) -> str:
    """State who is reading, describe a life, ask a question."""
    cat_lines = '\n'.join(f'  - {t}: {json.dumps(TARGETS[t])}' for t in targets)
    ask = '\n'.join(f'  {i}. {TARGET_QUESTION[t]}'
                    for i, t in enumerate(targets, start=1))
    # A wrapper keyed by target name is redundant when only one is asked for,
    # and every extra level of nesting is another brace the model can drop.
    if len(targets) == 1:
        cats = TARGETS[targets[0]]
        probs = ', '.join(f'"{c}": 0.0' for c in cats)
        shape = f'"label": "...", "probabilities": {{{probs}}}, "evidence": "..."'
    else:
        shape = ', '.join(
            f'"{t}": {{"label": "...", "probabilities": {{...}}}}'
            for t in targets) + ', "evidence": "..."'

    # In strict mode the model is told to split probability when it cannot
    # separate two categories, so demanding a confident pick would contradict
    # that. A label is still required for scoring; the uncertainty lives in the
    # probability vector instead.
    commit_rule = (
        '- Still give the single most likely label, but let the probabilities\n'
        '  carry how unsure you are.'
        if strict else
        '- Commit to one category per attribute even where the record is\n'
        '  ambiguous.')

    # only worth saying when an ordered attribute is actually being asked for
    ordered = [t for t in targets if t in ORDINAL_TARGETS]
    if ordered:
        which = ('Age group and income level are ordered scales' if len(ordered) > 1
                 else f'{ordered[0].replace("_", " ").capitalize()} is an '
                      'ordered scale')
        ordinal_rule = (
            f'- {which}, so a near miss is better than a\n'
            '  wild one. Put weight on neighbouring categories when the record\n'
            '  is ambiguous.\n')
    else:
        ordinal_rule = ''

    # dedent runs on the template, never on the substituted text: injected
    # blocks start at column 0 and would otherwise flatten the common prefix.
    template = dedent("""
    Below is a written record of several consecutive days in one anonymous
    person's life, reconstructed from GNSS tracking in Switzerland. It gives
    the times they set out, how far and by what means they travelled, where
    they stopped and for how long, and what stands near those places. No
    survey answer about them is included.

    ## Task
    Reading this as an account of someone's ordinary days, infer:
    {ask}

    ## Categories (copy verbatim)
    {cat_lines}

    ## Rules
    - Go on what the record says. A school near a place they stopped does not
      mean they have children; an office nearby does not mean they work there.
      Do not settle on an occupation, a family situation or a background first
      and then reason from that.
    - Nothing is summarised for . Working hours, routine, range and the
      rhythm of the week are all there in the clock times; read them off the
      record yourself.
    - Before you answer, ask what in this record argues for each category in
      turn, including the ones you are about to reject.
    - evidence must name something that separates the category you chose
      from its neighbours. "Commutes to work" does not separate 25-44 from
      45-65, because both do; "leaves at 04:27" might.
    {ordinal_rule}
    {commit_rule}

    ## Output
    End with a single JSON object on its own final line, giving for each
    attribute a probability for every category (summing to 1) and the single
    most likely label:
    {{{shape}}}
    "evidence" is one sentence, at most 30 words, on what in the record decided
    it. Close every brace you open.
    """).strip()

    instructions = (template
                    .replace('{ask}', ask)
                    .replace('{cat_lines}', cat_lines)
                    .replace('{ordinal_rule}\n', ordinal_rule)
                    .replace('{commit_rule}', commit_rule)
                    .replace('{{{shape}}}', '{' + shape + '}'))

    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    # the two blocks contradict each other, so strict wins and context is dropped
    extra = f'\n\n{STRICT_BLOCK}' if strict else (
        f'\n\n{SWISS_CONTEXT}' if context else '')
    steps = f'\n\n{COT_BLOCK}' if cot else ''
    return f'{head}{instructions}{extra}{steps}\n\n## The record\n{desc}'


# =============================================================================
# <<< section: hierarchical chain of thought (HiCoTraj-style) >>>
# =============================================================================
# Three chained calls instead of four steps inside one prompt. The difference
# matters: asked to "work through the steps" in a single prompt, the model
# writes the headings and still jumps to its prior -- every sample of CVTOW
# reached 25-44 via "typical of a working adult", which separates nothing.
# Chaining forces each stage to produce an artefact that the next one consumes,
# and Stage 1's output can be checked against the diary for invented facts.
#
# Stages 1 and 2 say nothing about gender, age or income, so they are computed
# once per person and shared by all three targets: 1 + 1 + 3*k calls rather
# than 3*(3*k). Stage 3 never sees the diary, only the two summaries, which is
# what stops it re-reading the raw text and skipping the chain.

STAGE1_TASK = """## Task
Extract the following from the record, without interpreting any of it. Copy
figures and clock times straight from the text. Write "not stated" where the
record does not say.

1. PLACES
   - each distinct place: what the record calls it, its area type, what
     stands nearby
   - how many separate visits to each, and roughly how long a visit lasts

2. TIMING
   - usual time of first departure on weekdays, and of the final return
   - the same for weekend days
   - any activity before 06:00 or after 22:00

3. TRAVEL
   - which modes appear, and the distances covered by each
   - the home-to-workplace distance
   - the longest single trip of the week

4. REGULARITY
   - which days repeat the same shape and which do not
   - whether the same places recur or new ones keep appearing

State no implications. Do not name an occupation, a family situation, an age,
a gender or an income. Facts only."""

STAGE2_TASK = """## Task
Below is a factual summary of one person's tracked week. Describe what this
pattern of movement is like as a way of living.

1. SCHEDULE - fixed hours, flexible hours, shift work, irregular, or no
   work-like pattern at all. Say which, and on what basis.
2. ACTIVITY SPACE - how far their ordinary life reaches, and whether it is
   concentrated or spread out.
3. TRANSPORT DEPENDENCE - what they rely on to get about, and whether they
   appear to have a choice of mode.
4. SHAPE OF THE WEEK - how far weekends differ from weekdays, and how much of
   the week is discretionary rather than obligatory.
5. WHAT IS MISSING - patterns you would expect in a week like this and do not
   find. Absence is evidence too.

Describe the way of living only. Do not name an age, a gender, an income, an
occupation or a family situation. That comes later."""


def build_stage1_prompt(desc: str, role: str = ROLE) -> str:
    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    intro = ('Below is a written record of several consecutive days in one\n'
             'anonymous person\'s life, reconstructed from GNSS tracking in\n'
             'Switzerland. No survey answer about them is included.')
    return f'{head}{intro}\n\n{STAGE1_TASK}\n\n## The record\n{desc}'


def build_stage2_prompt(stage1: str, role: str = ROLE) -> str:
    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    return f'{head}{STAGE2_TASK}\n\n## Factual summary\n{stage1}'


def build_stage3_prompt(stage1: str, stage2: str, targets: list[str],
                        role: str = ROLE, context: bool = True,
                        strict: bool = False) -> str:
    cat_lines = '\n'.join(f'  - {t}: {json.dumps(TARGETS[t])}' for t in targets)
    ask = '\n'.join(f'  {i}. {TARGET_QUESTION[t]}'
                    for i, t in enumerate(targets, start=1))
    if len(targets) == 1:
        cats = TARGETS[targets[0]]
        probs = ', '.join(f'"{c}": 0.0' for c in cats)
        shape = f'"label": "...", "probabilities": {{{probs}}}, "evidence": "..."'
    else:
        shape = ', '.join(f'"{t}": {{"label": "...", "probabilities": {{...}}}}'
                          for t in targets) + ', "evidence": "..."'

    ordered = [t for t in targets if t in ORDINAL_TARGETS]
    if ordered:
        which = ('Age group and income level are ordered scales'
                 if len(ordered) > 1 else
                 f'{ordered[0].replace("_", " ").capitalize()} is an ordered scale')
        ordinal_rule = (f'- {which}, so a near miss beats a wild one. Put weight\n'
                        '  on neighbouring categories where the summaries are\n'
                        '  ambiguous.\n')
    else:
        ordinal_rule = ''

    template = dedent("""
    You have already read one person's tracked week and produced two summaries
    of it. Use them, and nothing else, to answer.

    ## Task
    Infer:
    {ask}

    ## Categories (copy verbatim)
    {cat_lines}

    ## Rules
    - Work from the two summaries. Do not invent detail that is in neither.
    - Discard anything that fits every category equally well. Working regular
      hours does not separate 25-44 from 45-65; owning a car does not separate
      income bands. Name what you discarded.
    - Your evidence must cite something from the summaries that tells the
      chosen category apart from its neighbours.
    {ordinal_rule}
    - If nothing separates two categories, keep the probabilities close to even
      rather than picking one anyway. Still give the most likely label.

    ## Output
    End with a single JSON object on its own final line:
    {{{shape}}}
    "evidence" is one sentence, at most 30 words. Close every brace you open.
    """).strip()

    instructions = (template
                    .replace('{ask}', ask)
                    .replace('{cat_lines}', cat_lines)
                    .replace('{ordinal_rule}\n', ordinal_rule)
                    .replace('{{{shape}}}', '{' + shape + '}'))

    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    extra = f'\n\n{STRICT_BLOCK}' if strict else (
        f'\n\n{SWISS_CONTEXT}' if context else '')
    return (f'{head}{instructions}{extra}'
            f'\n\n## Factual summary\n{stage1}'
            f'\n\n## Behavioural reading\n{stage2}')


def response_schema(targets: list[str]) -> dict:
    props: dict = {'evidence': {'type': 'string'}}
    for t in targets:
        cats = TARGETS[t]
        props[t] = {
            'type': 'object',
            'properties': {
                'probabilities': {
                    'type': 'object',
                    'properties': {c: {'type': 'number'} for c in cats},
                    'required': list(cats),
                },
                'label': {'type': 'string', 'enum': list(cats)},
            },
            'required': ['probabilities', 'label'],
        }
    return {'type': 'object', 'properties': props,
            'required': ['evidence'] + list(targets)}


def stage_verbalize(args) -> None:
    print('=' * 70)
    print('Stage 2 — write diaries')
    print('=' * 70)

    if not WINDOW_OUT.exists():
        sys.exit(f'Missing {WINDOW_OUT}. Run the build stage first.')

    win = pd.read_csv(WINDOW_OUT, dtype={USER_COL: str, 'location_id': str},
                      low_memory=False)
    win[USER_COL] = win[USER_COL].astype(str).str.strip()
    win['started_at'] = pd.to_datetime(win['started_at'], errors='coerce', utc=True)
    win['finished_at'] = pd.to_datetime(win.get('finished_at'), errors='coerce',
                                        utc=True)
    win['day'] = pd.to_datetime(win['day'], errors='coerce')
    win = win.dropna(subset=['started_at', 'day'])

    gt = load_ground_truth()
    labelled = set(gt[USER_COL])
    users = [u for u in win[USER_COL].unique() if u in labelled]
    print(f'  {win[USER_COL].nunique():,} participants with a window; '
          f'{len(users):,} of them labelled.')

    rng = np.random.default_rng(args.seed)
    if args.sample_size is not None and args.sample_size < len(users):
        users = sorted(rng.choice(users, size=args.sample_size, replace=False))
    else:
        users = sorted(users)
    if args.limit:
        users = users[:args.limit]
    print(f'  Writing diaries for {len(users)} participants (seed {args.seed}).')
    mode = ('strict-record (no outside knowledge offered)' if args.strict_record
            else 'Swiss reference points' if args.context else 'bare')
    print(f'  Framing: {mode}; reasoning steps: {"on" if args.cot else "off"}')
    print(f'  Role: {"none" if not args.role.strip() else args.role.strip()[:60] + "..."}')

    target_sets = ([[t] for t in TARGETS] if args.target_mode == 'per_target'
                   else [list(TARGETS)])

    groups = {u: g.sort_values('started_at')
              for u, g in win.groupby(USER_COL, sort=False)}

    records = []
    n_places = []
    for uid in users:
        g = groups[uid]
        desc = build_description(g)
        n_places.append(g['location_id'].nunique())
        for ts in target_sets:
            prompt = build_prompt(desc, ts, args.role, args.context,
                                  args.strict_record, args.cot)
            records.append({
                'uid': uid,
                'targets': ts,
                'desc': desc,          # hierarchical mode builds its own prompts
                'prompt': prompt,
                'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16],
            })

    if args.dry_run:
        print('\n--- sample diary ---\n')
        print(records[0]['prompt'])
        print('\n--- end ---')
        lens = [len(r['prompt']) for r in records]
        print(f'\n  chars: min {min(lens)} median {int(np.median(lens))} '
              f'max {max(lens)}')
        print(f'  places per diary: median {int(np.median(n_places))}')
        return

    with open(PROMPTS_OUT, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    lens = [len(r['prompt']) for r in records]
    print(f'  Saved {len(records)} prompts ({len(users)} people x '
          f'{len(target_sets)} calls) -> {PROMPTS_OUT}')
    print(f'  Prompt chars: min {min(lens)} median {int(np.median(lens))} '
          f'max {max(lens)}')
    print(f'  Places per diary: median {int(np.median(n_places))}')

# =============================================================================
# <<< section: stage 3 — predict >>>
# =============================================================================

def _field(resp, name, default=None):
    if isinstance(resp, dict):
        return resp.get(name, default)
    return getattr(resp, name, default)


# Tried in order. Nothing here changes what is asked, only how much room the
# model has to answer and how tightly the answer is constrained.
RETRY_LADDER = [
    {},                                                    # whatever was configured
    {'num_predict': 4096},                                 # more room to answer
    {'num_predict': 4096, 'use_schema': False, 'think': 'low'},   # known-good
]


class EmptyResponse(RuntimeError):
    """Model ran but returned no final answer. Carries the diagnostics."""


class OllamaRunner:
    def __init__(self) -> None:
        try:
            import ollama
        except ImportError:
            sys.exit('pip install -U ollama')
        self.client = ollama.Client(host=OLLAMA_HOST, timeout=REQ_TIMEOUT)

    def __call__(self, prompt: str, schema: dict, temperature: float,
                 num_predict: int = NUM_PREDICT, use_schema: bool = USE_SCHEMA,
                 think: str = THINK_LEVEL) -> str:
        kwargs = {
            'model': MODEL_NAME, 'prompt': prompt, 'stream': False,
            'options': {'temperature': temperature, 'num_ctx': NUM_CTX,
                        'num_predict': num_predict},
        }
        if use_schema:
            kwargs['format'] = schema
        # 'default' omits the parameter entirely, which for gpt-oss means
        # unbounded reasoning -- kept only for comparison, not for real runs.
        if think and think != 'default':
            kwargs['think'] = think

        resp = self.client.generate(**kwargs)
        answer = str(_field(resp, 'response', '') or '').strip()
        if answer:
            return answer

        # Empty answers are the main failure mode with reasoning models, and
        # they are invisible without these four numbers.
        thinking = str(_field(resp, 'thinking', '') or '')
        raise EmptyResponse(
            f'no final answer (num_predict={num_predict}, '
            f'schema={use_schema}, think={think or "off"}, '
            f'thinking_chars={len(thinking)}, '
            f'done_reason={_field(resp, "done_reason", "?")}, '
            f'eval_count={_field(resp, "eval_count", "?")})'
        )


class HFRunner:
    """For the LoRA-tuned checkpoint. No schema enforcement; parsing falls back."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = HF_MODEL_DIR or MODEL_NAME
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='auto',
            trust_remote_code=True)
        self.model.eval()
        self.torch = torch

    def __call__(self, prompt: str, schema: dict, temperature: float) -> str:
        msgs = [{'role': 'user', 'content': prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors='pt').to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=NUM_PREDICT,
                do_sample=temperature > 0.05, temperature=max(temperature, 0.01),
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids['input_ids'].shape[1]:],
                               skip_special_tokens=True).strip()


CHAIN_OUT = OUT_DIR / f'chain_{TAG}.jsonl'


def _run_with_ladder(runner, prompt, schema, temp, label, i, n, uid):
    """One call, walking the retry ladder. Returns text or None."""
    last = None
    for attempt in range(1, PREDICT_RETRIES + 1):
        cfg = RETRY_LADDER[min(attempt - 1, len(RETRY_LADDER) - 1)]
        try:
            return runner(prompt, schema, temp, **cfg)
        except Exception as exc:                              # noqa: BLE001
            last = str(exc)
            print(f'  [{i}/{n}] {uid} {label} attempt {attempt}: {exc}')
            if attempt < PREDICT_RETRIES:
                time.sleep(2)
    return None


def _run_chain(runner, uid, desc, args, i, n, cache):
    """Stages 1 and 2, computed once per person and reused for every target."""
    if uid in cache:
        return cache[uid]
    free = {'type': 'object'}          # no schema: these stages return prose
    s1 = _run_with_ladder(runner, build_stage1_prompt(desc, args.role),
                          free, 0.1, 'stage1', i, n, uid)
    s2 = None
    if s1:
        s2 = _run_with_ladder(runner, build_stage2_prompt(s1, args.role),
                              free, 0.1, 'stage2', i, n, uid)
    cache[uid] = (s1, s2)
    if s1 and s2:
        with open(CHAIN_OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'user_id': uid, 'stage1': s1, 'stage2': s2},
                               ensure_ascii=False) + '\n')
    return cache[uid]


def stage_predict(args) -> None:
    print('=' * 70)
    mode = ('hierarchical: extract -> interpret -> infer'
            if args.hierarchical else 'single prompt')
    print(f'Stage 3 — inference ({args.backend}, {MODEL_NAME}, '
          f'{args.n_samples} sample(s) per prompt)')
    print(f'  Reasoning: {mode}')
    print('=' * 70)

    if not PROMPTS_OUT.exists():
        sys.exit(f'Missing {PROMPTS_OUT}. Run verbalize first.')

    runner = OllamaRunner() if args.backend == 'ollama' else HFRunner()

    if args.restart and PREDS_OUT.exists():
        PREDS_OUT.unlink()
        print('  Restart: previous checkpoint deleted.')

    done: set[tuple[str, str]] = set()
    if PREDS_OUT.exists():
        with open(PREDS_OUT, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('raw_outputs') and not rec.get('error'):
                    done.add((str(rec.get('user_id', '')).strip(),
                              str(rec.get('prompt_hash', '')).strip()))
        print(f'  Checkpoint: {len(done)} prompts already complete.')

    prompts = pd.read_json(PROMPTS_OUT, lines=True, dtype={'uid': str})
    prompts['uid'] = prompts['uid'].astype(str).str.strip()
    prompts['prompt_hash'] = prompts['prompt_hash'].astype(str).str.strip()
    todo = prompts.loc[[
        (u, h) not in done
        for u, h in zip(prompts['uid'], prompts['prompt_hash'])
    ]].copy()
    if args.limit:
        todo = todo.head(args.limit)
    print(f'  {len(todo)} prompts to run.')
    if todo.empty:
        print('  Nothing to do. Use --restart to regenerate.')
        return

    t0 = time.time()
    empty_prompts = 0
    chain_cache: dict[str, tuple] = {}
    if args.hierarchical and args.restart and CHAIN_OUT.exists():
        CHAIN_OUT.unlink()

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        targets = list(row.targets)
        schema = response_schema(targets)
        outputs, last_error = [], None
        s1 = s2 = None

        if args.hierarchical:
            s1, s2 = _run_chain(runner, row.uid, getattr(row, 'desc', ''),
                                args, i, len(todo), chain_cache)
            if not (s1 and s2):
                last_error = 'chain stage 1 or 2 produced nothing'

        for k in range(args.n_samples):
            if args.hierarchical and not (s1 and s2):
                break
            temp = 0.1 if args.n_samples == 1 else (0.2 + 0.5 * k / max(args.n_samples - 1, 1))
            prompt = (build_stage3_prompt(s1, s2, targets, args.role,
                                          args.context, args.strict_record)
                      if args.hierarchical else row.prompt)
            ans = _run_with_ladder(runner, prompt, schema, temp,
                                   f'sample {k + 1}', i, len(todo), row.uid)
            if ans:
                outputs.append(ans)
            else:
                last_error = last_error or 'no answer at any rung'

        rec = {'user_id': row.uid, 'prompt_hash': row.prompt_hash,
               'targets': targets, 'raw_outputs': outputs}
        if args.hierarchical:
            rec['stage1'] = s1
            rec['stage2'] = s2
        if not outputs:
            rec['error'] = last_error or 'unknown'
            empty_prompts += 1
        with open(PREDS_OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-9)
            print(f'  [{i}/{len(todo)}] {rate:.2f} prompts/s  '
                  f'eta {(len(todo) - i) / max(rate, 1e-9) / 60:.1f} min')

    print(f'  Predictions appended -> {PREDS_OUT}')
    if empty_prompts:
        print()
        print(f'  WARNING: {empty_prompts}/{len(todo)} prompts produced no '
              'answer at any rung of the retry ladder.')
        print('  Run `python probe_ollama.py` to see which setting the model '
              'will answer under, then set NUM_PREDICT / LLM_THINK / '
              'LLM_SCHEMA=0 accordingly.')


# =============================================================================
# <<< section: stage 4 — parse >>>
# =============================================================================

_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"')


def _balanced_objects(text: str) -> list[str]:
    """Every brace-balanced object in the text, longest-enclosing first.

    A greedy `\\{.*\\}` grabs an unbalanced substring whenever the model drops a
    closing brace, which it does often. This counts braces properly and ignores
    those inside strings.
    """
    out, stack, in_str, esc = [], [], False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            out.append(text[start:i + 1])
    return sorted(out, key=len, reverse=True)


def _repair(text: str) -> str | None:
    """Close up to three unclosed braces on a truncated object."""
    start = text.find('{')
    if start < 0:
        return None
    frag = text[start:].rstrip().rstrip(',')
    for _ in range(4):
        try:
            json.loads(frag)
            return frag
        except json.JSONDecodeError:
            frag += '}'
    return None


def _canon(value, cats: list[str]) -> str | None:
    if value is None:
        return None
    v = str(value).strip().strip('"\'').lower()
    v = (v.replace('chf', '').replace(' ', '')
          .replace('\u2013', '-').replace('\u2014', '-'))
    if not v:
        return None
    for c in cats:
        if v == c.lower().replace(' ', ''):
            return c
    aliases = {'16001+': '>16000', '16000+': '>16000', 'morethan16000': '>16000',
               'over16000': '>16000', '>=16000': '>16000',
               'lessthan4000': '<4000', '0-4000': '<4000', '<=4000': '<4000',
               'm': 'male', 'man': 'male', 'f': 'female', 'woman': 'female'}
    if aliases.get(v) in cats:
        return aliases[v]
    for c in sorted(cats, key=len, reverse=True):
        if c.lower().replace(' ', '') in v:
            return c
    return None


def _probs_from_obj(obj, cats: list[str]) -> np.ndarray | None:
    """Read a probability vector, tolerating percentages and missing keys."""
    if not isinstance(obj, dict):
        return None
    vec = np.zeros(len(cats), dtype=float)
    hit = False
    for raw_k, raw_v in obj.items():
        c = _canon(raw_k, cats)
        if c is None:
            continue
        try:
            vec[cats.index(c)] = max(float(raw_v), 0.0)
            hit = True
        except (TypeError, ValueError):
            continue
    if not hit or vec.sum() <= 0:
        return None
    return vec / vec.sum()


def _from_node(node, cats: list[str]) -> np.ndarray | None:
    """A node holding `probabilities` and/or `label`, in either nesting."""
    if not isinstance(node, dict):
        return _one_hot(_canon(node, cats), cats)
    vec = _probs_from_obj(node.get('probabilities'), cats)
    if vec is None:
        vec = _probs_from_obj({k: v for k, v in node.items()
                               if k not in ('label', 'evidence')}, cats)
    if vec is None:
        vec = _one_hot(_canon(node.get('label'), cats), cats)
    return vec


def _one_hot(label, cats: list[str]) -> np.ndarray | None:
    if not label:
        return None
    vec = np.zeros(len(cats))
    vec[cats.index(label)] = 1.0
    return vec


def parse_one(text: str, targets: list[str]) -> tuple[dict, dict]:
    """Return per-target probability vectors plus how each one was recovered.

    Order matters. The JSON paths are tried first because they carry the
    probabilities; the label regex keeps the right answer when the object is
    truncated; the bare prose scan is last and deliberately restricted, since
    scanning raw text matches the category names inside the probability dict
    rather than the chosen one.
    """
    out = {t: None for t in targets}
    how = {t: 'none' for t in targets}
    if not text:
        return out, how

    candidates = []
    for blob in _balanced_objects(text):
        try:
            candidates.append((json.loads(blob), 'json'))
        except json.JSONDecodeError:
            continue
    repaired = _repair(text)
    if repaired:
        try:
            candidates.append((json.loads(repaired), 'repaired'))
        except json.JSONDecodeError:
            pass

    for t in targets:
        cats = TARGETS[t]
        for data, tag in candidates:
            if not isinstance(data, dict):
                continue
            node = data.get(t)
            if node is None and t == 'income_level':
                node = data.get('household_income_level') or data.get('income')
            if node is None and ('probabilities' in data or 'label' in data):
                node = data                      # flat, single-target form
            vec = _from_node(node, cats) if node is not None else None
            if vec is not None:
                out[t], how[t] = vec, tag
                break

        if out[t] is None:                       # truncated: trust the label
            for m in _LABEL_RE.finditer(text):
                lab = _canon(m.group(1), cats)
                if lab:
                    out[t], how[t] = _one_hot(lab, cats), 'label_regex'
                    break

        if out[t] is None:                       # last resort, outside JSON only
            prose = re.sub(r'\{[^{}]*\}', ' ', text)
            for c in sorted(cats, key=len, reverse=True):
                if re.search(rf'(?<!\w){re.escape(c.lower())}(?!\w)', prose.lower()):
                    out[t], how[t] = _one_hot(c, cats), 'prose'
                    break
    return out, how


def stage_parse() -> None:
    print('=' * 70)
    print('Stage 4 — parse')
    print('=' * 70)

    if not PREDS_OUT.exists():
        sys.exit(f'Missing {PREDS_OUT}. Run predict first.')

    current = pd.read_json(PROMPTS_OUT, lines=True, dtype={'uid': str})
    current['uid'] = current['uid'].astype(str).str.strip()
    current['prompt_hash'] = current['prompt_hash'].astype(str).str.strip()
    valid = set(zip(current['uid'], current['prompt_hash']))

    preds = pd.read_json(PREDS_OUT, lines=True, dtype={'user_id': str})
    preds['user_id'] = preds['user_id'].astype(str).str.strip()
    preds['prompt_hash'] = preds['prompt_hash'].astype(str).str.strip()
    preds = preds.drop_duplicates(subset=['user_id', 'prompt_hash'], keep='last')
    preds = preds[[(u, h) in valid
                   for u, h in zip(preds['user_id'], preds['prompt_hash'])]]

    rows: dict[str, dict] = {}
    recovery: Counter = Counter()
    ties: Counter = Counter()
    rng = np.random.default_rng(SEED)
    n_calls = n_ok = 0
    for r in preds.itertuples(index=False):
        targets = list(r.targets)
        outs = list(getattr(r, 'raw_outputs', []) or [])
        n_calls += 1
        acc: dict[str, list[np.ndarray]] = defaultdict(list)
        for text in outs:
            vecs, hows = parse_one(str(text), targets)
            for t, vec in vecs.items():
                if vec is not None:
                    acc[t].append(vec)
                    recovery[hows[t]] += 1
        if acc:
            n_ok += 1
        row = rows.setdefault(r.user_id, {'user_id': r.user_id})
        for t, vecs in acc.items():                      # self-consistency
            mean = np.mean(np.vstack(vecs), axis=0)
            cats = TARGETS[t]
            # np.argmax always returns the first maximum, so an exact 50/50 --
            # which the model produces whenever it says there is no signal --
            # became a vote for whichever category is listed first. Break ties
            # at random instead, and count them.
            top = np.flatnonzero(np.isclose(mean, mean.max()))
            if len(top) > 1:
                ties[t] += 1
                pick = int(rng.choice(top))
            else:
                pick = int(top[0])
            row[t] = cats[pick]
            row[f'{t}_conf'] = round(float(mean.max()), 3)
            row[f'{t}_n_votes'] = len(vecs)
            # the full vector is what prior correction needs later
            row[f'{t}_probs'] = json.dumps(
                {c: round(float(v), 4) for c, v in zip(cats, mean)})

    clean = pd.DataFrame(list(rows.values()))
    for t in TARGETS:
        if t not in clean.columns:
            clean[t] = None
    if ties:
        print('  Exact ties broken at random: ' + ', '.join(
            f'{t} {n}' for t, n in ties.items()))
    clean.to_csv(PREDS_CLEAN, index=False)

    print(f'  {n_ok}/{n_calls} calls yielded a usable answer.')
    if recovery:
        total = sum(recovery.values())
        print('  How answers were recovered: ' + ', '.join(
            f'{k} {v} ({100 * v / total:.0f}%)'
            for k, v in recovery.most_common()))
        degraded = recovery['label_regex'] + recovery['prose']
        if degraded:
            print(f'  Note: {degraded} of {total} lost their probabilities to '
                  'malformed JSON and were read as a single label.')
    print(f'  {len(clean)} participants -> {PREDS_CLEAN}')
    for t in TARGETS:
        n = int(clean[t].notna().sum())
        print(f'    {t:14s} {n}/{len(clean)} ({100 * n / max(len(clean), 1):.1f}%)')


# =============================================================================
# <<< section: stage 5 — evaluate >>>
# =============================================================================

def _macro_f1(y_true: pd.Series, y_pred: pd.Series, cats: list[str]) -> float:
    f1s = []
    for c in cats:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
        elif tp == 0:
            continue
        else:
            p, r = tp / (tp + fp), tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float('nan')


def _balanced_acc(y_true: pd.Series, y_pred: pd.Series, cats: list[str]) -> float:
    recalls = [float(((y_true == c) & (y_pred == c)).sum()) / int((y_true == c).sum())
               for c in cats if int((y_true == c).sum()) > 0]
    return float(np.mean(recalls)) if recalls else float('nan')


def _boot_ci(y_true: np.ndarray, y_pred: np.ndarray, n: int = 2000,
             seed: int = SEED) -> tuple[float, float]:
    """Percentile CI on accuracy. With n=150 a 4-point lift is often noise."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    accs = (y_true[idx] == y_pred[idx]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _prior_correct(probs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rescale scores so the predicted class shares match a reference prior.

    The model ranks people better than it places them: it put 44% of
    participants in the top income band against a true 11%. Standard prior-shift
    correction divides each score by the share the model actually assigns to
    that class and multiplies by the share it should assign, which moves the
    decision boundary without touching the ranking.

        corrected(c) proportional to p(c) * reference(c) / model_share(c)

    The reference prior comes from labelled participants NOT in the evaluation
    sample, so no evaluated person's label is used. This is the same
    information a fitted classifier gets from its training set, and it must be
    reported as a separate condition from the raw output.
    """
    model_share = probs.mean(axis=0)
    model_share = np.where(model_share <= 1e-9, 1e-9, model_share)
    adj = probs * (reference / model_share)
    total = adj.sum(axis=1, keepdims=True)
    return np.divide(adj, np.where(total <= 0, 1.0, total))


def stage_evaluate() -> None:
    print('=' * 70)
    print('Stage 5 — evaluation')
    print('=' * 70)

    if not PREDS_CLEAN.exists():
        sys.exit(f'Missing {PREDS_CLEAN}. Run parse first.')

    pred = pd.read_csv(PREDS_CLEAN, dtype={'user_id': str})
    pred[USER_COL] = pred['user_id'].astype(str).str.strip()
    pred = pred.drop_duplicates(subset=USER_COL, keep='last')

    gt = load_ground_truth()
    merged = gt.merge(pred, on=USER_COL, how='inner',
                      suffixes=('_true', '_pred'), validate='one_to_one')
    print(f'  Matched participants: {len(merged)}')
    if merged.empty:
        sys.exit('  No overlap between predictions and ground truth.')

    evaluated = set(merged[USER_COL])
    held_out = gt.loc[~gt[USER_COL].isin(evaluated)]

    rows = []
    for target, cats in TARGETS.items():
        tcol, pcol = f'{target}_true', f'{target}_pred'
        if tcol not in merged.columns or pcol not in merged.columns:
            continue
        sub = merged.loc[merged[tcol].notna() & merged[pcol].notna(),
                         [tcol, pcol] + ([f'{target}_probs']
                                         if f'{target}_probs' in merged.columns
                                         else [])]
        if sub.empty:
            print(f'  {target}: nothing scorable.')
            continue

        yt, yp = sub[tcol], sub[pcol]
        acc = float((yt == yp).mean())
        maj = yt.value_counts(normalize=True)
        lo, hi = _boot_ci(yt.to_numpy(), yp.to_numpy())

        rec = {
            'target': target, 'n': len(sub),
            'accuracy': round(acc, 4),
            'acc_ci_low': round(lo, 4), 'acc_ci_high': round(hi, 4),
            'macro_f1': round(_macro_f1(yt, yp, cats), 4),
            'balanced_acc': round(_balanced_acc(yt, yp, cats), 4),
            'majority_baseline': round(float(maj.iloc[0]), 4),
            'majority_class': maj.index[0],
            'lift_over_majority': round(acc - float(maj.iloc[0]), 4),
            'beats_baseline': bool(lo > float(maj.iloc[0])),
        }
        if target in ORDINAL_TARGETS:
            ti = yt.map({c: i for i, c in enumerate(cats)})
            pi = yp.map({c: i for i, c in enumerate(cats)})
            ok = ti.notna() & pi.notna()
            rec['ordinal_mae'] = round(float((ti[ok] - pi[ok]).abs().mean()), 4)
            rec['adjacent_acc'] = round(float(((ti[ok] - pi[ok]).abs() <= 1).mean()), 4)

        # distribution skew: the mechanism behind the accuracy/macro-F1 gap
        pt = yt.value_counts(normalize=True).reindex(cats).fillna(0)
        pp = yp.value_counts(normalize=True).reindex(cats).fillna(0)
        rec['pred_dist_l1'] = round(float((pt - pp).abs().sum()), 4)
        rec['pred_top_class'] = pp.idxmax()
        rec['pred_top_share'] = round(float(pp.max()), 4)

        # prior-corrected variant, reported alongside rather than instead
        pcol_probs = f'{target}_probs'
        ref = held_out[target].value_counts(normalize=True).reindex(cats).fillna(0)
        if pcol_probs in sub.columns and sub[pcol_probs].notna().all() and ref.sum() > 0:
            try:
                P = np.vstack([
                    [json.loads(v).get(c, 0.0) for c in cats]
                    for v in sub[pcol_probs]
                ], dtype=float)
                Pc = _prior_correct(P, ref.to_numpy(dtype=float))
                yc = pd.Series([cats[i] for i in Pc.argmax(axis=1)], index=yt.index)
                acc_c = float((yt == yc).mean())
                ppc = yc.value_counts(normalize=True).reindex(cats).fillna(0)
                rec['acc_prior_corrected'] = round(acc_c, 4)
                rec['macro_f1_prior_corrected'] = round(_macro_f1(yt, yc, cats), 4)
                rec['dist_l1_prior_corrected'] = round(float((pt - ppc).abs().sum()), 4)
                rec['n_reference_users'] = int(held_out[target].notna().sum())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f'  {target}: prior correction skipped ({exc})')
        rows.append(rec)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_OUT, index=False)
    merged.to_csv(MERGED_OUT, index=False)

    print()
    print(metrics.to_string(index=False) if not metrics.empty else '  no metrics')
    print()
    print(f'  Metrics -> {METRICS_OUT}')
    print(f'  Merged  -> {MERGED_OUT}')

    for target, cats in TARGETS.items():
        tcol, pcol = f'{target}_true', f'{target}_pred'
        if tcol not in merged.columns or pcol not in merged.columns:
            continue
        sub = merged.loc[merged[tcol].notna() & merged[pcol].notna(), [tcol, pcol]]
        if sub.empty:
            continue
        table = pd.crosstab(sub[tcol], sub[pcol]).reindex(
            index=cats, columns=cats, fill_value=0)
        print(f'\n  Confusion — {target} (rows true, cols predicted)')
        print(table.to_string())




# =============================================================================
# <<< section: main >>>
# =============================================================================

def main() -> None:
    global NUM_PREDICT, THINK_LEVEL, USE_SCHEMA

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['build', 'verbalize', 'predict', 'parse',
                                      'evaluate', 'all'])
    ap.add_argument('--window-days', type=int, default=WINDOW_DAYS,
                    help='consecutive tracked days per diary')
    ap.add_argument('--sample-size', type=int, default=None,
                    help='number of participants to describe')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--flat', dest='hierarchical', action='store_false',
                    help='one prompt per target instead of the three-stage '
                         'chain (ablation)')
    ap.set_defaults(hierarchical=True)
    ap.add_argument('--no-cot', dest='cot', action='store_false',
                    help='drop the four reasoning steps and ask for the answer '
                         'directly (ablation)')
    ap.set_defaults(cot=True)
    ap.add_argument('--strict-record', action='store_true',
                    help='tell the model to work only from the record and to '
                         'split probability when it cannot separate two '
                         'categories. Implies --no-context, since the Swiss '
                         'reference points are outside knowledge')
    ap.add_argument('--no-context', dest='context', action='store_false',
                    help='withhold the Swiss reference points (ablation)')
    ap.set_defaults(context=True)
    ap.add_argument('--role', type=str, default=ROLE,
                    help='persona prepended to every prompt; pass "" to drop it')
    ap.add_argument('--target-mode', choices=['per_target', 'joint'],
                    default='per_target',
                    help='one call per attribute, or one for all three')
    ap.add_argument('--n-samples', type=int, default=3,
                    help='self-consistency samples per prompt; 1 disables it')
    ap.add_argument('--backend', choices=['ollama', 'hf'], default=BACKEND)
    ap.add_argument('--num-predict', type=int, default=NUM_PREDICT,
                    help='token budget for the answer; raise if responses are empty')
    ap.add_argument('--think', default=THINK_LEVEL,
                    choices=['low', 'medium', 'high', 'default'],
                    help="reasoning effort. 'default' omits the parameter, "
                         'which lets gpt-oss reason until the budget is gone '
                         'and return nothing; use it only for comparison')
    ap.add_argument('--schema', dest='use_schema', action='store_true',
                    help='constrain decoding to the JSON schema. Off by '
                         'default: gpt-oss:20b returns an empty response '
                         'under it. Check with probe_ollama.py first')
    ap.add_argument('--no-schema', dest='use_schema', action='store_false')
    ap.set_defaults(use_schema=USE_SCHEMA)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--chunksize', type=int, default=250_000)
    args = ap.parse_args()

    NUM_PREDICT, THINK_LEVEL, USE_SCHEMA = (
        args.num_predict, args.think, args.use_schema)
    RETRY_LADDER[0] = {'num_predict': NUM_PREDICT, 'use_schema': USE_SCHEMA,
                       'think': THINK_LEVEL}

    if args.stage in ('build', 'all'):
        stage_build(args)
    if args.stage in ('verbalize', 'all'):
        stage_verbalize(args)
        if args.dry_run:
            return
    if args.stage in ('predict', 'all'):
        stage_predict(args)
    if args.stage in ('parse', 'all'):
        stage_parse()
    if args.stage in ('evaluate', 'all'):
        stage_evaluate()


if __name__ == '__main__':
    main()
