#!/usr/bin/env python3
"""
Hierarchical chain-of-thought over verbalized mobility indicators
================================================================
Successor to `15_verbalized_indicator_pipeline_new_features.py`, restructured
as a HiCoTraj-style reasoning chain.

Why a chain
-----------
The single-prompt version scored 0.541 / 0.428 / 0.331 on gender / age /
income against baselines of 0.503 / 0.440 / 0.359 -- one target out of three
above the trivial predictor. Inspection of the responses showed the model
jumping straight from an indicator to a demographic label through a social
stereotype: "long car commutes suggest a male professional", "typical of a
working adult in the 25-44 age range". The second of those separates nothing,
since 45-65 also works. Asking for reasoning inside one prompt did not fix it;
the model wrote the headings and reached the same conclusion.

Splitting the reasoning across calls does fix the shortcut, because each stage
must produce an artefact the next stage consumes, and each artefact can be
inspected on its own.

    Stage 1  factual features      DETERMINISTIC -- verbalize_user()
    Stage 2  behavioural analysis  LLM, five dimensions, no demographics
    Stage 3  demographic inference LLM, evidence -> discard -> ranked answers

Stage 1 is not a model call here, which is a real advantage over the published
method. HiCoTraj asks an LLM to extract features from raw trajectories, so its
Stage 1 can invent facts; ours is computed from the feature matrix and cannot.
What the model receives is exactly what was measured.

Stage 2 says nothing about gender, age or income, so it is computed once per
participant and shared by all three targets: 1 + 3*k calls rather than 3*(2*k).

Stage 3 is asked for ranked alternatives, not a single guess. On two ordered
targets that is the natural output, and the ranking doubles as the probability
vector used for self-consistency averaging.

Stages
------
    verbalize   raw feature CSV   -> prompts JSONL (Stage 1 text)
    predict     prompts JSONL     -> checkpointed predictions JSONL
    parse       predictions JSONL -> clean predictions CSV
    evaluate    clean predictions -> metrics, confusion, bootstrap CI
 
    cd /data/baliu/thesis/11_test

LLM_MODEL='qwen3-30b-a3b-local:latest' \
LLM_RUN_TAG='qwen3_30b_full' \
python -u 18_hicot_indicator_pipeline_qwen3_30b_v1_0508.py predict \
  --n-samples 3 \
  --num-ctx 8192 \
  --num-predict 2048 \
  --stage2-num-predict 4096 \
  --think default \
  --no-schema \
  2>&1 | tee -a run_qwen3_full.log


Examples
--------
    # Reuse the existing prompt file and write a separate Qwen checkpoint.
    python 18_hicot_indicator_pipeline_qwen3_existing_prompts_0508.py predict

    # Resume the same Qwen checkpoint after an interruption.
    python 18_hicot_indicator_pipeline_qwen3_existing_prompts_0508.py predict

    # Parse and evaluate after prediction.
    python 18_hicot_indicator_pipeline_qwen3_existing_prompts_0508.py parse
    python 18_hicot_indicator_pipeline_qwen3_existing_prompts_0508.py evaluate
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

# from peft import LoraConfig
# from transformers import BitsAndBytesConfig

# bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
#                          bnb_4bit_compute_dtype='bfloat16',
#                          bnb_4bit_use_double_quant=True)
# lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
#                   target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
#                   task_type='CAUSAL_LM')

# =============================================================================
# <<< section: config >>>
# =============================================================================

USE_4WEEK_FILTER: bool = True
_feat_dir = Path(
    '/data/baliu/thesis/09_indicators/1_mobility_features_4weeks'
    if USE_4WEEK_FILTER else
    '/data/baliu/thesis/09_indicators/1_mobility_features_full'
)
FEATURES_FILE = Path(os.environ.get('MOBILITY_FEATURES_FILE',
                                    _feat_dir / 'feature_matrix_raw.csv'))
GROUND_TRUTH = Path(os.environ.get(
    'MOBILITY_MERGED_CSV',
    '/data/baliu/thesis/00data/sp_toponym_poi_purpose_demographics.csv'))
# Use the already generated 600-prompt file in 13_results.
# The prompt file is input-only in this Qwen-specific script.
OUT_DIR = Path(os.environ.get(
    'MOBILITY_OUT_DIR',
    '/data/baliu/thesis/18_results',
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = 'hicot_indicators_v1'
PROMPTS_OUT = Path(os.environ.get(
    'MOBILITY_PROMPTS_FILE',
    '/data/baliu/thesis/18_results/prompts_hicot_indicators_v1.jsonl',
))

USER_COL = 'user_id'
SEED = 42

SELECTED_FEATURES = [
    'stay_point_count', 'stay_radius_of_gyration', 'stay_area_km2',
    'stay_eccentricity', 'stay_direction_deg', 'top1_visit_frequency',
    'top2_visit_frequency', 'mean_travel_length_km',
    'motif_3_ratio', 'motif_1_ratio', 'motif_2_ratio', 'motif_4_ratio',
    'motif_5_ratio', 'motif_6_ratio', 'motif_7_ratio', 'motif_8_ratio',
    'motif_9_ratio', 'motif_99_ratio',
    'time_fragmented', 'travel_rhythm_entropy', 'rhythm_morning',
    'rhythm_afternoon', 'rhythm_evening', 'k_rog_ratio_2',
    'purpose_work_ratio', 'purpose_home_ratio', 'purpose_leisure_ratio',
    'top1_dur_ratio', 'top2_dur_ratio', 'top1_top2_dur_ratio',
    'landuse_residential_ratio', 'landuse_working_ratio',
    'landuse_mixed_ratio', 'landuse_entropy', 'home_work_lu_contrast',
    'commute_dist_km', 'dist_per_work_trip', 'work_travel_intensity',
    'work_peak_ratio', 'graph_density', 'n_weakly_connected',
    'n_strongly_connected', 'home_out_degree', 'home_betweenness',
    'mean_betweenness', 'mean_edge_weight', 'max_edge_weight', 'reciprocity',
]
DROP_CONSTANT_FEATURES = True

# --- model -------------------------------------------------------------------
BACKEND = os.environ.get('LLM_BACKEND', 'ollama')
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
MODEL_NAME = os.environ.get('LLM_MODEL', 'qwen3-30b-a3b-local:latest')

# LLM_MODEL='qwen3-30b-a3b-local:latest' \
# LLM_RUN_TAG='qwen3_30b_full' \


HF_MODEL_DIR = os.environ.get('HF_MODEL_DIR', '')
NUM_CTX = int(os.environ.get('LLM_NUM_CTX', '8192'))
NUM_PREDICT = int(os.environ.get('LLM_NUM_PREDICT', '2048'))
# Stage 2 writes a five-part analysis, several times longer than a stage-3
# answer. Sharing NUM_PREDICT with the reasoning channel starved it and it
# returned nothing for most participants.
STAGE2_NUM_PREDICT = int(os.environ.get('LLM_STAGE2_PREDICT', '4096'))
REQ_TIMEOUT = int(os.environ.get('LLM_TIMEOUT', '600'))
PREDICT_RETRIES = 3

# Set from probe_ollama.py against gpt-oss:20b. `think='low'` caps the hidden
# reasoning channel; omitting it lets the model reason until the budget is gone
# and return an empty response. JSON-schema decoding returns nothing at all for
# this model, so it is off by default.
THINK_LEVEL = os.environ.get('LLM_THINK', 'default')
USE_SCHEMA = os.environ.get('LLM_SCHEMA', '0') != '0'

# Keep the existing deterministic prompt file shared across models, while
# writing every Qwen-generated artefact to a separate file. In particular,
# never overwrite the completed gpt-oss output:
#   /data/baliu/thesis/13_results/preds_hicot_indicators_v1.jsonl
MODEL_TAG = re.sub(r'[^A-Za-z0-9._-]+', '_', MODEL_NAME).strip('_')
RUN_TAG = os.environ.get('LLM_RUN_TAG', 'qwen3_30b_from_existing_prompts')

PREDS_OUT = Path(os.environ.get(
    'MOBILITY_PREDS_FILE',
    str(OUT_DIR / f'preds_{TAG}_{RUN_TAG}.jsonl'),
))
CHAIN_OUT = Path(os.environ.get(
    'MOBILITY_CHAIN_FILE',
    str(OUT_DIR / f'chain_{TAG}_{RUN_TAG}.jsonl'),
))
PREDS_CLEAN = Path(os.environ.get(
    'MOBILITY_CLEAN_FILE',
    str(OUT_DIR / f'preds_{TAG}_{RUN_TAG}_clean.csv'),
))
MERGED_OUT = Path(os.environ.get(
    'MOBILITY_MERGED_OUT',
    str(OUT_DIR / f'merged_{TAG}_{RUN_TAG}.csv'),
))
METRICS_OUT = Path(os.environ.get(
    'MOBILITY_METRICS_OUT',
    str(OUT_DIR / f'metrics_{TAG}_{RUN_TAG}.csv'),
))

ROLE = os.environ.get(
    'LLM_ROLE',
    'You are a senior researcher in transport geography and travel-behaviour '
    'analysis. You have spent years reading GNSS mobility indicators alongside '
    'Swiss household travel survey data.'
)

# =============================================================================
# <<< section: label spaces >>>
# =============================================================================
# The feature-based chapter bins age as 18-24 / 25-44 / 45-65 (n=1,791). The
# earlier verbalized script used 45-66, which silently moves both the baseline
# and the matched count. 65 keeps the two comparable.
AGE_TOP = int(os.environ.get('AGE_TOP', '65'))

GENDER_CATS = ['male', 'female']
AGE_CATS = ['18-24', '25-44', f'45-{AGE_TOP}']
INCOME_CATS = ['<4000', '4001-8000', '8001-12000', '12001-16000', '>16000']

TARGETS = {
    'gender': GENDER_CATS,
    'age_group': AGE_CATS,
    'income_level': INCOME_CATS,
}
ORDINAL_TARGETS = {'age_group', 'income_level'}

TARGET_QUESTION = {
    'gender': "the participant's most likely gender",
    'age_group': "the participant's most likely age group",
    'income_level': "the household's most likely gross monthly income, in CHF",
}

# Published facts about Switzerland, not statistics from this study's labels.
# Present because the single-prompt version read "owns a car and commutes far"
# as wealth, predicting the top income band for 44% of participants against a
# true 11%. Withhold with --no-context to measure the effect.
SWISS_CONTEXT = """## Reference points
These describe Switzerland in general and say nothing about this participant.
- Median gross household income is roughly CHF 10,000 per month. Above
  CHF 16,000 is about the top tenth of households.
- Around four in five households own a car, and a 20-30 km commute is
  unremarkable. Neither is a sign of wealth.
- People hold full-time jobs across the whole 25-65 range, so working regular
  hours does not narrow the age down.
- Both men and women commute long distances by car, and both do the school run."""

# =============================================================================
# <<< section: stage 1 — verbalizers (deterministic) >>>
# =============================================================================

def _pct(x) -> str:
    return f'{100 * float(x):.0f}%'


def _num(x, nd=1) -> str:
    return f'{float(x):.{nd}f}'

def _int(x) -> str:
    return f'{int(round(float(x)))}'

G_SPATIAL = 'spatial'
V_SPATIAL = {
    'stay_convex_hull_diameter':
        lambda v: f'the two furthest places they visited are {_num(v)} km apart',
    'stay_radius_of_gyration':
        lambda v: f'their activity space has a radius of gyration of {_num(v)} km',
    'stay_area_km2':
        lambda v: f'it covers an area of about {_num(v)} km2',
    'stay_eccentricity':
        lambda v: (
            f'the activity space is strongly elongated along one corridor '
            f'(eccentricity {_num(v, 2)})' if float(v) > 0.9 else
            f'the activity space is moderately elongated (eccentricity {_num(v, 2)})'
            if float(v) > 0.6 else
            f'the activity space is roughly circular (eccentricity {_num(v, 2)})'
        ),
    'stay_point_count':
        lambda v: f'{_int(v)} stay-point records were observed',
    'unique_stay_locations':
        lambda v: f'spread over {_int(v)} distinct locations',
    'stay_entropy':
        lambda v: f'the entropy of the visit distribution is {_num(v, 2)}',
    'top1_visit_frequency':
        lambda v: f'{_pct(v)} of all visits fall on their single most visited place',
    'top2_visit_frequency':
        lambda v: f'and a further {_pct(v)} on the second most visited place',
    'k_rog_ratio_2':
        lambda v: (
            f'the radius-of-gyration ratio for the two most visited locations '
            f'versus all locations is {_num(v, 2)}; the two anchors are more '
            f'separated than the overall visit-weighted spatial scale'
            if float(v) > 1.05 else
            f'the radius-of-gyration ratio for the two most visited locations '
            f'versus all locations is {_num(v, 2)}, so the two anchors reproduce '
            f'most of the overall spatial scale'
            if float(v) >= 0.75 else
            f'the radius-of-gyration ratio for the two most visited locations '
            f'versus all locations is {_num(v, 2)}, indicating that locations '
            f'beyond the two main anchors contribute substantially to spatial spread'
        ),
}

G_TRAVEL = 'travel'
V_TRAVEL = {
    'n_travels':
        lambda v: f'{_int(v)} trips between places were observed',
    'total_travel_length_km':
        lambda v: f'covering {_num(v)} km in total',
    'mean_travel_length_km':
        lambda v: f'the mean distance between consecutive distinct locations is {_num(v)} km',
    'od_entropy':
        lambda v: f'the entropy over origin-destination pairs is {_num(v, 2)}',
}

G_RHYTHM = 'rhythm'
V_RHYTHM = {
    'rhythm_morning':
        lambda v: f'{_pct(v)} of departures occur in the morning (06-12h)',
    'rhythm_afternoon':
        lambda v: f'{_pct(v)} of departures occur in the afternoon (12-18h)',
    'rhythm_evening':
        lambda v: f'{_pct(v)} of departures occur in the evening (18-24h)',
    'travel_rhythm_entropy':
        lambda v: (
            f'departures are spread evenly across the day '
            f'(rhythm entropy {_num(v, 2)} of a maximum 3.0 bits)'
            if float(v) > 2.2 else
            f'departures are concentrated in a few fixed time windows '
            f'(rhythm entropy {_num(v, 2)} of a maximum 3.0 bits)'
        ),
    'time_fragmented':
        lambda v: (
            f'stay durations are very uneven, mixing short and long stays '
            f'(standard deviation {_num(v)} h)' if float(v) > 4 else
            f'stay durations are fairly uniform (standard deviation {_num(v)} h)'
        ),
}

G_SEMANTIC = 'semantic'
V_SEMANTIC = {
    'purpose_home_ratio':
        lambda v: f'{_pct(v)} of recorded time is spent at home',
    'purpose_work_ratio':
        lambda v: f'{_pct(v)} at work',
    'purpose_leisure_ratio':
        lambda v: f'{_pct(v)} at leisure activities',
    'purpose_entropy':
        lambda v: f'the entropy over activity purposes is {_num(v, 2)}',
    'top1_dur_ratio':
        lambda v: f'their most visited location absorbs {_pct(v)} of total time',
    'top2_dur_ratio':
        lambda v: f'the second most visited location {_pct(v)}',
    'top1_top2_dur_ratio':
        lambda v: f'a top-1 to top-2 dominance ratio of {_num(v, 2)}',
    'top1_purpose_home':
        lambda v: f'{_pct(v)} of the time at the top location is labelled home',
    'landuse_residential_ratio':
        lambda v: f'{_pct(v)} of time with known land use is in residential zones',
    'landuse_working_ratio':
        lambda v: f'{_pct(v)} of time with known land use is in employment or commercial zones',
    'landuse_mixed_ratio':
        lambda v: f'{_pct(v)} of time with known land use is in mixed-use zones',
    'landuse_entropy':
        lambda v: f'the entropy over land-use categories is {_num(v, 2)}',
    'home_work_lu_contrast':
        lambda v: (
            f'home and workplace sit in markedly different built environments '
            f'(land-use contrast {_num(v, 2)})' if float(v) > 0.5 else
            f'home and workplace sit in similar built environments '
            f'(land-use contrast {_num(v, 2)})'
        ),
}

G_WORK = 'work'
V_WORK = {
    'commute_dist_km':
        lambda v: (None if float(v) <= 0 else
                   f'home and workplace are {_num(v)} km apart'),
    'dist_per_work_trip':
        lambda v: (None if float(v) <= 0 else
                   f'work-bound trips average {_num(v)} km'),
    'work_peak_ratio':
        lambda v: (f'{_pct(v)} of departures on work-bound transitions fall '
                   f'inside peak hours (07-09h and 17-19h)'),
    'work_travel_intensity':
        lambda v: f'work-related travel makes up {_pct(v)} of total distance',
    'work_hour_dist_product':
        lambda v: f'the product of daily work hours and daily distance is {_num(v)}',
}

G_MOTIF = 'motif'
_MOTIF_DESC = {
    'motif_1_ratio': 'the day follows home-A-home',
    'motif_2_ratio': 'the day follows home-A-B-home',
    'motif_3_ratio': 'the participant remains at home all day',
    'motif_4_ratio': 'the day follows home-A-B-C-home',
    'motif_5_ratio': 'the day follows home-A-home-B-home',
    'motif_6_ratio': 'the day follows home-A-B-A-home',
    'motif_7_ratio': 'the day follows home-A-B-A-B-home',
    'motif_8_ratio': 'the day follows a longer home-return chaining pattern',
    'motif_9_ratio': 'the day follows the motif-9 equivalent pattern',
    'motif_99_ratio': 'the day does not match a common reference motif',
}
V_MOTIF = {
    k: (lambda desc: (lambda v: None if float(v) <= 0 else
                      f'on {_pct(v)} of valid diary days, {desc}'))(d)
    for k, d in _MOTIF_DESC.items()
}
V_MOTIF['motif_stayhome_days'] = \
    lambda v: f'{_int(v)} days were spent entirely at home'

G_GRAPH = 'graph'
V_GRAPH = {
    'n_nodes': lambda v: f'the graph has {_int(v)} distinct locations',
    'n_edges': lambda v: f'connected by {_int(v)} distinct directed links',
    'graph_density':
        lambda v: (
            f'the directed network is sparse, with only {_pct(v)} of possible '
            f'connections realised' if float(v) < 0.3 else
            f'the network is comparatively dense, with {_pct(v)} of possible '
            f'connections realised'
        ),
    'n_weakly_connected':
        lambda v: f'it splits into {_int(v)} weakly connected components',
    'n_strongly_connected':
        lambda v: (
            f'it contains {_int(v)} strongly connected components, indicating '
            f'fragmented mutual reachability between locations' if float(v) > 1 else
            f'it forms one strongly connected component, so every observed '
            f'location is reachable from every other through directed paths'
        ),
    'home_in_degree': lambda v: f'{_int(v)} distinct places lead back to home',
    'home_out_degree':
        lambda v: f'{_int(v)} distinct places are reached directly from home',
    'home_betweenness':
        lambda v: (
            f'home lies on almost every route between other places '
            f'(betweenness {_num(v, 2)}), a hub-and-spoke routine'
            if float(v) > 0.5 else
            f'home lies on only some routes between other places '
            f'(betweenness {_num(v, 2)}), so destinations are often chained directly'
        ),
    'mean_betweenness':
        lambda v: f'mean betweenness across all locations is {_num(v, 3)}',
    'mean_edge_weight': lambda v: f'each link is used {_num(v)} times on average',
    'max_edge_weight':
        lambda v: f'the single most used link is travelled {_int(v)} times',
    'reciprocity':
        lambda v: (
            f'{_pct(v)} of directed links have a reverse counterpart, consistent '
            f'with frequent out-and-back movement' if float(v) > 0.5 else
            f'only {_pct(v)} of directed links have a reverse counterpart'
        ),
}

GROUPS = [
    (G_SPATIAL, 'Where they go', V_SPATIAL),
    (G_TRAVEL, 'How much they travel', V_TRAVEL),
    (G_RHYTHM, 'When they travel', V_RHYTHM),
    (G_SEMANTIC, 'What the places are for', V_SEMANTIC),
    (G_WORK, 'Work and commuting', V_WORK),
    (G_MOTIF, 'How their days are shaped', V_MOTIF),
    (G_GRAPH, 'Their multi-day location network', V_GRAPH),
]

# `stay_direction_deg` is a compass bearing of the activity-space principal
# axis. It has no demographic meaning in any direction and only lengthens the
# prompt, so it has no verbalizer and is dropped.


def verbalize_user(row: pd.Series, keep_cols: list[str]) -> str:
    """Map one participant's raw indicator row to Stage 1 text."""
    paragraphs: list[str] = []
    for _key, heading, registry in GROUPS:
        clauses: list[str] = []
        for feat, fmt in registry.items():
            if feat not in keep_cols:
                continue
            val = row.get(feat, np.nan)
            if pd.isna(val):
                continue
            try:
                clause = fmt(val)
            except (TypeError, ValueError):
                continue
            if clause:
                clauses.append(clause)

        if _key == G_RHYTHM:
            needed = {'rhythm_morning', 'rhythm_afternoon', 'rhythm_evening'}
            if needed.issubset(keep_cols) and needed.issubset(row.index):
                vals = pd.to_numeric(row[list(needed)], errors='coerce')
                if vals.notna().all():
                    night = float(np.clip(1.0 - vals.sum(), 0.0, 1.0))
                    clauses.insert(3, f'{_pct(night)} of departures occur '
                                      'overnight (00-06h)')

        if not clauses:
            continue
        body = clauses[0][0].upper() + clauses[0][1:]
        if len(clauses) > 1:
            body += '; ' + '; '.join(clauses[1:])
        paragraphs.append(f'{heading}: {body}.')
    return '\n'.join(paragraphs)


# =============================================================================
# <<< section: stage 2 — behavioural pattern analysis >>>
# =============================================================================
# Five dimensions, each pointed at the indicators that actually carry it. The
# published framework lists an economic dimension built on venue price tiers;
# this data has none, so that heading is redirected onto the cost and reach of
# the movement itself and the model is told what it cannot see. Left unsaid,
# it invents spending behaviour -- the single-prompt version produced "frequents
# upscale leisure spots" from a record containing no venue information at all.

STAGE2_TASK = """## Task
Below is a statistical description of one anonymous participant's travel
behaviour over several weeks, measured by GNSS tracking in Switzerland. Read it
as evidence about how this person lives, and write a short analysis under five
headings. Cite the figures you rely on.

1. TEMPORAL - work-life structure
   When does the day begin and end? Are departures concentrated in fixed
   windows or spread across the day? Do work-bound trips fall in peak hours?
   How is recorded time divided between home, work and leisure? Does this look
   like fixed hours, flexible hours, shift work, or no work-like pattern?

2. ECONOMIC - what the movement itself costs and reaches
   How large is the activity space? How far is the commute and how much of
   total distance is work-related? Note that this record contains no venue
   names, prices or spending data, so judge only the extent and cost of the
   movement, and state plainly where that leaves you unable to tell.

3. SOCIAL - discretionary activity and lifestyle
   How much leisure time, and at what hours? Is there evening or overnight
   activity? Are days chained through several places or simple out-and-back
   trips? How varied are the land-use types they spend time in?

4. SPATIAL - living environment
   Which built environments absorb their time: residential, employment,
   mixed-use? Do home and workplace sit in contrasting environments? Is the
   activity space compact or stretched along a corridor?

5. STABILITY - routine consistency
   How much of the week repeats a common day shape and how much does not? How
   concentrated is time on one or two anchors? Is home the hub through which
   every journey passes, or are destinations chained directly?

Describe how this person lives. Do not name an age, a gender, an income, an
occupation or a family situation; that comes in the next step."""

def build_stage2_prompt(stage1: str, role: str = ROLE) -> str:
    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    return f'{head}{STAGE2_TASK}\n\n## Mobility statistics\n{stage1}'

# =============================================================================
# <<< section: stage 2.5 — export fine-tuning data >>>
# =============================================================================

FT_TRAIN = OUT_DIR / f'finetune_{TAG}_train.jsonl'
FT_VAL   = OUT_DIR / f'finetune_{TAG}_val.jsonl'


def _ft_target(gt_row) -> str:
    """Label-only JSON. Probabilities are omitted deliberately: the ground
    truth has none, and training one-hot vectors would collapse the
    self-consistency averaging in stage_parse to a constant."""
    return json.dumps({t: gt_row[t] for t in TARGETS}, ensure_ascii=False)


def stage_traindata(args) -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print('=' * 70 + '\nFine-tuning data export\n' + '=' * 70)
    if not PROMPTS_OUT.exists():
        sys.exit(f'Missing {PROMPTS_OUT}. Run verbalize first.')

    gt = load_ground_truth().dropna(subset=list(TARGETS))
    raw, _keep = load_features(args.features_file)
    merged = raw[[USER_COL]].merge(gt, on=USER_COL, how='inner')

    # Same split the feature-based gender classifier used, so the fine-tuned
    # model is never trained on a participant it is later scored on.
    idx = np.arange(len(merged))
    _, idx_test = train_test_split(
        idx, test_size=0.20, random_state=42,
        stratify=LabelEncoder().fit_transform(merged['gender']))
    held_out = set(merged.iloc[idx_test][USER_COL])
    print(f'  held out, never trained on: {len(held_out)}')

    gt_by_uid = merged.set_index(USER_COL).to_dict('index')
    chain = _load_chain_cache()
    print(f'  stage-2 analyses available: {len(chain)}')

    rows, skipped = [], 0
    for line in open(PROMPTS_OUT, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        uid = str(r['uid']).strip()
        if uid in held_out or uid not in gt_by_uid:
            continue
        s2 = chain.get(uid)
        if not s2:
            # Training on a fabricated stage 2 would teach the model to
            # condition on text the chain never produced.
            skipped += 1
            continue
        prompt = build_stage3_prompt(r['desc'], s2, list(TARGETS),
                                     args.role, args.context)
        rows.append({'messages': [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': _ft_target(gt_by_uid[uid])},
        ]})

    if skipped:
        print(f'  skipped {skipped} with no cached stage-2 analysis')
    if not rows:
        sys.exit('  no training examples — run predict first to populate the chain')

    tr, va = train_test_split(rows, test_size=0.1, random_state=SEED)
    for path, data in ((FT_TRAIN, tr), (FT_VAL, va)):
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f'  {len(data):>5} examples -> {path}')


# =============================================================================
# <<< section: stage 3 — demographic inference >>>
# =============================================================================
# Evidence, then discard, then ranked alternatives. The discard step exists
# because the single-prompt version offered "typical of a working adult" as
# evidence for 25-44, which separates nothing from 45-65; naming what was set
# aside makes that visible. Ranked alternatives suit two ordered targets and
# give the second choice a rationale rather than leaving it implicit.

def build_stage3_prompt(stage1: str, stage2: str, targets: list[str],
                        role: str = ROLE, context: bool = True) -> str:
    cat_lines = '\n'.join(f'  - {t}: {json.dumps(TARGETS[t])}' for t in targets)
    ask = '\n'.join(f'  {i}. {TARGET_QUESTION[t]}'
                    for i, t in enumerate(targets, start=1))
    def _node(t: str) -> str:
        # spelling the class keys out matters: the parser reads the
        # probability vector by key, and a literal "{...}" taught the model
        # to omit it, degrading every joint answer to a bare label
        pr = ', '.join(f'"{c}": 0.0' for c in TARGETS[t])
        return f'"{t}": {{"label": "...", "probabilities": {{{pr}}}}}'

    if len(targets) == 1:
        cats = TARGETS[targets[0]]
        probs = ', '.join(f'"{c}": 0.0' for c in cats)
        shape = f'"label": "...", "probabilities": {{{probs}}}, "evidence": "..."'
    else:
        shape = ', '.join(_node(t) for t in targets) + ', "evidence": "..."'

    ordered = [t for t in targets if t in ORDINAL_TARGETS]
    if ordered:
        which = ('Age group and income level are ordered scales'
                 if len(ordered) > 1 else
                 f'{ordered[0].replace("_", " ").capitalize()} is an ordered scale')
        ordinal_rule = (f'- {which}, so rank neighbouring categories second\n'
                        '  rather than distant ones, and let the probabilities\n'
                        '  fall away on both sides of your first choice.\n')
    else:
        ordinal_rule = ''

    # asking for three ranked answers when a target has two categories reads
    # as a mistake and invites the model to invent a third. Taking the minimum
    # across targets was wrong in joint mode: gender's two categories won, so
    # income's five were also asked for "both".
    if len(targets) == 1:
        n_cat = len(TARGETS[targets[0]])
        rank_n = ('rank both categories, most likely first' if n_cat == 2 else
                  'give your three most likely categories in order')
        coherence = ''
    else:
        rank_n = ('for each attribute in turn, rank its categories most likely '
                  'first: both of them where there are only two, otherwise the '
                  'top three')
        # the substantive reason for asking all three at once
        coherence = ('- Keep the three answers mutually consistent. An 18-24\n'
                     '  participant in the top income band, for instance, needs\n'
                     '  a reason. Infer them together, not one after another.\n'
                     '- Keep steps 1-3 to AT MOST THREE SHORT LINES PER\n'
                     '  ATTRIBUTE. The JSON is the required output and the\n'
                     '  reasoning is scratch work; a long analysis that never\n'
                     '  reaches the JSON counts as no answer at all.\n')

    # Joint mode needs its own template. Reusing the per-target one made the
    # model write EVIDENCE / DISCARD / RANKED ALTERNATIVES three times over,
    # reaching 32,000 characters without ever emitting the JSON. The fix is
    # not a larger budget -- it is putting the required output first and
    # capping the reasoning.
    if len(targets) > 1:
        template = dedent("""
        You have a statistical description of one participant's travel
        behaviour and your own behavioural analysis of it. Use both, and
        nothing else.

        ## Required output
        Your reply must be EXACTLY ONE JSON object and NOTHING ELSE. No
        preamble, no headings, no explanation outside the object.

        {{{shape}}}

        ## Categories (copy verbatim)
        {cat_lines}

        ## How to decide
        Think briefly before writing, but do not write your thinking down.
        Set aside anything that fits every category equally well: working
        regular hours does not separate 25-44 from 45-65, and owning a car
        does not separate income bands. If almost nothing is left, keep the
        probabilities close to even rather than committing on a hunch.

        ## Rules
        - Do not invent detail that is in neither text.
        {ordinal_rule}
        {coherence}
        - "evidence" is ONE sentence, at most 25 words, covering all three
          attributes together.
        - Probabilities for each attribute must sum to 1.
        - Close every brace you open. Write nothing after the final brace.
        """).strip()
    else:
        template = dedent("""
    You have a statistical description of one participant's travel behaviour
    and your own behavioural analysis of it. Use both, and nothing else.

    ## Task
    Infer:      
        Infer all three attributes for this participant at once. They are
        correlated, so use each to constrain the others.
    {ask}

    ## Categories (copy verbatim)
    {cat_lines}

    ## How to answer
    Work through these in order, briefly, then give the JSON.

    1. EVIDENCE - name the findings from the two texts that bear on this
       attribute, quoting the figures.
    2. DISCARD - name the findings that fit every category equally well and set
       them aside. Working regular hours does not separate 25-44 from 45-65;
       owning a car does not separate income bands. Say what you discarded.
    3. RANKED ALTERNATIVES - {rank_n}. For each, one sentence on what supports
       it and what argues against it.
    4. If step 2 removed almost everything, say so and keep the probabilities
       close to even rather than committing on a hunch.

    ## Rules
    - Do not invent detail that is in neither text.
    {ordinal_rule}
    {coherence}
    - Give a single most likely label, but let the probabilities carry your
      uncertainty.

    ## Output
    End with a single JSON object on its own final line:
    {{{shape}}}
    "evidence" is one sentence, at most 30 words, naming what decided it.
    Close every brace you open.
    """).strip()
    # (joint mode returns earlier with its own template)

    instructions = (template
                    .replace('{ask}', ask)
                    .replace('{cat_lines}', cat_lines)
                    .replace('{ordinal_rule}\n', ordinal_rule)
                    .replace('{coherence}\n', coherence)
                    .replace('{rank_n}', rank_n)
                    .replace('{{{shape}}}', '{' + shape + '}'))

    head = f'{role.strip()}\n\n' if role and role.strip() else ''
    ctx = f'\n\n{SWISS_CONTEXT}' if context else ''
    return (f'{head}{instructions}{ctx}'
            f'\n\n## Mobility statistics\n{stage1}'
            f'\n\n## Behavioural analysis\n{stage2}')


def response_schema(targets: list[str]) -> dict:
    """Only used if --schema is on; gpt-oss returns nothing under it."""
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

def build_flat_prompt(stage1: str, targets: list[str], role: str = ROLE,
                      context: bool = True) -> str:
    """Ablation: one call, no behavioural stage."""
    return build_stage3_prompt(stage1, '(not produced in this condition)',
                               targets, role, context)

# =============================================================================
# <<< section: ground truth (isolated read) >>>
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
    if not GROUND_TRUTH.exists():
        sys.exit(f'Missing {GROUND_TRUTH}')

    header = pd.read_csv(GROUND_TRUTH, nrows=0)
    id_col = 'participant_ID' if 'participant_ID' in header.columns else USER_COL
    gcol = next((c for c in ('gender', 'sex') if c in header.columns), None)
    usecols = [id_col] + [c for c in ('age', 'income') if c in header.columns]
    if gcol:
        usecols.append(gcol)

    gt = pd.read_csv(GROUND_TRUTH, usecols=list(dict.fromkeys(usecols)),
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
# <<< section: verbalize >>>
# =============================================================================

def load_features(features_file: Path) -> tuple[pd.DataFrame, list[str]]:
    if not features_file.exists():
        sys.exit(f'Missing feature matrix: {features_file}')
    raw = pd.read_csv(features_file, dtype={USER_COL: str}, low_memory=False)
    if USER_COL not in raw.columns:
        sys.exit(f'{features_file} has no {USER_COL!r} column.')
    raw[USER_COL] = raw[USER_COL].fillna('').astype(str).str.strip()
    if raw[USER_COL].duplicated().any():
        sys.exit('Duplicate user_id values in the feature matrix.')

    missing = [c for c in SELECTED_FEATURES if c not in raw.columns]
    if missing:
        sys.exit(f'Feature matrix missing {len(missing)} columns: {missing}')

    keep = list(SELECTED_FEATURES)
    raw[keep] = (raw[keep].apply(pd.to_numeric, errors='coerce')
                 .replace([np.inf, -np.inf], np.nan))
    if DROP_CONSTANT_FEATURES:
        constant = [c for c in keep if raw[c].nunique(dropna=False) <= 1]
        if constant:
            print(f'  Omitting {len(constant)} constant indicators: {constant}')
            keep = [c for c in keep if c not in constant]

    covered = {f for _k, _h, reg in GROUPS for f in reg}
    unhandled = [c for c in keep if c not in covered]
    if unhandled:
        print(f'  No verbalizer for {len(unhandled)}, omitted: {unhandled}')
        keep = [c for c in keep if c in covered]

    print(f'  Feature file: {features_file}')
    print(f'  Participants: {len(raw):,}; indicators verbalized: {len(keep)}')
    return raw, keep

def stage_verbalize(args) -> None:
    print('=' * 70)
    print('Stage 1 — verbalize indicators (deterministic)')
    print('=' * 70)

    raw, keep = load_features(args.features_file)
    gt = load_ground_truth()
    labelled = set(gt[USER_COL])
    raw = raw.loc[raw[USER_COL].isin(labelled)]
    print(f'  Labelled participants: {len(raw):,}')

    if args.sample_size is not None and args.sample_size < len(raw):
        raw = raw.sample(n=args.sample_size, random_state=args.seed)
    raw = raw.sort_values(USER_COL)
    if args.limit:
        raw = raw.head(args.limit)
    print(f'  Describing {len(raw)} participants (seed {args.seed}).')
    print(f'  Chain: {"on" if args.hierarchical else "off (--flat)"}; '
          f'reference points: {"on" if args.context else "off"}')

    target_sets = ([[t] for t in TARGETS] if args.target_mode == 'per_target'
                   else [list(TARGETS)])

    records = []
    for _, row in raw.iterrows():
        desc = verbalize_user(row, keep)
        for ts in target_sets:
            key = desc + '|' + ','.join(ts) + f'|{args.hierarchical}{args.context}'
            records.append({
                'uid': row[USER_COL],
                'targets': ts,
                'desc': desc,
                'prompt': build_flat_prompt(desc, ts, args.role, args.context),
                'prompt_hash': hashlib.sha256(key.encode()).hexdigest()[:16],
            })

    if args.dry_run:
        r = records[0]
        print('\n--- STAGE 1 (deterministic) ---\n')
        print(r['desc'])
        print('\n--- STAGE 2 PROMPT ---\n')
        p2 = build_stage2_prompt(r['desc'], args.role)
        print(p2[:p2.index('## Mobility statistics')].strip())
        print('\n--- STAGE 3 PROMPT ---\n')
        p3 = build_stage3_prompt(r['desc'], '<behavioural analysis>', r['targets'],
                                 args.role, args.context)
        print(p3[:p3.index('## Mobility statistics')].strip())
        lens = [len(x['desc']) for x in records]
        print(f'\n  Stage 1 chars: min {min(lens)} median {int(np.median(lens))} '
              f'max {max(lens)}')
        return

    with open(PROMPTS_OUT, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    lens = [len(r['desc']) for r in records]
    print(f'  Saved {len(records)} prompts ({raw[USER_COL].nunique()} people x '
          f'{len(target_sets)} calls) -> {PROMPTS_OUT}')
    print(f'  Stage 1 chars: min {min(lens)} median {int(np.median(lens))} '
          f'max {max(lens)}')

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
    {},
    {'num_predict': 4096, 'use_schema': False, 'think': 'default'},
    {'num_predict': 6144, 'use_schema': False, 'think': 'default'},
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
        if think == 'off':
            kwargs['think'] = False      # qwen3 takes a bool, not a level
        elif think and think != 'default':
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

    def __call__(self, prompt: str, schema: dict, temperature: float,
                 **_ignored) -> str:
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


def _run_with_ladder(runner, prompt, schema, temp, label, i, n, uid,
                     num_predict: int | None = None,
                     targets: list[str] | None = None):
    """One call, walking the retry ladder. Returns text or None.

    `targets` makes the ladder sensitive to truncation. Without it the ladder
    only fires on a completely empty response, so an answer cut off before the
    required JSON counted as a success and was silently lost -- which is how a
    joint run reached 199/200 non-empty responses but only 73/200 parsable
    ones. Pass targets for stage 3; leave it None for stage 2, which returns
    prose by design.
    """
    last = None
    for attempt in range(1, PREDICT_RETRIES + 1):
        cfg = dict(RETRY_LADDER[min(attempt - 1, len(RETRY_LADDER) - 1)])
        if num_predict is not None:
            cfg['num_predict'] = max(cfg.get('num_predict', 0), num_predict)
        try:
            ans = runner(prompt, schema, temp, **cfg)
        except Exception as exc:                              # noqa: BLE001
            last = str(exc)
            print(f'  [{i}/{n}] {uid} {label} attempt {attempt}: {exc}')
            if attempt < PREDICT_RETRIES:
                time.sleep(2)
            continue

        if targets:
            vecs, _ = parse_one(ans, targets)
            if all(v is None for v in vecs.values()):
                last = f'answered but nothing parsable ({len(ans)} chars)'
                print(f'  [{i}/{n}] {uid} {label} attempt {attempt}: {last}')
                if attempt < PREDICT_RETRIES:
                    time.sleep(1)
                    continue
        return ans
    return None


def _run_stage2(runner, uid, stage1, args, i, n, cache):
    """The behavioural analysis: one call per participant, shared by all targets.

    Stage 1 needs no call at all -- `stage1` is the deterministic verbalizer
    output, so it cannot contain a feature the measurement did not produce.

    A failure is never cached. Caching `None` meant one exhausted stage-2 call
    silently disqualified that participant on all three targets, and because
    the failures were not random the survivors were no longer the sample that
    was drawn: in one run the income majority class itself changed between the
    200 sampled and the 56 that survived.
    """
    if cache.get(uid):
        return cache[uid]
    free = {'type': 'object'}          # no schema: this stage returns prose
    s2 = _run_with_ladder(runner, build_stage2_prompt(stage1, args.role),
                          free, 0.1, 'stage2', i, n, uid,
                          num_predict=STAGE2_NUM_PREDICT)
    if s2:
        cache[uid] = s2
        with open(CHAIN_OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'user_id': uid, 'model': MODEL_NAME,
                                'stage1': stage1, 'stage2': s2},
                               ensure_ascii=False) + '\n')
    return s2

def _load_chain_cache() -> dict[str, str]:
    """Reuse completed Qwen behavioural analyses after an interrupted run."""
    cache: dict[str, str] = {}
    if not CHAIN_OUT.exists():
        return cache
    with open(CHAIN_OUT, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(rec.get('user_id', '')).strip()
            stage2 = str(rec.get('stage2', '') or '').strip()
            model = str(rec.get('model', MODEL_NAME)).strip()
            if uid and stage2 and model == MODEL_NAME:
                cache[uid] = stage2
    return cache


def stage_predict(args) -> None:
    print('=' * 70)
    mode = ('chained: indicators -> behavioural analysis -> inference'
            if args.hierarchical else 'single prompt (no behavioural stage)')
    print(f'Stage 3 — inference ({args.backend}, {MODEL_NAME}, '
          f'{args.n_samples} sample(s) per prompt)')
    print(f'  Reasoning: {mode}')
    print(f'  Context/output budgets: num_ctx={NUM_CTX}, '
          f'stage2={STAGE2_NUM_PREDICT}, stage3={NUM_PREDICT}')
    print(f'  Existing prompt input: {PROMPTS_OUT}')
    print(f'  Prediction checkpoint: {PREDS_OUT}')
    print(f'  Behavioural chain: {CHAIN_OUT}')
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
    required_prompt_cols = {'uid', 'targets', 'desc', 'prompt_hash'}
    missing_prompt_cols = required_prompt_cols.difference(prompts.columns)
    if missing_prompt_cols:
        sys.exit(
            f'{PROMPTS_OUT} is missing required columns: '
            f'{sorted(missing_prompt_cols)}'
        )
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
    fallbacks = 0
    if args.hierarchical and args.restart and CHAIN_OUT.exists():
        CHAIN_OUT.unlink()
    chain_cache: dict[str, str] = (
        _load_chain_cache() if args.hierarchical and not args.restart else {}
    )
    if chain_cache:
        print(f'  Reused {len(chain_cache)} completed behavioural analyses ' 
              f'from {CHAIN_OUT}.')

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        targets = list(row.targets)
        schema = response_schema(targets)
        outputs, last_error = [], None
        s1 = str(getattr(row, 'desc', '') or '')
        s2 = None

        chained = args.hierarchical
        if args.hierarchical:
            s2 = _run_stage2(runner, row.uid, s1, args, i, len(todo), chain_cache)
            if not s2:
                # Falling back keeps the participant in the sample. Dropping
                # them would bias it, since stage-2 failures are not random.
                chained = False
                fallbacks += 1
                print(f'  [{i}/{len(todo)}] {row.uid} behavioural stage failed; '
                      'falling back to the single-prompt form')

        for k in range(args.n_samples):
            temp = 0.1 if args.n_samples == 1 else (0.2 + 0.5 * k / max(args.n_samples - 1, 1))
            prompt = (build_stage3_prompt(s1, s2, targets, args.role,
                                          args.context)
                      if chained else
                      build_flat_prompt(s1, targets, args.role, args.context))
            ans = _run_with_ladder(runner, prompt, schema, temp,
                                   f'sample {k + 1}', i, len(todo), row.uid,
                                   targets=targets)
            if ans:
                outputs.append(ans)
            else:
                last_error = last_error or 'no answer at any rung'

        rec = {'user_id': row.uid, 'prompt_hash': row.prompt_hash,
               'model': MODEL_NAME, 'run_tag': RUN_TAG,
               'targets': targets, 'raw_outputs': outputs}
        if args.hierarchical:
            rec['stage2'] = s2
            rec['chained'] = bool(chained)      # so the fallbacks are auditable
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
    if fallbacks:
        print(f'  {fallbacks}/{len(todo)} prompts fell back to the single-prompt '
              'form because the behavioural stage returned nothing.')
        print('  These are recorded with chained=false; report them separately '
              'rather than mixing them into the chained condition.')
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
            if node is None:
                # models rename the keys freely; only income had aliases
                for alias in {
                    'gender': ('sex',),
                    'age_group': ('age', 'age_range', 'age_band', 'agegroup'),
                    'income_level': ('household_income_level', 'income',
                                     'household_income', 'income_band'),
                }.get(t, ()):
                    if data.get(alias) is not None:
                        node = data[alias]
                        break
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
    global NUM_CTX, NUM_PREDICT, STAGE2_NUM_PREDICT, THINK_LEVEL, USE_SCHEMA

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        'stage',
        choices=['verbalize', 'predict', 'parse', 'evaluate', 'all'],
        help=(
            'Run verbalize explicitly to build a prompt file (needed for '
            '--target-mode joint). "all" means predict, parse, and evaluate; '
            'it does not verbalize, so an existing prompt file is reused.'
        ),
    )
    ap.add_argument('--features-file', type=Path, default=FEATURES_FILE)
    ap.add_argument('--sample-size', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--target-mode', choices=['per_target', 'joint'],
                    default='per_target')
    ap.add_argument('--flat', dest='hierarchical', action='store_false',
                    help='skip the behavioural stage: indicators straight to '
                         'inference (ablation)')
    ap.set_defaults(hierarchical=True)
    ap.add_argument('--no-context', dest='context', action='store_false',
                    help='withhold the Swiss reference points (ablation)')
    ap.set_defaults(context=True)
    ap.add_argument('--role', type=str, default=ROLE,
                    help='persona prepended to every prompt; "" to drop it')
    ap.add_argument('--n-samples', type=int, default=3,
                    help='self-consistency samples for stage 3; 1 disables it')
    ap.add_argument('--backend', choices=['ollama', 'hf'], default=BACKEND)
    ap.add_argument('--num-ctx', type=int, default=NUM_CTX)
    ap.add_argument('--num-predict', type=int, default=NUM_PREDICT)
    ap.add_argument('--stage2-num-predict', type=int, default=STAGE2_NUM_PREDICT)
    ap.add_argument('--think', default=THINK_LEVEL,
                    choices=['off', 'low', 'medium', 'high', 'default'],
                    help="qwen3 takes a boolean here, not a level: 'off' "
                         "disables the reasoning channel; 'default' leaves "
                         "it unbounded, which is what returns empty answers.")
    ap.add_argument('--schema', dest='use_schema', action='store_true')
    ap.add_argument('--no-schema', dest='use_schema', action='store_false')
    ap.set_defaults(use_schema=USE_SCHEMA)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--restart', action='store_true')
    args = ap.parse_args()

    NUM_CTX, NUM_PREDICT, STAGE2_NUM_PREDICT, THINK_LEVEL, USE_SCHEMA = (
        args.num_ctx, args.num_predict, args.stage2_num_predict,
        args.think, args.use_schema)
    RETRY_LADDER[0] = {'num_predict': NUM_PREDICT, 'use_schema': USE_SCHEMA,
                       'think': THINK_LEVEL}
    # rungs 2 and 3 hardcoded 'default', so one failed call silently
    # re-enabled unbounded reasoning for the rest of that prompt
    for _rung in RETRY_LADDER[1:]:
        _rung['think'] = THINK_LEVEL

    # 'all' still does not verbalize: it reuses the existing prompt file so
    # that a model comparison runs on identical prompts. Ask for verbalize by
    # name when a new prompt file is wanted, e.g. for --target-mode joint.
    if args.stage == 'verbalize':
        stage_verbalize(args)
        return
    if args.stage in ('predict', 'all'):
        stage_predict(args)
    if args.stage in ('parse', 'all'):
        stage_parse()
    if args.stage in ('evaluate', 'all'):
        stage_evaluate()


if __name__ == '__main__':
    main()
