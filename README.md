# Sociodemographic Inference from GNSS Mobility Trajectories

Code accompanying the MSc thesis *[thesis title]* (University of Zurich, 2026).
The repository contains the full path from raw MOBIS staypoints to
sociodemographic predictions: mobility-indicator construction, indicator–target
association analysis, feature-based classification, and LLM-based inference from
two language representations of the same trajectories.

**Thesis:

---

## Data availability

**The MOBIS data are not included in this repository and cannot be
redistributed.** The dataset is governed by the data protection terms of the
MOBIS study; access must be requested from the study holders. See
Molloy et al. (2022), *The MOBIS dataset: a large GPS dataset of mobility
behaviour in Switzerland*, Transportation.

The OpenStreetMap POI layers used for geographic enrichment are derived from
public OSM extracts and can be rebuilt with `<script>` (see
[Pipeline](#pipeline), step 2).

Every script reads its inputs from the paths configured in `config.yaml`; no
data file is committed.

---

## Repository structure

```
.
├── config.yaml                  # all paths, thresholds and seeds
├── requirements.txt
├── src/
│   ├── preprocessing/           # cleaning, windowing, cohort linkage
│   ├── enrichment/              # reverse geocoding, POI build, POI proximity
│   ├── indicators/              # the 41 mobility indicators
│   ├── models/                  # feature-based classifiers
│   ├── llm/                     # verbalization, prompting, response parsing
│   └── evaluation/              # metrics, association analysis, tables
├── scripts/                     # entry points, numbered in run order
├── prompts/                     # prompt templates used in the LLM chapter
└── outputs/                     # generated tables and figures (git-ignored)
```

---

## Installation

```bash
git clone https://github.com/catsonthewall/mobility-llm-thesis.git
cd mobility-llm-thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or later. The feature-based models require only the packages in
`requirements.txt`. The LLM stage additionally requires a local
[Ollama](https://ollama.com) installation and the model weights:

```bash
ollama pull gpt-oss:20b
ollama pull qwen3:30b-a3b
```

All language models run locally; no trajectory data is sent to an external
service at any point.

---

## Pipeline

Scripts are numbered in run order and are idempotent: each writes its output to
`outputs/` and skips work that is already present unless `--force` is passed.

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `scripts/00_read_and_filter.py;01_merge_mobis_data.py ` | raw MOBIS staypoints, survey table | cleaned event table |
| 2 | `scripts/02_build_poi_layer.py` | OSM extract | projected POI GeoPackage (EPSG:2056) |
| 3 | `scripts/03_enrich_context.py` | event table, POI layer | reverse-geocoded events with POI context |
| 4 | `scripts/04_build_indicators.py` | enriched events | 41-dimensional indicator matrix |
| 5 | `scripts/05_association_analysis/051_age_association_analysis.py;052_gender_association_analysis.py;053_income_association_analysis.py;` | indicator matrix, targets | Spearman / rank-biserial associations, BH-adjusted |
| 6 | `scripts/06_train_classifiers/061_ml_classification_age.py;062_ml_classification_gender.py;063_ml_classification_income.py;` | indicator matrix, targets | fitted models, test-split metrics | evluation |
| 7 | `scripts/07_hicot_verbalized_indicator_pipeline.py` | indicator | verbalized indicators,  | inferred results | metrics | 
| 8 | `scripts/08_hicot_daily_pipeline.py` | textual daily diary representations, serialized diaries, prompts, | inferred results | metrics | 

Run the whole pipeline:

```bash
python scripts/01_prepare_cohort.py
# ... steps 2-9 ...
python scripts/10_make_tables.py
```

Or a single stage, for example the classifiers only:

```bash
python scripts/06_train_classifiers --target income --model random_forest
```

---

## Configuration

All parameters that affect results live in `config.yaml`, not in the code:

```yaml
window:
  length_days: 7days or 4 weeks            # analysis window per participant
  min_tracking_days: 50     # retention threshold
split:
  test_fraction: 0.2
  random_seed: 42           # governs split, folds and model initialization
poi:
  search_radius_m: 1000
  n_listed: 5               # POIs named per stay point in the diary
llm:
  models: [gpt-oss-20b, qwen3-30b-a3b]
  num_ctx: 8192
  num_predict: 2048
  self_consistency_k: 3
  temperatures: [0.2, 0.45, 0.7]
  max_retries: 3
```

Changing a value here changes every downstream stage consistently. Nothing is
tuned against the test split.

---

## Reproducibility

The feature-based results are exactly reproducible: a single seed governs the
train–test split, the cross-validation folds and model initialization.

The LLM results are **not** bit-reproducible. Self-consistency samples at
non-zero temperature, and GPU generation is not guaranteed identical across
runs even at a fixed seed. Every reported LLM figure is an average over
`self_consistency_k` samples, and the raw responses are written to
`outputs/llm_raw/` so that scoring can be re-run without re-querying the models.

Unparsable responses are recorded as majority-class predictions; the parse rate
is reported alongside every LLM metric.

---

## Outputs

`scripts/07_hicot_verbalized_indicator_pipeline.py`
 `| 8 |scripts/08_hicot_daily_pipeline.py`  `outputs/tables/`, and the association figures in
`outputs/figures/`. File names match the table and figure labels used in the
thesis.

---

