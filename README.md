# Sociodemographic Inference from GNSS Mobility Trajectories

Code accompanying the MSc thesis [thesis title] (University of Zurich, 2026). The repository contains the full path from raw MOBIS staypoints to sociodemographic predictions: mobility-indicator construction, indicator–target association analysis, feature-based classification, and LLM-based inference from two language representations of the same trajectories.

## Research Motivation
Human mobility patterns encode rich behavioral siganls. Daily routines- such as commuting, leisure activities, and spatial movement- are strongly associated with sociodemographic characteristics.


### Data availability

The MOBIS data are not included in this repository and cannot be redistributed. The dataset is governed by the data protection terms of the MOBIS study; access must be requested from the study holders. See Molloy et al. (2022), The MOBIS dataset: a large GPS dataset of mobility behaviour in Switzerland, Transportation.

The OpenStreetMap POI layers used for geographic enrichment are derived from public OSM extracts and can be rebuilt with <script> (see Pipeline, step 2).

Every script reads its inputs from the paths configured in config.yaml; no data file is committed.


## Method Overview
The pipeline consists of:
1. Trajectory processing
2. Geographic context enrichment
3. Prompt construction
4. LLM inference
5. Evaluation

## Models
- Qwen2-7B-Instruct (32k context)
- DeepSeek-LLM-7B-Chat (4k context)
- gpt-oss-20B-Instruct

## Project Structure
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


## Run Inference (on server)
```bash
# Qwen
HF_HOME=/data/baliu/hf_hub \
nohup /data/baliu/venvs/qwen_ft/bin/python src/model/predict_qwen.py \
> logs/predict_qwen.log 2>&1 &

# DeepSeek  
HF_HOME=/data/baliu/hf_hub \
nohup /data/baliu/venvs/deepseek_env/bin/python src/model/predict_deepseek_v2.py \
> logs/predict_deepseek.log 2>&1 &

# gpt-oss-20B
```

## Baselines
To assess the added value of LLMs, we compare against:
 Random forest/ XGBoost
 k-NN
 ....
```
