# Mobility LLM Thesis

This project investigates whether mobiity trajectories can reveal sociodemographic characteristics using large language models(LLMs).

Specifically, we aim to infer attributes such as:
age group, 
gender, 
household size, 
household income level
from GPS trajectories data using LLM inference.

## Research Motivation
Human mobility patterns encode rich behavioral siganls. Daily routines- such as commuting, leisure activities, and spatial movement- are strongly associated with sociodemographic characteristics.

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
src/
├── data/       # Data loading, cleaning, and sampling
├── geo/        # POI retrival, reverse geocoding (OSM) 
├── features/   # Mobility feature extraction
├── prompt/     # Compact prompt builder
├── model/      # Model loading and inference
└── utils/      # Shared I/O utilities
scripts/        # Entry point scripts
config/         # Parameters (config.yaml)
notebooks/      # EDA only
data/           # No, for privacy reason
logs/           # Experiment logs
```

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
