# Mobility LLM Thesis

Sociodemographic inference (age, gender, household size, income) 
from one week of GPS staypoint data using zero-shot LLM inference.

## Models
- Qwen2-7B-Instruct (32k context)
- DeepSeek-LLM-7B-Chat (4k context)

## Project Structure
```
src/
├── data/       # Data loading and cleaning
├── geo/        # POI context and geocoding  
├── features/   # Mobility feature extraction
├── prompt/     # Compact prompt builder
├── model/      # Model loading and inference
└── utils/      # Shared I/O utilities
scripts/        # Entry point scripts
config/         # Parameters (config.yaml)
notebooks/      # EDA only
data/           # Not tracked in git
logs/           # Not tracked in git
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
```
