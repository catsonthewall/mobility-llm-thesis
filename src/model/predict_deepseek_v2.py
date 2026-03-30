import json
import time
import re
import sys
import signal
import gc
import os
import threading
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------
# Config
# --------------------------------------
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
# /data/baliu/python_code/data/prompts_v4_compact_28Mar2026.txt
#PROMPT_TXT = Path("/data/baliu/python_code/data/prompts_v3_1week_age_householdsize_04Mar2026_qwen_clean_v4.txt")
PROMPT_TXT = Path("/data/baliu/python_code/data/prompts_v4_compact_28Mar2026.txt")

PRED_JSON  = Path("/data/baliu/python_code/data/preds_deepseek_v3_29Mar2026.jsonl")
PRED_CSV   = Path("/data/baliu/python_code/data/preds_deepseek_v3_29Mar2026.csv")

SEP              = "=" * 80
MAX_PROMPT_CHARS = 28000
TIMEOUT_SEC      = 300
COOLDOWN_EVERY   = 20
COOLDOWN_SEC     = 10
MAX_NEW_TOKENS   = 350
MODEL_MAX_CTX    = 4096

SYSTEM_PROMPT = (
    "You are a socioeconomic inference analyst for a Swiss mobility research project.\n"
    "You will receive one week of GPS staypoint data for one user.\n"
    "Your task: infer the user's demographics from mobility patterns.\n\n"

    "=== ALLOWED VALUES (you MUST use exactly these, no other values) ===\n"
    "age_group:             0-17 | 18-24 | 25-34 | 35-44 | 45-54 | 55-64 | 65+\n"
    "gender:                male | female | non-binary\n"
    "household_size:        1 | 2 | 3 | 4 | 5+\n"
    "household_income_level: <4000 | 4001-8000 | 8001-12000 | 12001-16000 | 16001+\n\n"

    "=== CRITICAL RULES ===\n"
    "- You MUST output a value for EVERY field — no nulls, no 'unknown', no 'cannot determine'\n"
    "- If evidence is weak, make your best guess using Swiss base rates\n"
    "- gender MUST be either 'male' or 'female' or 'non-binary' — no other values accepted\n"
    "- All values MUST come from the allowed list above — no exceptions\n\n"

    "=== SWISS BASE RATES (use as prior when evidence is weak) ===\n"
    "- Most common age group: 35-44\n"
    "- Gender split: ~50% male, ~50% female, ~1-5% non-binary\n"
    "- Most common household size: 2\n"
    "- Most common income: 4001-8000 CHF/month\n"
    "- Zürich postcodes 8000-8099: higher income (8001-12000 or above)\n"
    "- Car as only transport + rural area: likely higher income\n"
    "- Heavy public transport use: typical Swiss urban, middle income\n"
    "- Regular 08:00-17:00 weekday pattern: employed adult (25-54)\n"
    "- Frequent short trips, daytime only: possibly 65+\n"
    "- Very late night activity: likely 18-34\n\n"

    "=== OUTPUT FORMAT ===\n"
    "Step 1 — Write reasoning (max 100 words). Identify 3-5 mobility clues and what they suggest.\n"
    "Step 2 — Write ONE line of JSON, exactly like this example:\n"
    "{\"age_group\": \"35-44\", \"gender\": \"male\", \"household_size\": \"2\", "
    "\"household_income_level\": \"4001-8000\", "
    "\"prediction_rationale\": [\"Clue 1\", \"Clue 2\", \"Clue 3\"], "
    "\"prediction_confidence\": \"medium\"}\n\n"
    "=== FINAL REMINDERS ===\n"
    "- prediction_confidence: low | medium | high\n"
    "- prediction_rationale: list of 3-5 short strings\n"
    "- gender must be 'male' or 'female' or 'non-binary' — NEVER 'other' or 'prefer not to say'\n"
    "- Do NOT write anything after the JSON line\n"
    "- Do NOT use markdown code fences"
)

# --------------------------------------
# Timeout using threading
# --------------------------------------
class TimeoutError(Exception):
    pass

def generate_with_timeout(model, inputs, gen_kwargs, timeout=TIMEOUT_SEC):
    result = {"output": None, "error": None}

    def _generate():
        try:
            with torch.no_grad():
                result["output"] = model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_generate, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"Generation exceeded {timeout}s")
    if result["error"]:
        raise result["error"]
    return result["output"]

# --------------------------------------
# JSON extraction + value validation
# --------------------------------------
ALLOWED = {
    "age_group":             {"0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"},
    "gender":                {"male", "female", "non-binary"},
    "household_size":        {"1", "2", "3", "4", "5+"},
    "household_income_level":{"<4000", "4001-8000", "8001-12000", "12001-16000", "16001+"},
}

DEFAULTS = {
    "age_group": "35-44",
    "gender": "male",
    "household_size": "2",
    "household_income_level": "4001-8000",
}

def validate_and_fix(result: dict) -> dict:
    """Force all prediction fields to valid allowed values."""
    for field, allowed_vals in ALLOWED.items():
        val = str(result.get(field, "")).strip().lower()
        # normalize common variants
        val = val.replace("prefer not to say", "female")
        val = val.replace("other", "male")
        val = val.replace("5 or more", "5+")
        val = val.replace(">16000", "16001+")
        val = val.replace("16000+", "16001+")
        # check if valid
        if val in allowed_vals:
            result[field] = val
        else:
            # try to find closest match
            matched = False
            for av in allowed_vals:
                if av in val or val in av:
                    result[field] = av
                    matched = True
                    break
            if not matched:
                result[field] = DEFAULTS[field]
                result[f"{field}_defaulted"] = True
    return result

def extract_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        candidate = m.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
            return validate_and_fix(parsed)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r'(?<!")(\b[a-zA-Z_]\w*\b)(?!")\s*:', r'"\1":', candidate)
        fixed = fixed.replace("'", '"')
        try:
            parsed = json.loads(fixed)
            return validate_and_fix(parsed)
        except json.JSONDecodeError:
            continue
    return {"raw_output": text, "error": "json_parse_failed"}

# --------------------------------------
# Load done users
# --------------------------------------
def load_done_users(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "user_id" in obj:
                    done.add(obj["user_id"])
            except Exception:
                continue
    return done

# --------------------------------------
# Load prompts
# --------------------------------------
with open(PROMPT_TXT, "r", encoding="utf-8") as f:
    raw = f.read()

prompts = [p.strip() for p in raw.split(SEP) if p.strip()]
print(f" Loaded {len(prompts)} prompts")

done_users = load_done_users(PRED_JSON)
print(f"Found {len(done_users)} completed users")

# --------------------------------------
# Load model + tokenizer
# --------------------------------------
print("🔧 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

allocated = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded — GPU: {allocated:.1f} GB allocated")
if allocated < 0.1:
    print("WARNING: Model may not be on GPU — check nvidia-smi")

# --------------------------------------
# Main loop
# --------------------------------------
try:
    for i, prompt in enumerate(prompts, 1):

        m = re.search(r"User:\s*(\S+)", prompt)
        if not m:
            print(" Cannot find user_id, skipping")
            continue

        user_id = m.group(1)

        if user_id in done_users:
            print(f" Skipping {user_id}")
            continue

        print(f"\n [{i}/{len(prompts)}] Predicting {user_id}")

        if len(prompt) > MAX_PROMPT_CHARS:
            print("  Prompt too long (chars), skipping")
            continue

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]

            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            max_input_tokens = MODEL_MAX_CTX - MAX_NEW_TOKENS
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(model.device)

            n_tokens = inputs["input_ids"].shape[1]
            if n_tokens >= max_input_tokens:
                print(f" Truncated to {max_input_tokens} tokens")

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.2,       # lower = more deterministic, better category adherence
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.05,
            )

            outputs = generate_with_timeout(model, inputs, gen_kwargs, timeout=TIMEOUT_SEC)

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            del inputs, outputs, gen_ids
            torch.cuda.empty_cache()
            gc.collect()

            result = extract_json(decoded)
            result["user_id"] = user_id

            with open(PRED_JSON, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            done_users.add(user_id)
            # show prediction inline for quick monitoring
            age = result.get("age_group", "?")
            gender = result.get("gender", "?")
            hh = result.get("household_size", "?")
            inc = result.get("household_income_level", "?")
            print(f"Done → age:{age} gender:{gender} hh:{hh} income:{inc}")

        except TimeoutError:
            print(f"Timeout after {TIMEOUT_SEC}s")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        except Exception as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
            continue

        if i % COOLDOWN_EVERY == 0:
            print("🧹 Cooling down...")
            time.sleep(COOLDOWN_SEC)

except KeyboardInterrupt:
    print("\n Interrupted safely.")
    sys.exit(0)

# --------------------------------------
# Save csv snapshot
# --------------------------------------
if PRED_JSON.exists():
    df = pd.read_json(PRED_JSON, lines=True)
    df.to_csv(PRED_CSV, index=False)
    print(f"\n Saved → {PRED_CSV}")
else:
    print("No predictions generated")
