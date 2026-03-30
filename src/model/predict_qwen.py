import json
import time
import re
import sys
import signal
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------
# Config
# --------------------------------------
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

PROMPT_TXT = Path("/data/baliu/python_code/data/prompts_v3_1week_age_householdsize_04Mar2026_qwen_clean_v4.txt")
PRED_JSON  = Path("/data/baliu/python_code/data/preds_qwen_age_householdsize_26Mar2026_v2.jsonl")
PRED_CSV   = Path("/data/baliu/python_code/data/preds_qwen_age_householdsize_26Mar2026_v2.csv")

SEP             = "=" * 80
MAX_PROMPT_CHARS = 30000
TIMEOUT_SEC     = 600
COOLDOWN_EVERY  = 20
COOLDOWN_SEC    = 10
MAX_NEW_TOKENS  = 2048
MODEL_MAX_CTX   = 32768  # Qwen2-7B supports 32k

SYSTEM_PROMPT = (
    "You are a mobility behavior and socioeconomic status inference analyst. "
    "Based only on the mobility behaviour and nearby spatial context described below, "
    "please choose exactly one of the following categories:\n"
    "household_income_level (monthly CHF): <4000, 4001-8000, 8001-12000, 12001-16000, 16001+.\n"
    "age_group: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+.\n"
    "gender: male, female, other, prefer not to say.\n"
    "household_size: 1, 2, 3, 4, 5+.\n\n"
    "Guidelines:\n"
    "- This project is conducted in Switzerland. Use Swiss geographic context "
    "(urban/rural distribution, Zürich Zentrum, Oerlikon, public transport, local economic activities).\n"
    "- Use only information explicitly provided in the mobility record.\n"
    "- Base reasoning on mobility intensity, locations visited, transport modes, and activity patterns.\n"
    "- Keep a neutral and factual tone.\n"
    "- Use chain of thought — find multiple evidences and weigh them together.\n\n"
    "Output format:\n"
    "1. Reasoning (no more than 150 words).\n"
    "2. One final line in strict JSON:\n"
    "{\"household_income_level\": \"4001-8000\", \"age_group\": \"45-54\", \"gender\": \"other\", "
    "\"household_size\": 3, "
    "\"prediction_rationale\": [\"Evidence 1\", \"Evidence 2\", \"Evidence 3\"], "
    "\"prediction_confidence\": \"high\"}\n\n"
    "- prediction_rationale: list of 3-6 short strings, each a concrete mobility/semantic evidence.\n"
    "- prediction_confidence: one of low, medium, high.\n"
    "- No extra keys, no markdown, no commentary outside JSON."
)

# --------------------------------------
# Timeout
# --------------------------------------
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

# --------------------------------------
# Json extraction (robust)
# --------------------------------------
def extract_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        candidate = m.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # remove trailing commas
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # fallback: fix bare keys and single quotes
        fixed = re.sub(r'(?<!")(\b[a-zA-Z_]\w*\b)(?!")\s*:', r'"\1":', candidate)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
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
print(f"📦 Loaded {len(prompts)} prompts")

done_users = load_done_users(PRED_JSON)
print(f"🔁 Found {len(done_users)} completed users")

# --------------------------------------
# Load model + tokenizer once
# --------------------------------------
print("🔧 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",        # ← force GPU 0 explicitly, not "auto"
    dtype=torch.float16,
    trust_remote_code=True,
    offload_buffers=True,       # ← fixes the buffer warning
)
model.eval()
print("✅ Model loaded")
print(f"🖥️  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

# Sanity check — abort early if model is not on GPU
assert torch.cuda.memory_allocated() > 1e8, "❌ Model not on GPU! Check nvidia-smi."

# --------------------------------------
# Main loop
# --------------------------------------
try:
    for i, prompt in enumerate(prompts, 1):

        m = re.search(r"User:\s*(\S+)", prompt)
        if not m:
            print("⚠️  Cannot find user_id, skipping")
            continue

        user_id = m.group(1)

        if user_id in done_users:
            print(f"⏭️  Skipping {user_id}")
            continue

        print(f"\n🔮 [{i}/{len(prompts)}] Predicting {user_id}")

        if len(prompt) > MAX_PROMPT_CHARS:
            print("⚠️  Prompt too long (chars), skipping")
            continue

        try:
            signal.alarm(TIMEOUT_SEC)

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

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )

            signal.alarm(0)

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            result = extract_json(decoded)
            result["user_id"] = user_id

            with open(PRED_JSON, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            done_users.add(user_id)
            print("✅ Done")

        except TimeoutError:
            print("⏰ Timeout")
            signal.alarm(0)
            continue

        except Exception as e:
            print(f"❌ Error: {e}")
            signal.alarm(0)
            time.sleep(2)
            continue

        if i % COOLDOWN_EVERY == 0:
            print("🧹 Cooling down...")
            time.sleep(COOLDOWN_SEC)

except KeyboardInterrupt:
    print("\n🛑 Interrupted safely.")
    sys.exit(0)

# --------------------------------------
# Save CSV snapshot
# --------------------------------------
if PRED_JSON.exists():
    df = pd.read_json(PRED_JSON, lines=True)
    df.to_csv(PRED_CSV, index=False)
    print(f"\n🎉 Saved → {PRED_CSV}")
else:
    print("⚠️  No predictions generated")
