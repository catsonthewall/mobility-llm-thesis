# ---------------------------------------------------------------------------
# src/utils/io.py
# Shared I/O utilities: load done users, append JSONL, save CSV snapshot
# ---------------------------------------------------------------------------

import json
import pandas as pd
from pathlib import Path


def load_done_users(path: Path) -> set:
    """
    Read a JSONL prediction file and return the set of already-predicted user_ids.
    Used for resume / checkpoint logic.
    """
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


def append_jsonl(path: Path, record: dict) -> None:
    """Append one prediction record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv_snapshot(jsonl_path: Path, csv_path: Path) -> None:
    """Convert JSONL predictions to a CSV snapshot."""
    if not jsonl_path.exists():
        print(f"⚠️  {jsonl_path} does not exist, skipping CSV save")
        return
    df = pd.read_json(jsonl_path, lines=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV snapshot saved → {csv_path} ({len(df)} rows)")


def load_prompts(path: Path, sep: str = "=" * 80) -> list[str]:
    """
    Load prompts from a text file separated by SEP lines.
    Returns a list of prompt strings, one per user.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    prompts = [p.strip() for p in raw.split(sep) if p.strip()]
    print(f"📦 Loaded {len(prompts)} prompts from {path.name}")
    return prompts
