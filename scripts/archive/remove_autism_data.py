"""Archived: Filter out records that mention autism from data/pubmed.jsonl.

This script was archived per user request. Original behavior:
- creates a timestamped backup of data/pubmed.jsonl
- writes a filtered file (overwrites data/pubmed.jsonl) that excludes records where
  title/text/abstract/snippet contain the word 'autism' (case-insensitive)
"""
from pathlib import Path
import json
import shutil
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'pubmed.jsonl'

def backup_file(p: Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = p.with_suffix(p.suffix + f'.bak.{ts}')
    shutil.copy2(p, bak)
    return bak

def filter_file(p: Path):
    if not p.exists():
        print(f'ERROR: {p} not found')
        return
    bak = backup_file(p)
    print(f'[remove_autism] backup created: {bak}')

    pat = re.compile(r'\bautis\w*\b', re.I)
    total = 0
    removed = 0
    kept = 0
    tmp_path = p.with_suffix('.filtered.tmp')

    with p.open('r', encoding='utf-8') as rf, tmp_path.open('w', encoding='utf-8') as wf:
        for line in rf:
            total += 1
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # keep malformed lines to avoid accidental data loss
                wf.write(line + '\n')
                kept += 1
                continue

            text = ' '.join(str(obj.get(k, '') or '') for k in ('title', 'text', 'abstract', 'snippet'))
            if pat.search(text):
                removed += 1
                continue
            wf.write(json.dumps(obj, ensure_ascii=False) + '\n')
            kept += 1

    # Replace original with filtered
    tmp_path.replace(p)
    print(f'[remove_autism] total={total} kept={kept} removed={removed}')

if __name__ == '__main__':
    filter_file(DATA_PATH)
