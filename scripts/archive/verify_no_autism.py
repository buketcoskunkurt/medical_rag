"""Archived: Verify that 'autism' does not appear in data/pubmed.jsonl or data/meta.parquet.

This file was archived per user request. Original behavior:
- Scans JSONL and meta.parquet for 'autism' occurrences and prints counts/sample ids.
"""
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'pubmed.jsonl'
META_PATH = ROOT / 'data' / 'meta.parquet'

pat = re.compile(r'\bautis\w*\b', re.I)

def scan_jsonl(p: Path):
    if not p.exists():
        print(f'[verify] {p} not found')
        return 0, []
    total = 0
    hits = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = ' '.join(str(obj.get(k, '') or '') for k in ('title', 'text', 'abstract', 'snippet'))
            if pat.search(text):
                hits.append(obj.get('id') or obj.get('pmid') or f'line:{total}')
    return len(hits), hits[:10]

def scan_meta(p: Path):
    if not p.exists():
        print(f'[verify] {p} not found')
        return 0, []
    df = pd.read_parquet(p)
    hits = df[df['text'].str.contains(pat, na=False, regex=True)]
    ids = hits['id'].tolist() if 'id' in hits else []
    return len(ids), ids[:10]

if __name__ == '__main__':
    jcount, jsample = scan_jsonl(DATA_PATH)
    mcount, msample = scan_meta(META_PATH)
    print(f'[verify] jsonl autism_hits={jcount} sample_ids={jsample}')
    print(f'[verify] meta autism_hits={mcount} sample_ids={msample}')
