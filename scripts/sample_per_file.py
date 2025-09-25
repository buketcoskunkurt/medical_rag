#!/usr/bin/env python3
"""Sample up to N usable (non-empty text) records per file from data/raw and write combined JSONL.

Usage:
  python scripts/sample_per_file.py --per-file 200 --out data/combined_random_perfile.jsonl
"""
import argparse
import glob
import json
import random
from pathlib import Path


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def normalize_rec(d, source_hint=None):
    text = d.get('text') or d.get('abstract') or ''
    return {
        'id': d.get('id'),
        'title': d.get('title') or '',
        'text': text,
        'url': d.get('url') or d.get('link') or None,
        'source': d.get('source') or source_hint,
        'doi': d.get('doi') or d.get('DOI') or None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--per-file', type=int, default=200)
    p.add_argument('--out', default='data/combined_random_perfile.jsonl')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    raw_files = sorted(glob.glob('data/raw/*_*.jsonl'))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    per_file_counts = {}

    with outp.open('w', encoding='utf-8') as fh:
        for fp in raw_files:
            fname = Path(fp).name
            parts = fname.split('_', 1)
            source = parts[0] if len(parts) == 2 else None
            items = read_jsonl(fp)
            usable = []
            for d in items:
                text = (d.get('text') or d.get('abstract') or '')
                if text and str(text).strip():
                    usable.append(d)
            if not usable:
                per_file_counts[fname] = 0
                continue
            k = min(args.per_file, len(usable))
            sampled = random.sample(usable, k) if k < len(usable) else usable
            for d in sampled:
                rec = normalize_rec(d, source_hint=source)
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total_written += 1
            per_file_counts[fname] = len(sampled)

    print(f'Wrote {total_written} records to {outp}')
    for k, v in sorted(per_file_counts.items()):
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
