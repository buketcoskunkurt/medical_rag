#!/usr/bin/env python3
"""Merge per-topic OpenAlex JSONL files in data/raw into data/openalex.jsonl

Deduplicate by DOI when present, otherwise by id.
"""
import json
from pathlib import Path


def main():
    raw_dir = Path('data/raw')
    out = Path('data/openalex.jsonl')
    files = sorted(raw_dir.glob('openalex_*.jsonl'))
    seen = set()
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for fp in files:
            print('Merging', fp)
            for line in fp.open('r', encoding='utf-8'):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                key = (d.get('doi') or d.get('id') or '').lower()
                if not key:
                    # fallback to title-based short key
                    key = (d.get('title') or '')[:200].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                rec = {
                    'id': d.get('id'),
                    'title': d.get('title'),
                    'text': d.get('text'),
                    'url': d.get('url'),
                    'source': d.get('source', 'openalex')
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                written += 1

    print(f'Wrote {written} merged OpenAlex records to {out}')


if __name__ == '__main__':
    main()
