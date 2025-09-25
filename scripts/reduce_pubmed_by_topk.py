"""Reduce raw PubMed JSONL files by taking top-K entries per file (preserving relevance order).
Writes output to data/pubmed_reduced.jsonl
Usage: python scripts/reduce_pubmed_by_topk.py --topk 200
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--topk', type=int, default=200)
args = ap.parse_args()

root = Path(__file__).resolve().parents[1]
raw_dir = root / 'data' / 'raw'
out_file = root / 'data' / 'pubmed_reduced.jsonl'
raw_files = sorted(raw_dir.glob('*.jsonl'))

seen = set()
written = 0
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open('w', encoding='utf-8') as outf:
    for rf in raw_files:
        count = 0
        with rf.open('r', encoding='utf-8', errors='ignore') as inf:
            for line in inf:
                if count >= args.topk:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _id = str(obj.get('id') or '')
                if not _id:
                    continue
                if _id in seen:
                    continue
                seen.add(_id)
                out_obj = {
                    'id': _id,
                    'title': obj.get('title',''),
                    'text': obj.get('text',''),
                    'url': obj.get('url',''),
                    'source_file': rf.name
                }
                outf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                written += 1
                count += 1

print(f'Wrote {written} records to {out_file} from {len(raw_files)} files (topk={args.topk})')