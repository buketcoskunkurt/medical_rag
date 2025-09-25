import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw_dir = root / 'data' / 'raw'
out_file = root / 'data' / 'pubmed.jsonl'

raw_files = sorted(raw_dir.glob('*.jsonl'))
print('Found', len(raw_files), 'raw files')
seen = set()
written = 0
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open('w', encoding='utf-8') as outf:
    for rf in raw_files:
        print('Processing', rf)
        with rf.open('r', encoding='utf-8', errors='ignore') as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _id = str(obj.get('id') or obj.get('pmid') or '')
                if not _id:
                    continue
                if _id in seen:
                    continue
                seen.add(_id)
                # normalize minimal fields
                out_obj = {
                    'id': _id,
                    'title': obj.get('title',''),
                    'text': obj.get('text',''),
                    'url': obj.get('url','')
                }
                outf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                written += 1

print('Wrote', written, 'unique records to', out_file)