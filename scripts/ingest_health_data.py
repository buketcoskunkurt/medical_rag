"""
scripts/ingest_health_data.py

Simple ingestion helper to convert a folder of health documents into a JSONL
file compatible with the existing pipeline (fields: id, title, url, text).

Usage:
  python scripts\ingest_health_data.py --input-dir path/to/files --out data/health.jsonl

It supports .jsonl (appends), .json, .txt and .html files. HTML will be stripped of tags.
"""
import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup


def extract_text_from_html(path: Path) -> str:
    html = path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    # remove scripts/styles
    for s in soup(['script', 'style']):
        s.decompose()
    text = ' '.join(soup.stripped_strings)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out', default='data/health.jsonl')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open('a', encoding='utf-8') as fo:
        for fp in sorted(input_dir.rglob('*')):
            if fp.is_dir():
                continue
            try:
                if fp.suffix.lower() in ['.jsonl']:
                    # append all lines
                    for line in fp.read_text(encoding='utf-8', errors='ignore').splitlines():
                        if not line.strip():
                            continue
                        # ensure minimal fields
                        try:
                            j = json.loads(line)
                        except Exception:
                            continue
                        rec = {
                            'id': str(j.get('id', fp.stem)),
                            'title': j.get('title', '') or fp.stem,
                            'url': j.get('url', ''),
                            'text': j.get('text', '') or j.get('body', '') or ''
                        }
                        fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        written += 1
                elif fp.suffix.lower() in ['.json']:
                    j = json.loads(fp.read_text(encoding='utf-8', errors='ignore'))
                    rec = {
                        'id': str(j.get('id', fp.stem)),
                        'title': j.get('title', '') or fp.stem,
                        'url': j.get('url', ''),
                        'text': j.get('text', '') or j.get('body', '') or ''
                    }
                    fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                elif fp.suffix.lower() in ['.txt']:
                    txt = fp.read_text(encoding='utf-8', errors='ignore')
                    rec = {'id': fp.stem, 'title': fp.stem, 'url': '', 'text': txt}
                    fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                elif fp.suffix.lower() in ['.html', '.htm']:
                    txt = extract_text_from_html(fp)
                    rec = {'id': fp.stem, 'title': fp.stem, 'url': '', 'text': txt}
                    fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                else:
                    # skip unknown
                    continue
            except Exception as e:
                print('skip', fp, 'err', e)
    print(f'Wrote {written} records to {out_path}')


if __name__ == '__main__':
    main()
