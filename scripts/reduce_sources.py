#!/usr/bin/env python3
"""Reduce and clean source JSONL files (OpenAlex, PubMed).

Usage:
  python scripts/reduce_sources.py --in data/openalex.jsonl --out data/openalex.jsonl

This script will create a .bak of the original out file if it exists, then
filter out records with empty `text` fields and deduplicate by doi/id/title.
"""
import argparse
import json
from pathlib import Path


def norm_key(d):
    doi = (d.get('doi') or d.get('DOI') or '')
    if doi:
        return ('doi', doi.strip().lower())
    pid = d.get('id') or d.get('pmid') or d.get('pmcid')
    if pid:
        return ('id', str(pid).strip())
    title = (d.get('title') or '')[:200].strip().lower()
    if title:
        return ('title', title)
    return (None, None)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def reduce_file(in_path: Path, out_path: Path, backup=True):
    items = read_jsonl(in_path)
    seen = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + '.bak')
        out_path.replace(bak)
    written = 0
    with out_path.open('w', encoding='utf-8') as fh:
        for d in items:
            text = (d.get('text') or d.get('abstract') or '')
            if not text or not str(text).strip():
                continue
            ktype, kval = norm_key(d)
            if not kval:
                # skip records with no id/title/doi
                continue
            key = f"{ktype}:{kval}"
            if key in seen:
                continue
            seen.add(key)
            rec = {
                'id': d.get('id'),
                'title': d.get('title') or '',
                'text': text,
                'url': d.get('url') or d.get('link') or None,
                'source': d.get('source') or None,
                'doi': d.get('doi') or d.get('DOI') or None,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            written += 1
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inpath', required=True)
    p.add_argument('--out', dest='outpath', required=True)
    args = p.parse_args()
    in_p = Path(args.inpath)
    out_p = Path(args.outpath)
    n = reduce_file(in_p, out_p)
    print(f"Reduced {in_p} -> {out_p}: wrote {n} records")


if __name__ == '__main__':
    main()
