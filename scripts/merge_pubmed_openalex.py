#!/usr/bin/env python3
"""Merge PubMed and OpenAlex JSONL into a deduplicated combined JSONL.

Dedup keys: DOI (if present), PMID/id, otherwise normalized title prefix.
Normalizes output fields to: id, title, text, url, source, doi
"""
import json
from pathlib import Path


def norm_key(d):
    doi = (d.get('doi') or d.get('DOI') or '')
    if doi:
        return ('doi', doi.strip().lower())
    # pubmed may have pmid under 'id' if numeric
    pid = d.get('id') or d.get('pmid') or d.get('pmcid')
    if pid:
        return ('id', str(pid).strip())
    title = (d.get('title') or '')[:200].strip().lower()
    return ('title', title)


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.open('r', encoding='utf-8'):
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def main():
    pubmed = read_jsonl('data/pubmed.jsonl')
    openalex = read_jsonl('data/openalex.jsonl')
    sources = [('pubmed', pubmed), ('openalex', openalex)]

    seen = set()
    out_path = Path('data/combined.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open('w', encoding='utf-8') as fh:
        for name, items in sources:
            for d in items:
                # normalize fields
                rec = {
                    'id': d.get('id') or d.get('pmid') or d.get('pmcid') or d.get('doi') or None,
                    'title': d.get('title') or d.get('title') or '',
                    'text': d.get('text') or d.get('abstract') or '',
                    'url': d.get('url') or d.get('link') or None,
                    'source': d.get('source') or name,
                    'doi': d.get('doi') or d.get('DOI') or None,
                }
                ktype, kval = norm_key(rec)
                key = f"{ktype}:{kval}"
                if not kval:
                    # skip records without any id/title
                    continue
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                written += 1

    print(f"Wrote {written} combined records to {out_path}")


if __name__ == '__main__':
    main()
