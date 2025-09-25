#!/usr/bin/env python3
"""Fetch Works from OpenAlex and write records with abstracts to JSONL.

Reconstructs abstract from 'abstract_inverted_index' when present.
"""
import argparse
import json
import time
from pathlib import Path
import requests

BASE = 'https://api.openalex.org/works'


def reconstruct_abstract(inv):
    # inv: dict mapping token -> list of positions
    # simplest reconstruction: build a list sized by max pos and fill tokens
    if not inv:
        return ''
    # find max position
    maxpos = 0
    for token, poses in inv.items():
        for p in poses:
            if p > maxpos:
                maxpos = p
    arr = [''] * (maxpos + 1)
    for token, poses in inv.items():
        for p in poses:
            if 0 <= p <= maxpos:
                arr[p] = token
    # join and return
    txt = ' '.join([t for t in arr if t])
    # postprocess: collapse multiple spaces
    return ' '.join(txt.split())


def fetch(query, out_path: Path, retmax=200, per_page=50, sleep=0.2):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    cursor = '*'
    params = {
        'filter': None,
        'per_page': per_page,
        'cursor': cursor,
        'search': query,
    }

    session = requests.Session()
    url = BASE

    while written < retmax:
        params['per_page'] = min(per_page, retmax - written)
        # OpenAlex uses cursor param as '?cursor=...' and 'search' for fulltext-like search
        resp = session.get(url, params={'search': query, 'per_page': params['per_page'], 'cursor': cursor}, timeout=30)
        if resp.status_code != 200:
            print(f"[openalex] error status {resp.status_code}: {resp.text[:200]}")
            break
        j = resp.json()
        results = j.get('results', [])
        if not results:
            print('[openalex] no more results')
            break

        for item in results:
            # Only keep records that have abstract_inverted_index
            aii = item.get('abstract_inverted_index')
            if not aii:
                continue
            abstract = reconstruct_abstract(aii)
            if not abstract:
                continue
            rec = {
                'id': item.get('id'),
                'title': item.get('title'),
                'text': abstract,
                'doi': item.get('doi'),
                'url': item.get('id'),
                'source': 'openalex',
            }
            out_path.open('a', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False) + '\n')
            written += 1
            if written >= retmax:
                break

        # update cursor
        cursor = j.get('meta', {}).get('next_cursor')
        if not cursor:
            break
        time.sleep(sleep)

    print(f"[openalex] wrote {written} records to {out_path}")
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', '-q', required=True)
    p.add_argument('--retmax', type=int, default=200)
    p.add_argument('--per_page', type=int, default=50)
    p.add_argument('--out', default='data/raw/openalex.jsonl')
    args = p.parse_args()

    fetch(args.query, Path(args.out), retmax=args.retmax, per_page=args.per_page)


if __name__ == '__main__':
    main()
