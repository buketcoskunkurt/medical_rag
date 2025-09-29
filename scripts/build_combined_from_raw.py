#!/usr/bin/env python3
"""
Build data/combined.jsonl from all files under data/raw/ with simple dedup.

Dedup strategies:
- url (default): remove duplicates by exact URL match
- url+hash: also remove duplicates by exact text hash (normalized)

Usage (Windows PowerShell):
    conda run -n rag-med python scripts\build_combined_from_raw.py \
        --raw-dir data\raw \
        --out data\combined.jsonl \
        --dedup-mode url
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import List, Iterable
import hashlib

def _iter_files(raw_dir: Path) -> Iterable[Path]:
    if raw_dir.is_file():
        yield raw_dir
        return
    for ext in ("*.jsonl", "*.ndjson", "*.json"):
        yield from raw_dir.rglob(ext)

def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _read_json(p: Path) -> List[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return []

def _pick_text(r: dict) -> str:
    for k in ("text", "content", "body", "abstract", "summary"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def _normalize_text(s: str) -> str:
    import re
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _hash_text(s: str) -> str:
    return hashlib.sha1(_normalize_text(s).lower().encode("utf-8")).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/combined.jsonl")
    p.add_argument("--dedup-mode", choices=["url", "url+hash"], default="url",
                   help="Select dedup strategy. 'url' (fast) or 'url+hash' (also exact text duplicates)")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url_set = set()
    hash_set = set()

    accepted: List[dict] = []
    use_hash = ("hash" in args.dedup_mode)

    for fp in _iter_files(raw_dir):
        rows: List[dict] = []
        if fp.suffix.lower() == ".jsonl" or fp.suffix.lower() == ".ndjson":
            rows = _read_jsonl(fp)
        elif fp.suffix.lower() == ".json":
            rows = _read_json(fp)
        else:
            continue
        # per-file temporary sets to avoid intra-batch dup work
        batch_seen_urls = set()
        batch_seen_hash = set()
        for r in rows:
            t = _pick_text(r)
            if not t or len(_normalize_text(t)) < 20:
                continue
            u = r.get("url") if isinstance(r.get("url"), str) else None
            if u and (u in url_set or u in batch_seen_urls):
                continue
            h = _hash_text(t) if use_hash else None
            if use_hash and (h in hash_set or h in batch_seen_hash):
                continue
            # accept immediately (simple dedup only)
            accepted.append(r)
            if u:
                url_set.add(u)
                batch_seen_urls.add(u)
            if use_hash and h is not None:
                hash_set.add(h)
                batch_seen_hash.add(h)

    # write combined.jsonl
    with out_path.open("w", encoding="utf-8") as fh:
        for r in accepted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "raw_files": len(list(_iter_files(raw_dir))),
        "accepted": len(accepted),
        "out": str(out_path)
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
