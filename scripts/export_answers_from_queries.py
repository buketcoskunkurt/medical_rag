#!/usr/bin/env python3
"""
Export model answers for a list of queries to a CSV for manual evaluation.

Input CSV: must contain a 'Query' column (e.g., data/test/10_Q_A_with_Reference.csv)
Output CSV: writes 'Query','Answer','Retrieval_Time_MS','Generation_Time_MS','Total_Time_MS'

Notes:
- To avoid changing model behavior or retesting, we normalize the "insufficient" token for English outputs here:
    if the pipeline returns the Turkish token "yetersiz" in the English field, we map it to "insufficient".

It first tries the running API at API_URL/qa, else falls back to calling app.main.qa directly.
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Dict

import requests
from requests.exceptions import RequestException

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def call_api_or_local(question: str, k: int = 5, prefer_local: bool = False) -> Dict:
    """Call /qa if API available, else import app.main and call qa() directly.
    Returns a dict like the API schema with english.text and timing fields.
    """
    # Try API unless prefer_local
    if not prefer_local:
        try:
            r0 = time.time()
            resp = requests.post(f"{API_URL}/qa", json={"question": question, "k": k}, timeout=8)
            r1 = time.time()
            resp.raise_for_status()
            data = resp.json()
            # ensure total time present
            data.setdefault("total_time_seconds", (r1 - r0))
            return data
        except (RequestException, ValueError, Exception):
            pass

    # Fallback to local import
    try:
        import importlib, sys
        proj_root = Path(__file__).resolve().parent.parent
        if str(proj_root) not in sys.path:
            sys.path.insert(0, str(proj_root))
        rag = importlib.import_module('app.main')
        req = rag.RetrieveRequest(question=question, k=k)
        r0 = time.time()
        res = rag.qa(req)
        r1 = time.time()
        data = res.model_dump() if hasattr(res, 'model_dump') else res.dict()
        data.setdefault('total_time_seconds', (r1 - r0))
        return data
    except Exception:
        return {"english": {"text": ""}, "retrieval_time_seconds": 0.0, "generation_time_seconds": 0.0, "total_time_seconds": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to CSV with a 'Query' column")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--k", type=int, default=5, help="Top-k snippets for retrieval")
    ap.add_argument("--local", action="store_true", help="Skip API and call local app.main.qa directly for speed")
    args = ap.parse_args()

    qpath = Path(args.queries)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with qpath.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, str]] = list(reader)

    fieldnames = ["Query", "Answer", "Retrieval_Time_MS", "Generation_Time_MS", "Total_Time_MS"]
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows, 1):
            q = (row.get("Query") or "").strip()
            if not q:
                continue
            data = call_api_or_local(q, k=args.k, prefer_local=args.local)
            ans = ((data.get("english", {}) or {}).get("text") or "").strip()
            # Normalize the English insufficient token without changing model behavior
            norm = ''.join(ch for ch in ans.lower() if 'a' <= ch <= 'z')
            if norm == 'yetersiz':
                ans = 'insufficient'
            rt_ms = int(round(float(data.get("retrieval_time_seconds", 0.0)) * 1000))
            gt_ms = int(round(float(data.get("generation_time_seconds", 0.0)) * 1000))
            tt_ms = int(round(float(data.get("total_time_seconds", 0.0)) * 1000))
            w.writerow({
                "Query": q,
                "Answer": ans,
                "Retrieval_Time_MS": rt_ms,
                "Generation_Time_MS": gt_ms,
                "Total_Time_MS": tt_ms,
            })
            if i % 10 == 0:
                print(f"Exported {i}/{len(rows)}")

    print(f"Wrote answers to: {outp}")


if __name__ == "__main__":
    main()
