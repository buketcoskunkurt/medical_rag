#!/usr/bin/env python3
# Batch test English queries against local API and write CSV metrics.

from __future__ import annotations
import csv
import os
import sys
import time
import argparse
import requests
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def run_query(session: requests.Session, query: str, k: int = 5, timeout: float = 60.0):
    t0 = time.time()
    r = session.post(f"{API_URL}/qa", json={"question": query, "k": k}, timeout=timeout)
    t1 = time.time()
    r.raise_for_status()
    data = r.json()
    # API returns seconds; convert to ms per request
    rt_ms = int(round(float(data.get("retrieval_time_seconds", 0.0)) * 1000))
    gt_ms = int(round(float(data.get("generation_time_seconds", 0.0)) * 1000))
    tt_ms = int(round(float(data.get("total_time_seconds", 0.0)) * 1000))
    # Safety fallback: if any are missing, use wall time
    if not (rt_ms and gt_ms and tt_ms):
        elapsed_ms = int(round((t1 - t0) * 1000))
        tt_ms = tt_ms or elapsed_ms
    return {
        "Retrieval_Time_MS": rt_ms,
        "Generation_Time_MS": gt_ms,
        "Total_Time_MS": tt_ms,
        "english_text": (data.get("english", {}) or {}).get("text", ""),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(Path("data")/"queries_100_en.csv"))
    p.add_argument("--output", default=str(Path("data")/"test"/"results_en.csv"))
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    with inp.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Query" not in reader.fieldnames:
            print("CSV must have a 'Query' column", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    session = requests.Session()

    with outp.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["Query", "Query_Length", "Retrieval_Time_MS", "Generation_Time_MS", "Total_Time_MS"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            q = (row.get("Query") or "").strip()
            ql = row.get("Query_Length") or ""
            if not q:
                continue
            try:
                res = run_query(session, q, k=args.k)
            except Exception as e:
                # On error, write zeros but keep the query for traceability
                res = {"Retrieval_Time_MS": 0, "Generation_Time_MS": 0, "Total_Time_MS": 0, "english_text": str(e)}
            writer.writerow({
                "Query": q,
                "Query_Length": ql,
                "Retrieval_Time_MS": res["Retrieval_Time_MS"],
                "Generation_Time_MS": res["Generation_Time_MS"],
                "Total_Time_MS": res["Total_Time_MS"],
            })
            if i % 10 == 0:
                print(f"Processed {i}/{len(rows)} queries...")

    print(f"Wrote: {outp}")


if __name__ == "__main__":
    main()
