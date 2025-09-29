#!/usr/bin/env python3
# Compute only Perplexity per query and a summary (mean, median, p90, min, max)
from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
import statistics
from typing import List
import requests

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

API_URL = os.environ.get("API_URL", "http://localhost:8000")

class PPL:
    def __init__(self, model_id: str = "gpt2"):
        self.model_id = model_id
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        if AutoModelForCausalLM is None:
            self.model = None
            self.tok = None
        else:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

    def score(self, text: str) -> float:
        if not self.model or not self.tok or not text:
            return float("nan")
        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, labels=enc["input_ids"])  # type: ignore
            loss = float(out.loss.detach().cpu().item())
        return float(math.exp(loss))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(v for v in values if not math.isnan(v))
    if not vs:
        return float("nan")
    k = (len(vs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vs[int(k)]
    d0 = vs[f] * (c - k)
    d1 = vs[c] * (k - f)
    return d0 + d1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(Path("data")/"test"/"queries_100_en.csv"))
    ap.add_argument("--out", default=str(Path("data")/"test"/"perplexity_en.csv"))
    ap.add_argument("--summary", default=str(Path("data")/"test"/"perplexity_summary_en.csv"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    qpath = Path(args.queries)
    outp = Path(args.out)
    spath = Path(args.summary)
    outp.parent.mkdir(parents=True, exist_ok=True)

    ppl = PPL()
    sess = requests.Session()

    with qpath.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    per_row = []
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Query","Query_Length","Perplexity"])
        w.writeheader()
        for i, row in enumerate(rows, 1):
            q = (row.get("Query") or "").strip()
            ql = (row.get("Query_Length") or "").strip()
            if not q:
                continue
            try:
                r = sess.post(f"{API_URL}/qa", json={"question": q, "k": args.k}, timeout=60)
                r.raise_for_status()
                data = r.json()
                text = ((data.get("english",{}) or {}).get("text") or "").strip()
            except Exception:
                text = ""
            pv = ppl.score(text)
            per_row.append(pv)
            w.writerow({"Query": q, "Query_Length": ql, "Perplexity": pv})
            if i % 10 == 0:
                print(f"PPL {i}/{len(rows)}")

    mean = statistics.fmean([v for v in per_row if not math.isnan(v)]) if per_row else float("nan")
    med = statistics.median([v for v in per_row if not math.isnan(v)]) if per_row else float("nan")
    p90 = percentile(per_row, 0.9)
    mn = min([v for v in per_row if not math.isnan(v)], default=float("nan"))
    mx = max([v for v in per_row if not math.isnan(v)], default=float("nan"))

    with spath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Count","Mean","Median","P90","Min","Max"])
        w.writeheader()
        w.writerow({
            "Count": len([v for v in per_row if not math.isnan(v)]),
            "Mean": mean,
            "Median": med,
            "P90": p90,
            "Min": mn,
            "Max": mx,
        })

    print(f"Wrote: {outp}\nWrote summary: {spath}")


if __name__ == "__main__":
    main()
