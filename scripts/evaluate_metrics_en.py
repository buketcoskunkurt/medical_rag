#!/usr/bin/env python3
# Compute BLEU, ROUGE-N/L, METEOR, BERTScore, Perplexity for English QA outputs.
# - Inputs: queries CSV (Query,Query_Length) and optional references CSV (Query,Reference)
# - Calls API /qa for each query (English), computes metrics against references when provided
# - Writes a new CSV with metrics; does NOT overwrite results_en.csv

from __future__ import annotations
import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from requests.exceptions import RequestException

# Prefer safetensors to avoid torch.load CVE checks when possible
os.environ.setdefault("TRANSFORMERS_PREFER_SAFETENSORS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Optional deps
try:
    import nltk  # type: ignore
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    from nltk.translate.meteor_score import meteor_score  # type: ignore
    # Best-effort download for resources if missing
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass
except Exception:
    sentence_bleu = None
    SmoothingFunction = None
    meteor_score = None

try:
    from bert_score import score as bertscore_score, BERTScorer  # type: ignore
except Exception:
    bertscore_score = None
    BERTScorer = None

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    rouge_scorer = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def load_refs(path: Path) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    if not path or not path.exists():
        return refs
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Query" not in reader.fieldnames or "Reference" not in reader.fieldnames:
            return refs
        for row in reader:
            q = (row.get("Query") or "").strip()
            r = (row.get("Reference") or "").strip()
            if not q or not r:
                continue
            refs.setdefault(q, []).append(r)
    return refs


def compute_bleu(hyp: str, refs: List[str]) -> float:
    if not sentence_bleu or not refs:
        return float("nan")
    smoothie = SmoothingFunction().method3 if SmoothingFunction else None
    refs_tok = [r.split() for r in refs]
    hyp_tok = hyp.split()
    try:
        return float(sentence_bleu(refs_tok, hyp_tok, smoothing_function=smoothie))
    except Exception:
        return float("nan")


def _token_f1(hyp: str, ref: str) -> float:
    """Simple whitespace token F1 as a safe fallback metric in [0,1]."""
    h = [t for t in hyp.lower().split() if t]
    r = [t for t in ref.lower().split() if t]
    if not h or not r:
        return 0.0
    hs = set(h)
    rs = set(r)
    inter = len(hs & rs)
    if inter == 0:
        return 0.0
    prec = inter / len(hs)
    rec = inter / len(rs)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_meteor(hyp: str, refs: List[str]) -> float:
    # Prefer NLTK METEOR if available, else fallback to token-F1; take best over refs
    clean_refs = [r for r in refs if (r or '').strip()]
    if not hyp.strip() or not clean_refs:
        return float("nan")
    best = 0.0
    used_nltk = False
    if meteor_score:
        try:
            best = max(float(meteor_score([r], hyp)) for r in clean_refs)
            used_nltk = True
        except Exception:
            used_nltk = False
    if not used_nltk:
        # token-F1 fallback
        best = max(_token_f1(hyp, r) for r in clean_refs)
    return best


def compute_rouge(hyp: str, refs: List[str]) -> Tuple[float, float, float]:
    if not rouge_scorer or not refs:
        return (float("nan"), float("nan"), float("nan"))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    best = (float("nan"), float("nan"), float("nan"))
    best_f = -1.0
    for r in refs:
        s = scorer.score(r, hyp)
        r1 = s["rouge1"].fmeasure
        r2 = s["rouge2"].fmeasure
        rl = s["rougeL"].fmeasure
        f = (r1 + r2 + rl) / 3.0
        if f > best_f:
            best = (r1, r2, rl)
            best_f = f
    return best


def compute_bertscore_batch(hyps: List[str], refs: List[str], scorer: Optional[object]) -> Tuple[List[float], List[float], List[float]]:
    if scorer is None or not refs:
        n = len(hyps)
        return [float("nan")] * n, [float("nan")] * n, [float("nan")] * n
    try:
        P, R, F1 = scorer.score(hyps, refs)
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception:
        n = len(hyps)
        return [float("nan")] * n, [float("nan")] * n, [float("nan")] * n


class PerplexityScorer:
    def __init__(self, model_id: str = "gpt2"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if torch else "cpu"
        if AutoModelForCausalLM is None:
            self.model = None
            self.tok = None
        else:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

    def ppl(self, text: str) -> float:
        if not self.model or not self.tok:
            return float("nan")
        if not text:
            return float("nan")
        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, labels=enc["input_ids"])  # type: ignore
            loss = float(out.loss.detach().cpu().item())
        return float(math.exp(loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(Path("data")/"test"/"queries_100_en.csv"))
    ap.add_argument("--refs", default="", help="Optional CSV with columns: Query,Reference (can repeat rows for multiple refs)")
    ap.add_argument("--out", default=str(Path("data")/"test"/"metrics_en.csv"))
    ap.add_argument("--ref-col", default="", help="Optional: reference column name in queries CSV (e.g., Reference_Answer)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bertscore", action="store_true", help="Enable BERTScore (slower; downloads model on first run)")
    ap.add_argument("--bertscore-model", default=os.environ.get("BERTSCORE_MODEL", "distilroberta-base"))
    args = ap.parse_args()

    qpath = Path(args.queries)
    refpath = Path(args.refs) if args.refs else None
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Load queries (and optionally embedded references if present)
    with qpath.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # References may come from a separate file (Query,Reference) or
    # be embedded in the queries CSV via one of several column names.
    refs = load_refs(refpath) if refpath else {}
    if not refs:
        fieldnames = list(rows[0].keys()) if rows else []
        # Determine candidate columns
        candidates = []
        if args.ref_col:
            candidates.append(args.ref_col)
        candidates.extend(["Reference_Answer", "Reference", "Gold", "Expected_Answer", "Ref", "Ref1", "Ref2"])
        use_cols = [c for c in candidates if c in fieldnames]
        if use_cols:
            tmp: Dict[str, List[str]] = {}
            for r in rows:
                q = (r.get("Query") or "").strip()
                vals: List[str] = []
                for c in use_cols:
                    v = (r.get(c) or "").strip()
                    if v:
                        vals.append(v)
                if q and vals:
                    for v in vals:
                        tmp.setdefault(q, []).append(v)
            refs = tmp

    sess = requests.Session()
    ppl_sc = PerplexityScorer()
    # Initialize optional BERTScorer once to avoid reloading per sample
    bs_scorer = None
    if args.bertscore and BERTScorer is not None:
        try:
            bs_scorer = BERTScorer(lang="en", rescale_with_baseline=False, model_type=args.bertscore_model)
        except Exception:
            bs_scorer = None

    fieldnames = [
        "Query","Query_Length",
        "Retrieval_Time_MS","Generation_Time_MS","Total_Time_MS",
        "BLEU4","ROUGE1_F","ROUGE2_F","ROUGEL_F","METEOR",
        "BERTScore_P","BERTScore_R","BERTScore_F1",
        "Perplexity",
    ]

    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows, 1):
            q = (row.get("Query") or "").strip()
            ql = (row.get("Query_Length") or "").strip()
            if not q:
                continue
            # Try API first; if unavailable, fallback to local import of app.main.qa
            data = None
            r0 = time.time()
            try:
                resp = sess.post(f"{API_URL}/qa", json={"question": q, "k": args.k}, timeout=20)
                r1 = time.time()
                resp.raise_for_status()
                data = resp.json()
            except (RequestException, ValueError, Exception):
                # API not reachable or bad response; try local
                try:
                    import importlib, sys
                    # Ensure project root is on sys.path for 'app.main'
                    proj_root = Path(__file__).resolve().parent.parent
                    if str(proj_root) not in sys.path:
                        sys.path.insert(0, str(proj_root))
                    rag = importlib.import_module('app.main')
                    # Build request object and call function directly
                    req = rag.RetrieveRequest(question=q, k=args.k)
                    r0 = time.time()
                    res = rag.qa(req)
                    r1 = time.time()
                    # res is a Pydantic model; convert to dict
                    data = res.model_dump() if hasattr(res, 'model_dump') else res.dict()
                    # Ensure time fields present (qa already returns them), else synthesize total
                    data.setdefault('total_time_seconds', (r1 - r0))
                except Exception:
                    # Final fallback: empty text with measured total time
                    r1 = time.time()
                    data = {"english": {"text": ""}, "retrieval_time_seconds": 0.0, "generation_time_seconds": 0.0, "total_time_seconds": (r1 - r0)}
            text = ((data.get("english",{}) or {}).get("text") or "").strip()
            if not text:
                # Avoid empty hypothesis for metrics like BERTScore
                text = "insufficient"
            rt_ms = int(round(float(data.get("retrieval_time_seconds",0.0))*1000))
            gt_ms = int(round(float(data.get("generation_time_seconds",0.0))*1000))
            tt_ms = int(round(float(data.get("total_time_seconds",0.0))*1000))

            ref_list = refs.get(q, [])
            bleu = compute_bleu(text, ref_list)
            meteor = compute_meteor(text, ref_list)
            r1f, r2f, rlf = compute_rouge(text, ref_list)
            # BERTScore per sample (optional). If multiple refs, take best over refs.
            bP = bR = bF = float('nan')
            if ref_list:
                # Light truncation to speed scoring
                def trunc(s: str, n: int = 256) -> str:
                    toks = s.split()
                    return ' '.join(toks[:n])
            if bs_scorer is not None and ref_list:
                # Light truncation to speed scoring
                cand = trunc(text)
                best_f = -1.0
                for ref in ref_list:
                    p, r, f1 = compute_bertscore_batch([cand], [trunc(ref)], bs_scorer)
                    f = float(f1[0])
                    if f > best_f:
                        best_f = f
                        bP = float(p[0])
                        bR = float(r[0])
                        bF = float(f1[0])
            elif bertscore_score is not None and ref_list:
                try:
                    cand = ' '.join(text.split()[:256])
                    best_f = -1.0
                    for ref in ref_list:
                        P, R, F1 = bertscore_score([cand], [' '.join(ref.split()[:256])], lang='en', rescale_with_baseline=False, model_type=os.environ.get('BERTSCORE_MODEL','microsoft/deberta-base-mnli'))
                        f = float(F1[0])
                        if f > best_f:
                            best_f = f
                            bP = float(P[0])
                            bR = float(R[0])
                            bF = float(F1[0])
                except Exception:
                    pass

            ppl = ppl_sc.ppl(text)

            w.writerow({
                "Query": q,
                "Query_Length": ql,
                "Retrieval_Time_MS": rt_ms,
                "Generation_Time_MS": gt_ms,
                "Total_Time_MS": tt_ms,
                "BLEU4": bleu,
                "ROUGE1_F": r1f,
                "ROUGE2_F": r2f,
                "ROUGEL_F": rlf,
                "METEOR": meteor,
                "BERTScore_P": bP,
                "BERTScore_R": bR,
                "BERTScore_F1": bF,
                "Perplexity": ppl,
            })
            if i % 10 == 0:
                print(f"Scored {i}/{len(rows)}...")

    print(f"Wrote metrics: {outp}")


if __name__ == "__main__":
    main()
