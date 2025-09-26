#!/usr/bin/env python3
# Canonical generator entrypoint: minimal RAG QA generator (one-sentence)

from pathlib import Path
import argparse
import json
import time
import re
from typing import List

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError('Missing retrieval dependencies: ' + str(e))

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    raise RuntimeError('Missing transformers dependency: ' + str(e))


def load_meta(meta_path: Path):
    if meta_path.suffix.lower() == '.parquet':
        try:
            import pandas as pd
            df = pd.read_parquet(meta_path)
            recs = df.to_dict(orient='records')
            return {i: r for i, r in enumerate(recs)}
        except Exception:
            alt = meta_path.with_suffix('.jsonl')
            if alt.exists():
                meta = {}
                with alt.open('r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        meta[i] = d
                return meta
            raise
    meta = {}
    with meta_path.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            try:
                d = json.loads(line)
            except Exception:
                continue
            meta[i] = d
    return meta


def retrieve_topk(query: str, emb_model: SentenceTransformer, index_path: Path, meta: dict, k: int = 5, cand_k: int = 10, diverse_per_source: int = 0) -> List[dict]:
    if k <= 0 or cand_k <= 0:
        return []
    q_emb = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.read_index(str(index_path))
    D, I = index.search(np.array(q_emb, dtype='float32'), max(cand_k, k))
    ids = [int(x) for x in I[0].tolist() if x >= 0]
    dists = [float(x) for x in D[0].tolist()][:len(ids)]

    candidates = []
    for pid, dist in zip(ids, dists):
        rec = meta.get(pid)
        if not rec:
            continue
        candidates.append({
            'pid': pid,
            'title': rec.get('title') or '',
            'text': rec.get('text') or '',
            'url': rec.get('url'),
            'source': rec.get('source'),
            'score': float(-dist)
        })

    if diverse_per_source and diverse_per_source > 0:
        from collections import defaultdict, deque
        by_src = defaultdict(list)
        for c in candidates:
            by_src[(c.get('source') or 'unknown')].append(c)
        for src in list(by_src.keys()):
            by_src[src] = deque(by_src[src][:diverse_per_source])
        merged = []
        sources = list(by_src.keys())
        si = 0
        while len(merged) < k and any(by_src.values()):
            src = sources[si % len(sources)]
            if by_src[src]:
                merged.append(by_src[src].popleft())
            si += 1
        return merged[:k]

    return candidates[:k]


def build_prompt(query: str, snippets: List[dict]) -> str:
    parts = []
    parts.append('Answer the user query using ONLY the provided snippets. If you cannot answer from the snippets, reply with exactly the single word: yetersiz')
    parts.append('\nUser query: ' + query)
    parts.append('Key terms from the question to stay on-topic: ' + query)
    parts.append('\nSnippets:')
    for i, s in enumerate(snippets[:3], start=1):
        title = (s.get('title') or '').strip()
        text = (s.get('text') or '').strip()
        if len(text) > 400:
            text = text[:400].rsplit(' ', 1)[0] + '...'
        parts.append(f"[{i}] {title}: {text}")
    parts.append('\nWrite exactly one concise factual sentence that directly answers the user query using only the snippets.')
    parts.append('Use key terms from the question (e.g., problem/condition names) so the sentence stays on-topic. Paraphrase; do not copy snippets verbatim.')
    parts.append('If the snippets do not contain enough information, reply exactly: yetersiz')
    parts.append('Do NOT add headings, labels, sections, lists, citations, study mentions, or extra commentary; output only the single sentence or "yetersiz".')
    return '\n'.join(parts)


def generate_answer(prompt: str, model, tok, device, max_new_tokens: int = 256, temperature: float = 0.25) -> str:
    import torch
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    is_seq2seq = getattr(model.config, 'model_type', '').lower().startswith('t5') or 't5' in getattr(model.config, 'model_type', '')
    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature)
    if is_seq2seq:
        if temperature == 0:
            gen_kwargs.update(dict(do_sample=False, num_beams=4, early_stopping=True, min_length=10))
        else:
            gen_kwargs.update(dict(do_sample=True, top_p=0.9, top_k=40, min_length=10))
    else:
        gen_kwargs.update(dict(do_sample=True))

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)
    try:
        inp_dec = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if text.startswith(inp_dec):
            text = text[len(inp_dec):].strip()
    except Exception:
        pass
    return text.strip()


def enforce_single_sentence(text: str, snippets: List[dict]) -> str:
    t = (text or '').strip()
    if not t:
        return 'yetersiz'
    if t.lower() == 'yetersiz':
        return 'yetersiz'
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r"^[\"'`\s]*[A-Za-z0-9 \-()]{1,100}[:\-–—]\s*", "", t)
    m = re.search(r'(.+?[\.!?])\s', t + ' ')
    if m:
        s = m.group(1).strip()
    else:
        s = t
    words = s.split()
    if len(words) > 10:
        mid = len(words) // 2
        first_half = ' '.join(words[:mid])
        second_half = ' '.join(words[mid:])
        fh = re.sub(r'[^a-z0-9]', '', first_half.lower())
        sh = re.sub(r'[^a-z0-9]', '', second_half.lower())
        if fh and sh and (fh == sh or fh in sh or sh in fh):
            s = first_half.rstrip(' ,;:') + '.'
    s = s.strip()
    if not s:
        for sn in snippets:
            for sent in (sn.get('text') or '').split('. '):
                st = sent.strip()
                if len(st) > 30:
                    return (st.rstrip('.') + '.')
        return 'yetersiz'
    if len(s) > 200:
        s = s[:200].rsplit(' ', 1)[0].rstrip(' ,;:') + '...'
    if s and s[-1] not in '.!?':
        s = s + '.'
    return s


def sanitize_paragraph(par: str) -> str:
    if not par:
        return par
    par = re.sub(r'\[\d+\]', '', par)
    par = re.sub(r'\bSNIPPET\b', '', par, flags=re.I)
    par = re.sub(r'\bSNIPET\b', '', par, flags=re.I)
    par = re.sub(r'^(Abstract|Özet|Summary|Conclusion|Başlık)[:\-–—]\s*', '', par, flags=re.I)
    par = re.sub(r'([\.!?]){2,}', r'\1', par)
    par = re.sub(r'\s{2,}', ' ', par).strip()
    par = re.sub(r'\bno lists, no citations, no extra sections\.\?\b', '', par, flags=re.I).strip()
    par = re.sub(r'\bdo not add( any)? headings[^\.]*\.?', '', par, flags=re.I).strip()
    par = re.sub(r'\bstart directly with[^\.]*\.?', '', par, flags=re.I).strip()
    if par:
        par = par[0].upper() + par[1:]
    return par


def main():
    import os
    from sentence_transformers import models

    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--cand', type=int, default=10)
    p.add_argument('--index', default='data/index.faiss')
    p.add_argument('--meta', default='data/meta.parquet')
    p.add_argument('--out', default=None)
    p.add_argument('--model', default='google/flan-t5-small')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--lang', choices=['en', 'tr'], default='en')
    p.add_argument('--diverse-per-source', type=int, default=0)
    p.add_argument('--emb-model', default=os.getenv('EMB_MODEL', 'dmis-lab/biobert-base-cased-v1.1'))
    args = p.parse_args()

    def translate(text: str, _unused_model_name: str, target_lang: str | None = None) -> str:
        if not text:
            return text
        try:
            from argostranslate import translate as atrans  # type: ignore
            if target_lang in {"tr", "en"}:
                src_code = "en" if target_lang == "tr" else "tr"
                res = atrans.translate(text, src_code, target_lang)
                if isinstance(res, str) and res.strip() and res.strip() != text.strip():
                    return res.strip()
            else:
                def looks_tr(s: str) -> bool:
                    return any(ch in s for ch in 'çğıöşüÇĞİÖŞÜ')
                if looks_tr(text):
                    res = atrans.translate(text, 'tr', 'en')
                else:
                    res = atrans.translate(text, 'en', 'tr')
                if isinstance(res, str) and res.strip() and res.strip() != text.strip():
                    return res.strip()
        except Exception:
            pass
        return text

    query_tr = args.query
    if args.lang == 'tr':
        user_query_en = translate(query_tr, '', target_lang='en') or query_tr
    else:
        user_query_en = args.query

    meta = load_meta(Path(args.meta))
    _hf = models.Transformer(args.emb_model)
    _pool = models.Pooling(_hf.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    emb_model = SentenceTransformer(modules=[_hf, _pool])

    t0 = time.time()
    snippets = retrieve_topk(user_query_en, emb_model, Path(args.index), meta, k=args.topk, cand_k=args.cand, diverse_per_source=args.diverse_per_source)
    t1 = time.time()

    prompt = build_prompt(user_query_en, snippets)
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    tgen0 = time.time()
    raw = generate_answer(prompt, mdl, tok, device='cpu', max_new_tokens=args.max_tokens, temperature=args.temperature)
    eng = enforce_single_sentence(sanitize_paragraph(raw), snippets)
    if eng.strip().lower() == 'yetersiz':
        eng = 'insufficient'
    tgen1 = time.time()

    tr_text = translate(eng, '', target_lang='tr')
    if eng.strip().lower() == 'insufficient':
        tr_text = 'yetersiz'

    def normalize_tr(s: str) -> str:
        if not s:
            return s
        rep = {
            'Inzonia': 'Uykusuzluk',
            'İnzonia': 'Uykusuzluk',
            'Insomnia': 'Uykusuzluk',
            'insomnia': 'Uykusuzluk',
        }
        for k, v in rep.items():
            s = s.replace(k, v)
        return s

    tr_text = normalize_tr(tr_text)

    result = {
        'query_tr': query_tr if args.lang == 'tr' else '',
        'query_en': user_query_en,
        'retrieval_time_seconds': round(t1 - t0, 4),
        'generation_time_seconds': round(tgen1 - tgen0, 4),
        'total_time_seconds': round((t1 - t0) + (tgen1 - tgen0), 4),
        'english': {'text': eng},
        'turkish': {'text': tr_text},
        'used_snippets': [
            {
                'id': str(s.get('pid')),
                'title': s.get('title'),
                'url': s.get('url'),
                'source': s.get('source'),
                'text': s.get('text'),
                'score': float(s.get('score', 0.0)),
            } for s in snippets
        ],
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
