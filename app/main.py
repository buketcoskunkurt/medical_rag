import os
import time
import re
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers import CrossEncoder

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index.faiss")      # FAISS index
META_PATH = os.path.join(BASE_DIR, "data", "meta.parquet")      # Parquet meta

"""
Embedding model must match the FAISS index dimension.
Default to local BioBERT safetensors to avoid torch.load restrictions.
Override with EMB_MODEL env var if needed.
"""
EMB_MODEL = os.getenv("EMB_MODEL", os.path.join(BASE_DIR, "models", "biobert-base-cased-v1.1-sf"))
_hf = models.Transformer(EMB_MODEL) 
_pool = models.Pooling(_hf.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
_embed_model = SentenceTransformer(modules=[_hf, _pool])
EMBED_DIM = _hf.get_word_embedding_dimension()
 
# generator model (default per Option A): Flan-T5 base
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google/flan-t5-base")

#cross-encoder model for re-ranking
MAX_K = 20   # Kullanıcıya dönecek max sonuç/aday
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.6")) # CE vs FAISS ağırlığı (α)
TOPN_MULT = int(os.getenv("TOPN_MULT", "20")) # FAISS’ten geniş aday havuzu çekmek için çarpan
TOPN_CAP = int(os.getenv("TOPN_CAP", "500")) # CE’yi yormamak için üst sınır
EVIDENCE_CE_MIN = float(os.getenv("EVIDENCE_CE_MIN", "0.2")) # CE normalleştirilmiş skor alt eşiği (kanıt filtresi)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index bulunamadı: {INDEX_PATH}")

index = faiss.read_index(INDEX_PATH)

if index.d != EMBED_DIM:
    # FAISS index dimension ile EMBED_DIM uyuşmuyorsa erken uyarı
    raise ValueError(
        (
            f"FAISS index dim({index.d}) != EMBED_DIM({EMBED_DIM}). "
            f"API EMB_MODEL='{EMB_MODEL}'. Lütfen API embedder ile indeksi aynı modelle hizalayın."
        )
    )

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Meta dosyası bulunamadı: {META_PATH}")

# Parquet -> list[dict]
meta_df = pd.read_parquet(META_PATH)
meta_data = meta_df.to_dict(orient="records")

# Komşu chunk haritası (Bozuk/kesilmiş snippet’leri önceki chunk’ın kuyruğuyla tamamla)
_neighbors_map = {}
for m in meta_data:
    mid = str(m.get("id", ""))
    if "_" in mid:
        prefix = mid.rpartition("_")[0]
        try:
            idx = int(mid.split("_")[-1])
        except Exception:
            continue
        _neighbors_map.setdefault(prefix, []).append((idx, m.get("text", "")))
for k in _neighbors_map:
    _neighbors_map[k].sort()


# API Şeması
app = FastAPI(title="Medical RAG API", version="1.0")

class RetrieveRequest(BaseModel):
    question: str
    k: int = 5

class RetrieveResult(BaseModel):
    id: str
    title: str        # Turkish title (translated)
    title_en: str     # Original English title
    url: str
    score: float
    snippet: str      # Turkish snippet (translated)
    snippet_en: str   # Original English snippet

class RetrieveResponse(BaseModel):
    query_tr: str
    query_en: str
    retrieval_ms: int
    results: List[RetrieveResult]

class QAResponse(BaseModel):
    query_tr: str
    query_en: str
    retrieval_time_seconds: float
    generation_time_seconds: float
    total_time_seconds: float
    english: dict
    turkish: dict
    used_snippets: list

#Çeviri fonksiyonları
def to_tr(text: str, source_hint: str = "en") -> str:
    # Prefer Argos offline translation if available; fallback: return text
    try:
        from argostranslate import translate as atrans  # type: ignore
        src = "en" if source_hint.lower().startswith("en") else "tr"
        tgt = "tr"
        out = atrans.translate(text or "", src, tgt)
        return (out or text or "").strip()
    except Exception:
        return text

def to_en(text: str, source_hint: str = "tr") -> str:
    # Heuristic: if it already looks like English, return as-is
    t = (text or "").strip()
    if not t:
        return t
    has_tr_chars = any(ch in t for ch in "çğıöşüÇĞİÖŞÜ")
    looks_english = (not has_tr_chars) and bool(re.search(r"\b(the|and|of|to|in|is|are|what|how|when|which)\b", t, flags=re.I))
    if looks_english:
        return t
    try:
        from argostranslate import translate as atrans  # type: ignore
        src = "tr" if source_hint.lower().startswith("tr") else "en"
        tgt = "en"
        out = atrans.translate(t, src, tgt)
        return (out or t).strip()
    except Exception:
        return t

# =========================== Generator (Flan-T5) ============================
# Single source of truth for model id
GEN_MODEL_ID = GENERATOR_MODEL
_gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
_gen_mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)
try:
    import torch
    # Force CUDA for generation when available
    if torch.cuda.is_available():
        _GEN_DEVICE = "cuda"
    else:
        _GEN_DEVICE = "cpu"
    _gen_mdl.to(_GEN_DEVICE)
except Exception:
    _GEN_DEVICE = "cpu"

#Prompt oluşturma ve işleme fonksiyonları
def build_prompt_en(query: str, snippets: List[dict]) -> str:
    parts = []
    parts.append('Answer the user query using ONLY the provided snippets. If you cannot answer from the snippets, reply with exactly the single word: yetersiz')
    parts.append('\nUser query: ' + query)
    parts.append('Key terms from the question to stay on-topic: ' + query)
    parts.append('\nSnippets:')
    for i, s in enumerate(snippets[:5], start=1):
        title = (s.get('title') or '').strip()
        text = (s.get('text') or '').strip()
        if len(text) > 400:
            text = text[:400].rsplit(' ', 1)[0] + '...'
        parts.append(f"[{i}] {title}: {text}")
    parts.append('\nWrite exactly one concise factual sentence that directly answers the user query using only the snippets.')
    parts.append('Use key terms from the question so the sentence stays on-topic. Paraphrase; do not copy snippets verbatim.')
    parts.append('If the question asks for causes or risk factors, name the cause(s) or factor(s) from the snippets succinctly.')
    parts.append('If the question is about triggers/causes but the snippets do not explicitly mention triggers/causes, reply exactly: yetersiz')
    parts.append('Do NOT add headings, lists, citations, study mentions, or extra commentary; output only the single sentence or "yetersiz".')
    return '\n'.join(parts)

def sanitize_paragraph(par: str) -> str:
    if not par:
        return par
    par = re.sub(r'\[\d+\]', '', par)
    par = re.sub(r'\bSNIPPET\b', '', par, flags=re.I)
    par = re.sub(r'\bSNIPET\b', '', par, flags=re.I)
    par = re.sub(r'^(Abstract|Özet|Summary|Conclusion|Başlık)[:\-–—]\s*', '', par, flags=re.I)
    par = re.sub(r'([\.!?]){2,}', r'\1', par)
    par = re.sub(r'\s{2,}', ' ', par).strip()
    # Strip possible prompt leakage
    par = re.sub(r'\bno lists, no citations, no extra sections\.\?\b', '', par, flags=re.I).strip()
    par = re.sub(r'\bdo not add( any)? headings[^\.]*\.?', '', par, flags=re.I).strip()
    par = re.sub(r'\bstart directly with[^\.]*\.?', '', par, flags=re.I).strip()
    if par:
        par = par[0].upper() + par[1:]
    return par

def enforce_single_sentence(text: str, snippets: list | None = None) -> str:
    t = (text or '').strip()
    if not t:
        return 'yetersiz'
    t = re.sub(r'\s+', ' ', t).strip()
    # Strip leading heading-like patterns such as "Some Title: ..." or "Heading - ..."
    t = re.sub(r'^[\"\'\`\s]*[A-Za-z0-9 \-()]{1,100}[:\-–—]\s*', '', t)
    m = re.search(r'(.+?[\.!?])\s', t + ' ')
    s = m.group(1).strip() if m else t
    # Simple duplicate half heuristic
    words = s.split()
    if len(words) > 10:
        mid = len(words) // 2
        fh = re.sub(r'[^a-z0-9]', '', ' '.join(words[:mid]).lower())
        sh = re.sub(r'[^a-z0-9]', '', ' '.join(words[mid:]).lower())
        if fh and sh and (fh == sh or fh in sh or sh in fh):
            s = ' '.join(words[:mid]).rstrip(' ,;:') + '.'
    # Fallback to extractive if empty after cleaning
    if not s.strip() and snippets:
        for sn in snippets:
            txt = (sn.get('text') or '')
            for seg in txt.split('. '):
                seg = seg.strip()
                if len(seg) > 30:
                    s = seg.rstrip('.') + '.'
                    break
            if s:
                break
    if len(s) > 200:
        s = s[:200].rsplit(' ', 1)[0].rstrip(' ,;:') + '...'
    if s and s[-1] not in '.!?':
        s += '.'
    return s

def generate_one_sentence_en(prompt: str, max_new_tokens: int = 256, temperature: float = 0.5) -> str:
    import torch
    inputs = _gen_tok(prompt, return_tensors='pt', truncation=True, max_length=1024)
    try:
        inputs = {k: v.to(_GEN_DEVICE) for k, v in inputs.items()}
    except Exception:
        pass
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature == 0:
        gen_kwargs.update(dict(do_sample=False, num_beams=4, early_stopping=True, min_length=10))
    else:
        gen_kwargs.update(dict(do_sample=True, top_p=0.9, top_k=40, min_length=10, temperature=temperature))
    with torch.no_grad():
        out = _gen_mdl.generate(**inputs, **gen_kwargs)
    # Move to CPU before decoding if tensor is on GPU
    try:
        out_ids = out[0].to('cpu') if hasattr(out[0], 'device') else out[0]
    except Exception:
        out_ids = out[0]
    text = _gen_tok.decode(out_ids, skip_special_tokens=True)
    return enforce_single_sentence(sanitize_paragraph(text))

# ======================================================================
#                      SNIPPET SEÇİM YARDIMCISI
# ======================================================================
def pick_snippet(doc: dict) -> str:
    """
    Özet/fragman için alan seçimi:
    Öncelik: snippet -> abstract -> summary -> text -> body -> content
    Hiçbiri yoksa title döner.
    """
    for key in ["text", "snippet", "abstract", "summary", "body", "content"]:
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            # Heuristic: if snippet looks like a fragment (very short or starts with lowercase or missing first chars)
            def looks_broken(x: str) -> bool:
                if not x:
                    return True
                if len(x) < 30:
                    return True
                # starts with lowercase letter (likely cut) or starts with a non-letter
                first = x[0]
                if first.isalpha() and first.islower():
                    return True
                return False

            if looks_broken(s):
                # Try to expand snippet using precomputed neighbors map
                _id = str(doc.get("id", ""))
                if "_" in _id:
                    prefix, _, suffix = _id.rpartition("_")
                    try:
                        cur_idx = int(suffix)
                    except Exception:
                        cur_idx = None
                    if cur_idx is not None:
                        neighbors = _neighbors_map.get(prefix, [])
                        if neighbors:
                            # neighbors already sorted by chunk idx
                            # find prev chunk text if available
                            prev_text = None
                            for idx_i, t in neighbors:
                                if idx_i == cur_idx - 1:
                                    prev_text = t or ""
                                    break
                            if prev_text is not None:
                                prev_tail = prev_text[-300:]
                                combined = (prev_tail + " " + s).strip()
                                if len(combined) > len(s):
                                    return combined
            return s
    return str(doc.get("title", "") or "")

# ======================================================================
#                      QUERY EMBEDDING (PLACEHOLDER)
# ======================================================================
def embed_query(text: str) -> np.ndarray:
    """
    Sorgu metnini SentenceTransformer ile encode eder, L2-normalize eder ve float32 vektör döner.
    FAISS araması için tek örnek (1, d) şekline getirilir.
    """
    if not text:
        return np.zeros(EMBED_DIM, dtype="float32")
    emb = _embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return emb.reshape(-1).astype("float32")

# ======================================================================
#                               SEARCH
# ======================================================================
def search(query_en: str, k: int = 5):
    if k <= 0:
        k = 1
    if k > MAX_K:
        k = MAX_K

    # Use the query directly without condition-specific expansions
    qv = embed_query(query_en).reshape(1, -1)  # (1, d)
    # FAISS arama: genişçe alıp (top_n) cross-encoder ile yeniden sıralayacağız
    top_n = min(MAX_K * TOPN_MULT, TOPN_CAP) 
    D, I = index.search(qv, top_n)

    candidates = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta_data):
            continue
        doc = meta_data[idx]
        candidates.append((idx, float(dist), doc))

    # prepare pairs for cross-encoder: (query, candidate_text)
    pairs = []
    for _, _, doc in candidates:
        cand_text = pick_snippet(doc)
        pairs.append((query_en, cand_text))

    # score with cross-encoder (higher is better)
    if pairs:
        scores = _cross_encoder.predict(pairs)
    else:
        scores = [0.0] * len(candidates)

    # combine and sort using a weighted sum of normalized cross-encoder score and original FAISS score
    # normalize both to [0,1]
    # FAISS distances: smaller is better; convert to higher-is-better scores
    orig_scores = [-c[1] for c in candidates]
    ce_scores = list(scores)

    def minmax_norm(arr):
        if not arr:
            return []
        lo = min(arr)
        hi = max(arr)
        if hi - lo <= 1e-12:
            return [0.0 for _ in arr]
        return [(a - lo) / (hi - lo) for a in arr]

    orig_n = minmax_norm(orig_scores)
    ce_n = minmax_norm(ce_scores)

    alpha = RERANK_ALPHA  # weight for cross-encoder; from env

    # Generic lexical overlap boost (topic-agnostic): encourage snippets sharing query keywords
    def extract_keywords(q: str) -> set[str]:
        ql = (q or '').lower()
        # minimal stopword list; topic-agnostic
        stops = {
            'the','and','of','to','in','is','are','a','an','on','for','with','by','as','at','from','that','this',
            'which','what','when','how','why','where','who','whom','whose','be','or','it','its','into','over','under'
        }
        toks = re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", ql)
        return {t for t in toks if t not in stops and len(t) >= 3}

    q_terms = extract_keywords(query_en)
    lex_raw: list[float] = []
    for (_, _, doc) in candidates:
        txt = ((doc.get('title') or '') + ' ' + pick_snippet(doc)).lower()
        hits = 0
        for t in q_terms:
            if t in txt:
                hits += 1
        score = (hits / max(1, len(q_terms))) if q_terms else 0.0
        lex_raw.append(float(score))

    def minmax_norm_local(arr):
        if not arr:
            return []
        lo = min(arr)
        hi = max(arr)
        if hi - lo <= 1e-12:
            return [0.0 for _ in arr]
        return [ (a - lo) / (hi - lo) for a in arr ]

    lex_n = minmax_norm_local(lex_raw)
    beta = float(os.getenv('LEXICAL_BOOST', '0.2'))  # small, generic bias

    combined = []
    for (idx, dist, doc), o_n, c_n, l_n in zip(candidates, orig_n, ce_n, lex_n):
        base = alpha * c_n + (1.0 - alpha) * o_n
        final_score = base * (1.0 - beta) + beta * l_n
        combined.append({
            "id": str(doc.get("id", idx)),
            "title": str(doc.get("title", "") or ""),
            "url": str(doc.get("url", "") or ""),
            "source": str(doc.get("source", "") or ""),
            "score": float(final_score),
            "snippet": pick_snippet(doc),
            "_cross_score": float(c_n),
            "_orig_score": float(o_n)
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:k]

# ======================================================================
#                             ENDPOINTS
# ======================================================================
@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(inp: RetrieveRequest):
    """
    TR soruyu EN'e çevir -> FAISS ile ara -> sonuçların başlık & özetini TR'ye çevir.
    """
    t0 = time.time() #retrieve t0
    q_tr = inp.question
    if not q_tr.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # TR -> EN
    q_en = to_en(q_tr, source_hint="tr")

    hits_en = search(q_en, k=inp.k)

    # Sonuçları TR'leştir
    results: List[RetrieveResult] = []
    for h in hits_en:
        title_en = h["title"]
        snippet_en = h["snippet"]
        title_tr = to_tr(h["title"], source_hint="en")
        snippet_tr = to_tr(h["snippet"], source_hint="en")
        results.append(RetrieveResult(
            id=h["id"],
            title=title_tr,
            title_en=title_en,
            url=h["url"],
            score=h["score"],
            snippet=snippet_tr,
            snippet_en=snippet_en
        ))

    return RetrieveResponse(
        query_tr=q_tr,
        query_en=q_en,
        retrieval_ms=int((time.time() - t0) * 1000), #retrieve süresi ms cinsinden
        results=results
    )

@app.post("/qa", response_model=QAResponse)
def qa(inp: RetrieveRequest):
    """Unified schema: retrieve snippets and return generation placeholders with requested fields."""
    t0 = time.time()
    q_tr = inp.question
    if not q_tr.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")
    q_en = to_en(q_tr, source_hint="tr")

    # retrieval timing
    r0 = time.time()
    hits = search(q_en, k=inp.k)
    r1 = time.time()

    # Map hits -> unified snippet objects
    used_snippets = []
    for h in hits:
        used_snippets.append({
            "id": h["id"],
            "title": h["title"],
            "url": h["url"],
            "source": h.get("source", ""),
            "text": h["snippet"],
            "score": h["score"],
            "_cross_score": h.get("_cross_score", 0.0),
        })

    # Build prompt from EN query and top snippets (already EN in meta)
    g0 = time.time()
    prompt = build_prompt_en(q_en, used_snippets)
    english_text = generate_one_sentence_en(prompt, max_new_tokens=256, temperature=0.5)
    # Evidence guard for trigger/causes: require snippets to explicitly mention trigger/cause concepts
    # and have minimal relevance.
    def _has_trigger_evidence(snips: list[dict]) -> bool:
        if not snips:
            return False
        trigger_markers = [' trigger', 'triggers', 'precipitat', 'cause', 'causes', 'risk factor', ' risk factors', ' tetikley', ' neden']
        specific = ['stress', 'sleep deprivation', 'lack of sleep', 'bright light', 'alcohol', 'caffeine', 'chocolate', 'cheese', 'menstruation', 'menstrual', 'hormonal', 'estrogen']
        # minimal relevance gate based on cross-encoder normalized score if available
        # we stored _cross_score during retrieval normalization pipeline
        for s in snips:
            txt = ((s.get('title') or '') + ' ' + (s.get('text') or '')).lower()
            has_tc = any(w in txt for w in trigger_markers) or any(w in txt for w in specific)
            if not has_tc:
                continue
            try:
                ce_n = float(s.get('_cross_score', 0.0))
            except Exception:
                ce_n = 0.0
            # require modest relevance (normalized > 0.2) to avoid totally off-topic snippets triggering the guard
            if ce_n > EVIDENCE_CE_MIN:
                return True
        return False
    if any(x in q_en.lower() for x in ['trigger', 'triggers', 'cause', 'causes']) or any(x in q_tr.lower() for x in ['tetikley', 'neden']):
        if not _has_trigger_evidence(used_snippets):
            english_text = 'insufficient'
    # Normalize to detect 'yetersiz' or 'insufficient' regardless of case/punctuation
    norm = re.sub(r'[^a-z]', '', (english_text or '').lower())
    if norm in {'yetersiz', 'insufficient'}:
        english_text = 'insufficient'
        turkish_text = 'yetersiz'  # match pipeline expectation
    else:
        turkish_text = to_tr(english_text, source_hint='en')
    # Apply final sentence enforcement with snippet-aware fallback (skip if insufficient)
    if english_text.strip().lower() != 'insufficient':
        english_text = enforce_single_sentence(sanitize_paragraph(english_text), used_snippets)
    # Small normalization for Turkish medical terms (parity with demo)
    def _normalize_tr(s: str) -> str:
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
    turkish_text = _normalize_tr(turkish_text)
    g1 = time.time()

    # Final hard normalization: never return 'yetersiz' in english field
    try:
        _norm_en = re.sub(r'[^a-z]', '', (english_text or '').lower())
        if _norm_en == 'yetersiz':
            english_text = 'insufficient'
            if not turkish_text or re.search(r'insufficient', (turkish_text or ''), flags=re.I):
                turkish_text = 'yetersiz'
    except Exception:
        pass

    resp = QAResponse(
        query_tr=q_tr,
        query_en=q_en,
        retrieval_time_seconds=round(r1 - r0, 4),
        generation_time_seconds=round(g1 - g0, 4),
        total_time_seconds=round((r1 - r0) + (g1 - g0), 4),
        english={"text": english_text},
        turkish={"text": turkish_text},
        used_snippets=used_snippets,
    )

    # Best-effort SQLite logging (optional via env)
    try:
        from .sqlite_logger import log_qa
        log_qa(resp.model_dump(), k=inp.k)
    except Exception:
        pass

    return resp

@app.get("/health")
def health():
    # Basit durum raporu
    return {
        "status": "ok",
        "index_size": index.ntotal, # toplam chunk sayısı
        "faiss_dim": index.d,
        "expected_embed_dim": EMBED_DIM, 
        "meta_rows": len(meta_data), # toplam meta satır sayısı
        "generator_device": _GEN_DEVICE,
    }
