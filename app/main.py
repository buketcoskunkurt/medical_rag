# app/main.py
import os
import time
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers import CrossEncoder

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.argos_util import translate_offline

# ======================================================================
#                           YAPILANDIRMA
# ======================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index.faiss")      # FAISS index
META_PATH = os.path.join(BASE_DIR, "data", "meta.parquet")      # Parquet meta

# FAISS index ile aynı olmalı (index'i üretirkenki embedding boyutu)
# intfloat/multilingual-e5-base => 768-dim
EMBED_DIM = 768

# load embedding model once at startup (use BioBERT by default for biomedical domain)
# Default embedder: dmis-lab/biobert-base-cased-v1.1 (768-dim). Ensure compatibility
# with SentenceTransformer or replace with a sentence-transformers wrapper.
EMB_MODEL = os.getenv("EMB_MODEL", "dmis-lab/biobert-base-cased-v1.1")
# Build a SentenceTransformer wrapper explicitly to avoid "creating a new one with mean pooling" message
_hf = models.Transformer(EMB_MODEL)
_pool = models.Pooling(_hf.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
_embed_model = SentenceTransformer(modules=[_hf, _pool])
 
# generator model (for future RAG / generation) - default to a BioGPT variant
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "microsoft/biogpt")

# kaç sonuç getirileceği için bir üst sınır (kötü kullanım önlemi)
MAX_K = 20
# Cross-encoder reranker (optional but improves top-k quality)
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
# reranker weight (alpha) controls contribution of cross-encoder vs FAISS score
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.6"))
# top_n multiplier controls how many FAISS candidates to prefetch before reranking
TOPN_MULT = int(os.getenv("TOPN_MULT", "20"))
# absolute cap for top_n
TOPN_CAP = int(os.getenv("TOPN_CAP", "500"))
# ======================================================================
#                        INDEX & META YÜKLEME
# ======================================================================
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index bulunamadı: {INDEX_PATH}")

index = faiss.read_index(INDEX_PATH)

if index.d != EMBED_DIM:
    # FAISS index dimension ile EMBED_DIM uyuşmuyorsa erken uyarı
    raise ValueError(
        f"FAISS index dim({index.d}) != EMBED_DIM({EMBED_DIM}). "
        f"Index hangi embedding ile üretildiyse EMBED_DIM ona eşit olmalı."
    )

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Meta dosyası bulunamadı: {META_PATH}")

# Parquet -> list[dict]
meta_df = pd.read_parquet(META_PATH)
meta_data = meta_df.to_dict(orient="records")

# precompute neighbors map: prefix -> list of (chunk_idx, text)
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

# ======================================================================
#                           FASTAPI MODEL
# ======================================================================
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

# ======================================================================
#                      ÇEVİRİ YARDIMCI FONKSİYONLAR
# ======================================================================
def to_tr(text: str, source_hint: str = "en") -> str:
    """
    EN -> TR offline çeviri (Argos). Hata olursa metni değiştirme.
    """
    try:
        if not text or source_hint == "tr":
            return text
        return translate_offline(text, source_hint, "tr")
    except Exception:
        return text

def to_en(text: str, source_hint: str = "tr") -> str:
    """
    TR -> EN offline çeviri (Argos). Hata olursa metni değiştirme.
    """
    try:
        if not text or source_hint == "en":
            return text
        return translate_offline(text, source_hint, "en")
    except Exception:
        return text

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
    DEMO amaçlı: deterministik 'mock' embedding.
    (Gerçek projede burada sentence-transformers vb. ile gerçek embedding üret)
    """
    # use the same encoder as the index builder; returns normalized float32 vector
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

    qv = embed_query(query_en).reshape(1, -1)  # (1, d)
    # FAISS arama: genişçe alıp (top_n) cross-encoder ile yeniden sıralayacağız
    top_n = min(MAX_K * 10, 200)
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
    import math
    orig_scores = [c[1] for c in candidates]
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

    alpha = 0.6  # weight for cross-encoder; tweakable
    combined = []
    for (idx, dist, doc), o_n, c_n in zip(candidates, orig_n, ce_n):
        final_score = alpha * c_n + (1.0 - alpha) * o_n
        combined.append({
            "id": str(doc.get("id", idx)),
            "title": str(doc.get("title", "") or ""),
            "url": str(doc.get("url", "") or ""),
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
    t0 = time.time()
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
        retrieval_ms=int((time.time() - t0) * 1000),
        results=results
    )

@app.get("/health")
def health():
    # Basit durum raporu
    return {
        "status": "ok",
        "index_size": index.ntotal,
        "faiss_dim": index.d,
        "expected_embed_dim": EMBED_DIM,
        "meta_rows": len(meta_data),
    }
