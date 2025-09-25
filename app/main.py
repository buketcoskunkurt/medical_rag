import time
import os
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.parquet"

app = FastAPI(title="rag-med-minimal (retriever)")

# Load artifacts
index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)
model = SentenceTransformer(MODEL_NAME)

class Query(BaseModel):
    question: str
    k: int | None = 5

@app.post("/retrieve")
def retrieve(q: Query):
    t0 = time.time()
    q_emb = model.encode([q.question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, pids = index.search(q_emb, q.k or 5)
    results = []
    for pid, sc in zip(pids[0], scores[0]):
        r = meta.iloc[int(pid)]
        results.append({
            "id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "score": float(sc),
            "snippet": r["chunk"][:600]
        })
    return {
        "query": q.question,
        "retrieval_ms": int((time.time() - t0) * 1000),
        "results": results
    }

@app.get("/health")
def health():
    return {"status": "ok", "chunks": int(meta.shape[0])}