import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DATA_IN = Path("data/pubmed.jsonl")
OUT_DIR = Path("data")

def chunk_text(txt: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    txt = " ".join((txt or "").split())
    if not txt:
        return [""]
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(txt):
        chunks.append(txt[i:i+size])
        i += step
    return chunks

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    model = SentenceTransformer(MODEL_NAME)

    ids, titles, urls, chunks = [], [], [], []
    with open(DATA_IN, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            for ci, c in enumerate(chunk_text(text)):
                ids.append(f'{d.get("id","")}_{ci}')
                titles.append(d.get("title", ""))
                urls.append(d.get("url", ""))
                chunks.append(c)

    print(f"[build_index] total chunks: {len(chunks)} (model={MODEL_NAME})")
    embs = model.encode(
        chunks,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    pids = np.arange(len(chunks), dtype="int64")
    index.add_with_ids(embs, pids)

    meta = pd.DataFrame({
        "pid": pids,
        "id": ids,
        "title": titles,
        "url": urls,
        "chunk": chunks
    })

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    meta.to_parquet(OUT_DIR / "meta.parquet", index=False)
    print("[build_index] wrote: data/index.faiss, data/meta.parquet")

if __name__ == "__main__":
    main()