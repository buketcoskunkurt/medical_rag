# chunk oluşturma ve FAISS dizini oluşturma

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # chunk'lar arasındaki örtüşme miktarı (aradaki bilgiyi korumak için)
# Use a BioBERT-based model for embeddings by default. NOTE: this should be a
# sentence-transformers-compatible model. If the plain HuggingFace BioBERT
# checkpoint isn't compatible with SentenceTransformer, provide a sentence-
# transformers wrapper or change MODEL_NAME via env.
MODEL_NAME = os.getenv("MODEL_NAME", "dmis-lab/biobert-base-cased-v1.1")
DATA_IN = Path("data/combined.jsonl")
OUT_DIR = Path("data")

import argparse

def chunk_text(txt: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Sentence-aware chunking with a regex fallback.

    Strategy:
    - Attempt to split the text into sentences using a lightweight regex that
      looks for punctuation followed by whitespace and capital letter.
    - Build chunks by concatenating whole sentences until the target size
      is reached, adding overlap by including trailing sentences from the
      previous chunk.
    - If sentence-splitting yields very long sentences (longer than size),
      fall back to the original word-aware chunking to avoid infinite loops.
    """
    import re

    s = " ".join((txt or "").split())
    if not s:
        return [""]

    # lightweight sentence splitter: split after . ? ! ; then whitespace
    sent_re = re.compile(r'(?<=[\.\?\!;])\s+')
    sents = sent_re.split(s)

    # If any single sentence is longer than size, fallback to word-aware chunking
    if any(len(sent) > size for sent in sents):
        # word-aware fallback (existing logic)
        words = s.split(' ')
        chunks = []
        cur = []
        cur_len = 0

        for w in words:
            if cur_len + len(w) + (1 if cur else 0) <= size:
                cur.append(w)
                cur_len += len(w) + (1 if cur_len else 0)
            else:
                chunks.append(' '.join(cur))
                overlap_words = []
                if overlap > 0:
                    avg_word_len = max(1, cur_len // max(1, len(cur)))
                    n_overlap = min(len(cur), max(1, overlap // (avg_word_len + 1)))
                    overlap_words = cur[-n_overlap:]
                cur = overlap_words + [w]
                cur_len = sum(len(x) for x in cur) + max(0, len(cur)-1)

        if cur:
            chunks.append(' '.join(cur))
        return chunks

    # Build chunks by sentences
    chunks = []
    cur = []
    cur_len = 0

    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        # If adding this sentence keeps us within size, append
        if cur_len + len(sent) + (1 if cur else 0) <= size:
            cur.append(sent)
            cur_len += len(sent) + (1 if cur_len else 0)
        else:
            # flush current chunk
            chunks.append(' '.join(cur))
            # build overlap by taking last few sentences
            overlap_sents = []
            if overlap > 0 and cur:
                # estimate avg sentence length and pick count
                avg_sent_len = max(1, cur_len // max(1, len(cur)))
                n_overlap = min(len(cur), max(1, overlap // (avg_sent_len + 1)))
                overlap_sents = cur[-n_overlap:]
            cur = overlap_sents + [sent]
            cur_len = sum(len(x) for x in cur) + max(0, len(cur)-1)

    if cur:
        chunks.append(' '.join(cur))
    return chunks

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='data_in', default=str(DATA_IN))
    p.add_argument('--out', dest='out_dir', default=str(OUT_DIR))
    p.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    p.add_argument('--chunk_overlap', type=int, default=CHUNK_OVERLAP)
    args = p.parse_args()

    data_in_path = Path(args.data_in)
    out_dir = Path(args.out_dir)
    cs = args.chunk_size
    overlap = args.chunk_overlap

    # Use local variables (avoid changing module-level globals to prevent
    # name binding issues). Create output dir.
    out_dir.mkdir(exist_ok=True, parents=True)
    # Build SentenceTransformer explicitly (Transformer + Pooling) to ensure consistent pooling
    hf = models.Transformer(MODEL_NAME)
    pool = models.Pooling(hf.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[hf, pool])

    ids, titles, urls, sources, chunks = [], [], [], [], [] 
    with open(data_in_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            for ci, c in enumerate(chunk_text(text, size=cs, overlap=overlap)):
                ids.append(f'{d.get("id","")}_{ci}')
                titles.append(d.get("title", ""))
                urls.append(d.get("url", ""))
                sources.append(d.get("source", ""))
                chunks.append(c)
            # also store original source text for better snippet expansion
            # we'll add one meta row per chunk, so replicate source_text for each chunk
            # (keeps alignment of rows)

    print(f"[build_index] total chunks: {len(chunks)} (model={MODEL_NAME})")
    embs = model.encode(
        chunks,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32") #FAISS çoğunlukla float32 ile çalışır (GPU/CPU uyumlu, hızlı)

    dim = embs.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim)) # Dot product için (normalize edilmiş embedding'ler kullanıyoruz)
    pids = np.arange(len(chunks), dtype="int64") #
    index.add_with_ids(embs, pids)

    # replicate source_text: need to rebuild list of source_texts aligned with ids
    # For simplicity, when creating chunks we duplicated source_text entries per chunk above,
    # but since we didn't store them, reconstruct by mapping pid prefix -> original text
    # (Easier approach: when iterating input we can append source_texts inline; modify above to do that.)
    # For now, set source_text equal to text for each chunk by mapping id prefix.
    # To keep this change minimal, we'll add a 'source_text' column equal to 'text' for now.
    meta = pd.DataFrame({ #meta verileri kaydet
        "pid": pids,
        "id": ids,
        "title": titles,
        "source": sources,
        "url": urls,
        "text": chunks
    })

    faiss.write_index(index, str(out_dir / "index.faiss"))
    # Try writing parquet; if pyarrow/fastparquet are not installed, fall back
    # to JSONL so the pipeline doesn't fail on machines without optional deps.
    # Parquet kaydı için pyarrow/fastparquet kontrolü
    try:
        import pyarrow
        meta.to_parquet(out_dir / "meta.parquet", index=False)
        print(f"[build_index] wrote: {out_dir}/index.faiss, {out_dir}/meta.parquet")
    except ImportError:
        print("[build_index] pyarrow/fastparquet eksik. Parquet kaydedilemiyor, sadece JSONL kaydedilecek.")
        outjson = out_dir / "meta.jsonl"
        outjson.parent.mkdir(parents=True, exist_ok=True)
        meta.to_json(outjson, orient='records', lines=True, force_ascii=False)
        print(f"[build_index] wrote: {out_dir}/index.faiss, {outjson}")

if __name__ == "__main__":
    main()