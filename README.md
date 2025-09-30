# medical_rag

Basit ama sağlam bir RAG (Retrieve Augmented Generation) zinciri:
- Verileri chunk’layıp SentenceTransformer ile embedding üretir ve FAISS’e yazar.
- FastAPI ile iki uç sunar: `/retrieve` ve `/qa`.
- Cevap üretimi için kısa ve tek cümlelik Flan‑T5 jeneratörü kullanır.
- Çeviri offline ve sorunsuz olması için Argos Translate ile yapılır (EN↔TR).

## 1) Kurulum

Python 3.10+ önerilir. PyTorch’u donanıma göre kurun (CPU/GPU). Aşağıdaki örnek PowerShell içindir.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Argos en↔tr dil paketleri (offline çeviri) için:

```powershell
python scripts\install_argos.py
```

## 2) Veri Hazırlama (opsiyonel)

Repoda hazır küçük bir birleşik veri dosyası var: `data/combined_small.jsonl`. Kendi verinizi oluşturmak isterseniz:

1) OpenAlex’tan çekin:

```powershell
python scripts\fetch_openalex.py --query "sleep insomnia" --retmax 500 --out data\raw\openalex_sleep.jsonl --overwrite
```

2) PubMed’ten çekin (NCBI API için email belirtin):

```powershell
python scripts\fetch_pubmed.py --query "insomnia" --retmax 1000 --batch-size 50 --email you@example.com --out data\raw\pubmed_sleep.jsonl --overwrite --dedup --mindate 2015
```

3) Hepsini birleştirin (basit dedup ile):

```powershell
python scripts\build_combined_from_raw.py --raw-dir data\raw --out data\combined.jsonl --dedup-mode url+hash
```

Not: İsterseniz mevcut `data\combined_small.jsonl` dosyasını doğrudan kullanabilirsiniz.

## 3) BioBERT’i safetensors olarak dışa aktarma (önerilen)

Bu adım BioBERT’i yerel klasöre güvenli şekilde indirip `model.safetensors` olarak kaydeder.

```powershell
python scripts\export_biobert_safetensors.py --model dmis-lab/biobert-base-cased-v1.1 --out models\biobert-base-cased-v1.1-sf
```

Çıktı klasörü: `models\biobert-base-cased-v1.1-sf` (config + tokenizer + model.safetensors)

## 4) İndeks oluşturma

- Girdi dosyası: `data/combined_small.jsonl` (veya 2. adımda oluşturduğunuz `data/combined.jsonl`)
- BioBERT safetensors ile 768‑dim indeks üretimi (çıktı `data\index.faiss`, `data\meta.parquet`):

```powershell
# Hazır küçük veriyle (varsayılan dosya: data\combined_small.jsonl)
python -u scripts\build_index.py --in data\combined_small.jsonl --out data --emb-model models\biobert-base-cased-v1.1-sf

# Kendi oluşturduğunuz birleşik veriyle
python -u scripts\build_index.py --in data\combined.jsonl --out data --emb-model models\biobert-base-cased-v1.1-sf
```

Notlar:
- `--emb-model` alanına yerel bir klasör (ör. `models\biobert-base-cased-v1.1-sf`) veya bir HF model adı verebilirsiniz.
- FAISS dim, seçtiğiniz embedding modelinin çıktı boyutudur; API tarafında da aynı embedder kullanılmalıdır.

## Hazır ZIP artifacts ile hızlı kurulum (Model + İndeks)


1) ZIP’leri indir (GitHub Releases)

- GitHub’da Releases sayfasını açın: https://github.com/buketcoskunkurt/medical_rag/releases
- Aşağıdaki iki dosyayı indirin:
  - biobert_model.zip (model)
  - faiss_index_biobert.zip (index + meta)
- İsterseniz yerelde `artifacts/` klasörü oluşturup dosyaları oraya koyabilirsiniz:

```powershell
mkdir artifacts -ea 0
# İndirilen zip dosyalarını artifacts/ içine taşıyın (Explorer veya PowerShell ile)
```

2) ZIP’leri doğru konuma aç

```powershell
# Model (BioBERT) → models\biobert-base-cased-v1.1-sf\
Expand-Archive artifacts\biobert_model.zip -DestinationPath models\biobert-base-cased-v1.1-sf -Force

# İndeks → data\ (index.faiss ve meta.parquet bu klasörde olmalı)
Expand-Archive artifacts\faiss_index_biobert.zip -DestinationPath data -Force
```

Beklenen dosyalar:
- models\biobert-base-cased-v1.1-sf\model.safetensors, config.json, tokenizer dosyaları
- data\index.faiss, data\meta.parquet

3) İndeks boyutunu doğrula (BioBERT = 768-dim)

```powershell
python -c "import faiss; print(f'FAISS dim = {faiss.read_index(r"data\\index.faiss").d}')"
```

Çıktı 768 olmalı. Farklıysa indeks ile embed modeli eşleşmiyor demektir.

## 5) Servisi çalıştırma

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Sonra:
- http://localhost:8000/health (index boyutu/dim ve cihaz bilgisi)
- http://localhost:8000/docs (Swagger UI)

### /retrieve (POST)
Örnek gövde:

```json
{"question":"Uykusuzluğun yaygın nedenleri nelerdir?", "k":5}
```

Dönen şema:
- query_tr, query_en
- retrieval_ms
- results[]: id, title (TR), title_en (EN), url, score, snippet (TR), snippet_en (EN)

### /qa (POST)
Örnek gövde:

```json
{"question":"Uykusuzluğun yaygın nedenleri nelerdir?", "k":5}
```

Dönen şema:
- query_tr, query_en
- retrieval_time_seconds, generation_time_seconds, total_time_seconds
- english: { text }
- turkish: { text }
- used_snippets: [{ id, title, url, text, score }]

“Yetersiz” halinde: english.text = "insufficient", turkish.text = "yetersiz".

## 6) Basit UI (Streamlit)

API üzerinde basit bir arayüzle soru sormak için:

```powershell
streamlit run ui\streamlit_app.py
```

Notlar:
- API varsayılanı `http://localhost:8000`’dir; sol panelden değiştirebilirsiniz.
- Soru yazıp “Ask” ile çalıştırın; üstte tek cümlelik EN yanıt, altta referanslar listelenir.

## İpuçları ve Sorun Giderme

- Boyut eşleşmesi: `/health` çıktısında `faiss_dim` ile `expected_embed_dim` değerlerinin eşit olduğundan emin olun. Eşit değilse indeksi aynı embedder (ör. BioBERT 768-dim) ile yeniden oluşturun.
- Parquet yazımı için `pyarrow` yoksa script otomatik olarak `meta.jsonl` üretir; API tarafı `meta.parquet` bekler, bu yüzden `pyarrow` kurulu olduğundan emin olun.

## Proje Yapısı

```
medical_rag/
  app/
    main.py              # FastAPI (/retrieve, /qa)
  scripts/
    test/
      evaluate_metrics_en.py
      evaluate_perplexity_en.py
      run_performance_tests_en.py
    build_index.py       # combined*.jsonl -> index.faiss + meta.parquet
    build_combined_from_raw.py
    fetch_openalex.py
    fetch_pubmed.py
    install_argos.py     # Argos en↔tr paketleri
    export_biobert_safetensors.py
  data/
    combined_small.jsonl # Örnek küçük veri
    index.faiss
    meta.parquet
  models/
    biobert-base-cased-v1.1-sf/  # Yerel safetensors BioBERT (önerilen)
  requirements.txt
  README.md
```