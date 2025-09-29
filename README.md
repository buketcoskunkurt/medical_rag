# medical_rag

Basit ama sağlam bir RAG (Retrieve‑then‑Generate) zinciri:
- Verileri chunk’layıp SentenceTransformer ile embedding üretir ve FAISS’e yazar.
- FastAPI ile iki uç sunar: `/retrieve` ve `/qa`.
- Cevap üretimi için kısa ve tek cümlelik bir Flan‑T5 jeneratörü kullanır.
- Çeviri offline ve sorunsuz olması için Argos Translate ile yapılır (EN↔TR).

Önemli notlar:
- FAISS indeksinin embedding boyutu ile API’de kullanılan model aynı olmalı. Bu repo’daki eski hazır indeks 384 boyutundaydı (MiniLM). Kendi indeksinizi oluştururken hangi modelle embed ettiyseniz, sorgulama tarafında da onu kullanın.
- CUDA ile Torch < 2.6 kullanıyorsanız, Hugging Face .bin yüklemeleri güvenlik nedeniyle engellenebilir; bunun için yerel safetensors modele geçilmiştir (BioBERT). Aşağıdaki adımları takip edin.
- “Yetersiz” durumunda API şu şemayı döner: english.text = "insufficient", turkish.text = "yetersiz".

## Kurulum

Python 3.10+ önerilir. PyTorch’u donanıma göre kurun (CPU/GPU). Aşağıdaki örnek CPU içindir.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Argos dil paketleri (en↔tr) için script:

```bash
python scripts/install_argos.py
```

## Veri ve İndeks

- Girdi dosyası: `data/combined.jsonl` (id, title, text, url, source alanları)
- İndeks oluşturma (BioBERT safetensors ile 768‑dim):

```powershell
# (Opsiyonel) BioBERT'i CPU güvenli ortamda safetensors olarak dışa aktarın
conda run -n rag-export python scripts\export_biobert_safetensors.py --model dmis-lab/biobert-base-cased-v1.1 --out models\biobert-base-cased-v1.1-sf

# Ardından indeksi oluşturun (varsayılan artık yerel safetensors kullanan BioBERT yoludur)
conda run -n rag-med python -u scripts\build_index.py --in data\combined.jsonl --out data --emb-model models\biobert-base-cased-v1.1-sf
```

Notlar:
- Build script artık `--emb-model` kabul eder; yerel bir klasör veya HF repo adı verebilirsiniz.
- Oluşan FAISS boyutu seçtiğiniz modelin çıktı boyutudur; sorgulama/API tarafında da aynı modeli kullanmalısınız.

## Servisi Çalıştırma

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### /health
Basit durum: index boyutu ve dim bilgisi.

### /retrieve (POST)
Gövde:

```json
{"question":"Uykusuzluğun yaygın nedenleri nelerdir?", "k":5}
```

Dönen şema:
- query_tr, query_en
- retrieval_ms
- results[]: id, title (TR), title_en (EN), url, score, snippet (TR), snippet_en (EN)

### /qa (POST)
Gövde:

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

## CLI Demo (tek cümlelik QA)

API’ye gerek duymadan örnek üretim almak için:

```powershell
# 768‑dim BioBERT ile üretilmiş indeksle demo örneği
python scripts\generate_qa.py --query "Uykusuzluğun yaygın nedenleri nelerdir?" --lang tr \
  --topk 5 --cand 20 --index data\index.faiss --meta data\meta.parquet \
  --emb-model models\biobert-base-cased-v1.1-sf \
  --out data\demo_out_biobert.json
```

Notlar:
- `--lang tr` ile TR sorular EN’e çevrilir, üretim EN yapılır, sonra TR’ye çevrilir.
- “Yetersiz” ise çıktı: EN="insufficient", TR="yetersiz".

## Gelişmiş Notlar

- Embedding Boyutu: API `app/main.py` içinde kullanılan embedder ile FAISS dim eşleşmelidir. Yeni indeksinizi BioBERT (768‑dim) ile ürettiyseniz, API’yı da aynı embedder ile başlatın.
- Çeviri: Argos Translate offline paketleriyle çalışır; HF/çevrimiçi bağımlılık yoktur.

## Proje Yapısı

```
medical_rag/
  app/
    main.py              # FastAPI uygulaması (/retrieve, /qa)
  scripts/
    build_index.py       # combined.jsonl -> index.faiss + meta.parquet
    generate_qa.py       # Tek cümlelik QA demo scripti
    install_argos.py     # Argos en↔tr paketleri kurulumu
  data/
    combined.jsonl       # Girdi veri seti
    index.faiss
    meta.parquet
  requirements.txt
  README.md
```