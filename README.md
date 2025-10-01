## Proje Açıklaması  

Bu proje, medikal sorulara bilimsel kaynaklardan desteklenmiş yanıtlar üreten bir **Retrieval-Augmented Generation (RAG)** sistemidir.  

- **Chunking & Embedding:** Makale özetleri belirli boyutlarda parçalanır ve **SentenceTransformer (BioBERT)** kullanılarak embedding vektörleri üretilir. Bu vektörler **FAISS** veritabanında saklanır.  
- **Retrieval:** Kullanıcı soruları embedding’e dönüştürülüp FAISS üzerinde arama yapılır, sonuçlar **Cross-Encoder** ve **Lexical Boost** yöntemleriyle yeniden sıralanır.  
- **Generation:** Elde edilen snippet’ler kullanılarak **Flan-T5** jeneratörü ile kısa, tek cümlelik yanıtlar üretilir.  
- **Çeviri:** Yanıtlar İngilizce üretilir, ardından **Argos Translate** ile Türkçe’ye çevrilerek çift dil desteği sağlanır.  
- **API:** Sistem, **FastAPI** ile `/retrieve` ve `/qa` uç noktaları üzerinden erişilebilir.  

Ek özellikler:  
- **Performans metrikleri** (Retrieval, Generation süreleri)  
- **Opsiyonel loglama** (SQLite)  
- **Opsiyonel Streamlit UI** (kolay demo arayüzü)  

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

Repoda hazır iki örnek veri dosyası bulunmaktadır:  

- `data/combined_small.jsonl` → Küçük boyutlu set (~3.8K abstract) — hızlı test ve geliştirme için.  
- `data/combined_large.jsonl` → Büyük boyutlu set (~23K abstract) — daha kapsamlı arama ve değerlendirme için.  

Varsayılan kurulum küçük veri setini kullanır. Daha kapsamlı deneyim için `combined_large.jsonl` dosyasını FAISS indexleme aşamasında kullanabilirsiniz.

### Kendi verinizi oluşturmak isterseniz 

1) OpenAlex’tan çekin:

```powershell
python scripts\fetch_openalex.py --query "sleep insomnia" --retmax 500 `
  --out data\raw\openalex_sleep.jsonl --overwrite
```

2) PubMed’ten çekin (NCBI API için email belirtin):

```powershell
python scripts\fetch_pubmed.py --query "insomnia" --retmax 1000 --batch-size 50 `
  --email you@example.com --out data\raw\pubmed_sleep.jsonl --overwrite --dedup --mindate 2015
```

3) Hepsini birleştirin (basit dedup ile):

```powershell
python scripts/export_biobert_safetensors.py `
  --model dmis-lab/biobert-base-cased-v1.1 `
  --out models/biobert-base-cased-v1.1-sf
```

Not: İsterseniz yukarıdaki adımları uygulamadan, doğrudan hazır gelen `data\combined_small.jsonl` veya `data\combined_large.jsonl` dosyalarını kullanabilirsiniz.

## 3) BioBERT’i safetensors olarak dışa aktarma (önerilen)

Bu adım BioBERT’i yerel klasöre güvenli şekilde indirip `model.safetensors` olarak kaydeder.

```powershell
python scripts\export_biobert_safetensors.py --model dmis-lab/biobert-base-cased-v1.1 --out models\biobert-base-cased-v1.1-sf
```

Çıktı klasörü: `models\biobert-base-cased-v1.1-sf` (config + tokenizer + model.safetensors)

## 4) İndeks oluşturma

FAISS aramaları için önce veriyi embedding vektörlerine dönüştürüp indekslemeniz gerekir.  

- Girdi dosyası seçenekleri:  
  - `data/combined_small.jsonl` → Küçük set (~3.8K abstract) — hızlı test için  
  - `data/combined_large.jsonl` → Büyük set (~23K abstract) — daha kapsamlı deneyler için  
  - `data/combined.jsonl` → 2. adımda kendiniz oluşturduysanız  

- Embedding modeli: **BioBERT safetensors** (`models/biobert-base-cased-v1.1-sf`)  
- Çıktı dosyaları: `data/index.faiss` ve `data/meta.parquet`  

### Örnek Komut (PowerShell)

```powershell
python -u scripts\build_index.py --in data\combined_small.jsonl --out data --emb-model models\biobert-base-cased-v1.1-sf
```

Notlar: - --emb-model alanına yerel bir klasör (ör. models\biobert-base-cased-v1.1-sf) veya bir HF model adı verebilirsiniz. - FAISS dim, seçtiğiniz embedding modelinin çıktı boyutudur; API tarafında da aynı embedder kullanılmalıdır.

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
python -c "import faiss; print('FAISS dim =', faiss.read_index('data/index.faiss').d)"
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

## 7) Loglama (Opsiyonel)

API çağrıları SQLite veritabanına loglanabilir (varsayılan local çalışma için).  
Docker image içerisinde bu özellik devre dışı bırakılmıştır.

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

## Docker 

Bu projeyi Docker ile paketleyip çalıştırabilirsiniz. Image, CUDA destekli bir runtime (nvidia/cuda:12.1) üzerinde Python 3.10 kullanır; FAISS CPU’da, jeneratör (Flan‑T5) ise GPU varsa otomatik olarak CUDA’yı kullanır (app/main.py içindeki torch.cuda.is_available kontrolü sayesinde).

Önkoşullar:
- Docker kurulu olmalı
- GPU kullanacaksanız: NVIDIA sürücüleri + nvidia-container-toolkit

### 1) Build

```powershell
docker build -t medical-rag .
```

### 2) Çalıştır (CPU)

```powershell
docker run --rm -p 8080:8080 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\models:/app/models" `
  medical-rag
```

### 3) Sağlık testi

```powershell
Invoke-WebRequest -Uri "http://localhost:8080/health" -Method GET
```

### 4) Örnek QA çağrısı

```powershell
Invoke-WebRequest -Uri "http://localhost:8080/qa" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"question":"What are migraine triggers?","k":5}'
```

### 5) GPU ile çalıştırma (opsiyonel)

```powershell
docker run --rm --gpus all -p 8080:8080 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\models:/app/models" `
  medical-rag
```

### 6) API Testi

Swagger UI üzerinden test etmek için:
- Docker çalıştırmada: http://localhost:8080/docs

### 7) Streamlit UI(Opsiyonel)
Repo’da bir Streamlit arayüzü varsa (streamlit_app.py), Docker’daki API’ya bağlanarak yerelden çalıştırabilirsiniz. Docker içinde değil, kendi makinenizde çalıştırmalısınız.

Önkoşul:
```powershell
pip install streamlit>=1.37.0
```
Çalıştırma:

API konteynerini başlatın (ayrı terminalde açık kalsın)
Streamlit arayüzünü kendi makinenizde çalıştırın
```powershell
streamlit run ui/streamlit_app.py --server.port 8501
```
Streamlit uygulamasında API taban URL’sini "http://localhost:8080" olarak değiştirin.


Notlar:
- Image, modelleri ve veriyi içermez. `/models` ve `/data` host’tan volume olarak bağlanmalıdır; app içinde `/app/models` ve `/app/data` olarak erişilir.
- FAISS CPU paketidir (faiss-cpu). Retrieval CPU üzerinde çalışır; jeneratör model (Flan‑T5) GPU varsa otomatik CUDA’ya geçer.
- İlk çalıştırmada modelleri (Flan‑T5 vb.) indirmek birkaç dakikanızı alabilir; önceden host’taki `models/` klasörüne indirdiğinizde konteyner direkt kullanır.