# rag-med-minimal

Basit bir **RAG (Retrieve-then-Read)** örneği: küçük bir gövdeyle metinlerden **embedding** çıkarır, **FAISS** ile en benzer parçaları bulur ve bir **FastAPI** endpoint'i üzerinden döner. İlk hedef: **retrieval çalışsın**. (LLM ile cevap üretme opsiyoneldir, sonra eklenebilir.)

## Hızlı Başlangıç

> Python 3.10+ önerilir. `torch` kurulumunu işletim sistemine göre ayrıca yapın (aşağıda).

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch (CUDA'lı sistemler için resmi index kullanın; CPU isterseniz normal pip'ten yükleyin)
# CUDA'lı (örnek: CUDA 12.1 wheel):
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### 1) Örnek veri
`data/pubmed.jsonl` içinde örnek iki kayıt var. Kendi verinizi aynı formatta büyütebilirsiniz:

```json
{"id":"pmid1","title":"Metformin and T2D","text":"Metformin is first-line in T2D per guidelines...","url":"https://example.org/pmid1"}
{"id":"pmid2","title":"Hypertension stages","text":"Stage 1 hypertension management includes lifestyle...","url":"https://example.org/pmid2"}
```

### 2) İndeks oluştur
```bash
python scripts/build_index.py
```

### 3) Servisi çalıştır
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) Test
```bash
curl -s -X POST http://localhost:8000/retrieve -H "Content-Type: application/json"   -d '{"question":"Tip 2 diyabette ilk basamak?", "k":5}' | jq .
```

## Modeli değiştirme (BioBERT vb.)
Varsayılan model genel bir ST modelidir (`sentence-transformers/all-MiniLM-L6-v2`). Biyomedikal bir ST checkpoint'i kullanmak isterseniz:

```bash
export MODEL_NAME="pritamdeka/S-BioBert-snli-sts"   # örnek isim
python scripts/build_index.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Eğer ST olmayan BioBERT kullanacaksanız, pooling'i manuel uygulayan sürüme geçebilirsiniz (ileride `scripts/build_index_biobert_torch.py` ekleyebilirsiniz).

## Yapı
```
rag-med-minimal/
  app/
    main.py
  scripts/
    build_index.py
  data/
    pubmed.jsonl
  models/           # (ileride training çıktıları)
  requirements.txt
  README.md
  .gitignore
```