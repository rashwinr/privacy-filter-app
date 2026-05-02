# Privacy Filter App

Test harness for [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) — upload a file, get personal information detected and redacted, download both versions.

- **Backend:** FastAPI (single container)
- **Model:** `openai/privacy-filter` (1.5B params, 50M active, Apache-2.0)
- **Frontend:** vanilla HTML/JS (no build step)
- **Formats:** `.txt`, `.md`, `.csv`, `.pdf`, `.docx`, `.png/.jpg/.tiff` (OCR), `.dcm` (DICOM tag scrub)
- **Deploy target:** Google Cloud Run (CPU, 4 GiB / 2 vCPU)

PII categories detected: `private_person`, `private_email`, `private_phone`, `private_address`, `private_url`, `private_date`, `account_number`, `secret`.

---

## 1. Local development with conda

```bash
# from the project root
conda env create -f environment.yml
conda activate privacy-filter

# (macOS) if Tesseract not on PATH:
brew install tesseract poppler

# Run
cp .env.example .env
uvicorn app.main:app --reload --port 8080
```

Open http://localhost:8080. First request triggers a one-time model download (~1.5 GB) into `~/.cache/huggingface/`.

### File locations
- Originals → `./data/uploads/{job_id}__{filename}`
- Redacted → `./data/redacted/{job_id}__redacted.{ext}`

Both are downloadable from the UI and persist across runs.

---

## 2. Run with Docker locally

```bash
docker build -t privacy-filter .
docker run --rm -p 8080:8080 -v "$PWD/data:/tmp/data" privacy-filter
```

The model weights are baked into the image during build, so cold start is ~5 s instead of ~60 s.

---

## 3. Deploy to Cloud Run

### Prerequisites
- `gcloud` CLI authenticated: `gcloud auth login && gcloud auth application-default login`
- A GCP project with billing enabled
- Cloud Run + Cloud Build APIs enabled:
  ```bash
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
  ```

### Option A — One-command deploy

```bash
PROJECT_ID=your-gcp-project ./scripts/deploy.sh
```

For persistent storage across instances/redeployments, create a GCS bucket and pass it:
```bash
gsutil mb -l asia-south1 gs://your-bucket-name
PROJECT_ID=your-gcp-project GCS_BUCKET=your-bucket-name ./scripts/deploy.sh
```

### Option B — Cloud Build pipeline

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_REGION=asia-south1,_SERVICE=privacy-filter,_BUCKET=your-bucket-name
```

### Recommended Cloud Run settings (already in `deploy.sh`)
| Setting          | Value         | Why                                                         |
|------------------|---------------|-------------------------------------------------------------|
| Memory           | 4 GiB         | Holds 1.5B-param MoE weights + Tesseract working set        |
| CPU              | 2 vCPU        | Token-class throughput on CPU                               |
| Concurrency      | 4             | Single uvicorn worker, model is GIL-bound                   |
| Min instances    | 0             | Scale to zero when idle                                     |
| Max instances    | 5             | Adjust for your workload                                    |
| Timeout          | 300 s         | Large PDFs / OCR can take time                              |
| CPU boost (opt.) | enabled       | `--cpu-boost` cuts cold start by ~30 %                      |

### Ephemeral vs persistent storage on Cloud Run
- Default (`STORAGE_BACKEND=local`) writes to `/tmp/data`, which is **per-instance and lost on restart**. Fine for testing.
- Set `STORAGE_BACKEND=gcs` + `GCS_BUCKET=…` to make originals + redacted outputs survive restarts and shareable across instances.
- The runtime service account needs `roles/storage.objectAdmin` on the bucket.

---

## 4. Testing

A full pytest suite ships with the project. It runs in **~7 seconds** because the real 1.5 GB model is replaced by a deterministic fake — no network, no GPU, no HuggingFace download required.

```bash
pip install -r requirements-dev.txt
make test                # full suite
make test-cov            # with coverage report
make test-real           # opt-in: hits the real openai/privacy-filter (~1.5 GB)
```

What's covered (62 tests, 89% line coverage):

| Area                           | Tests |
|--------------------------------|-------|
| Format dispatcher              | 15    |
| `.txt` extractor + redactor    | 6     |
| `.docx` extractor + redactor   | 4     |
| `.pdf` (PyMuPDF) redaction     | 4     |
| Image OCR redaction            | 2     |
| DICOM tag scrubbing (VR-aware) | 5     |
| Storage (local + mocked GCS)   | 7     |
| Model chunking + normalisation | 9     |
| FastAPI endpoints (TestClient) | 10    |
| Real-model smoke (opt-in)      | 1     |

Markers: `slow`, `requires_model`, `requires_tesseract`, `requires_pymupdf`. Deselect heavy tests with `pytest -m "not slow"`.

---

## 5. API

| Method | Path                          | Description                              |
|--------|-------------------------------|------------------------------------------|
| GET    | `/`                           | Web UI                                   |
| GET    | `/api/health`                 | Liveness + model status                  |
| GET    | `/api/supported-types`        | Accepted file extensions                 |
| POST   | `/api/redact`                 | `multipart/form-data` field `file`       |
| GET    | `/api/files/{kind}/{key}`     | Download (`kind` ∈ `uploads`, `redacted`) |

`POST /api/redact` returns:
```json
{
  "job_id": "ab12cd34ef56",
  "filename": "memo.pdf",
  "content_type": "application/pdf",
  "entities": [
    {"entity_group": "private_person", "score": 0.999, "word": "Harry Potter", "start": 11, "end": 23}
  ],
  "entity_counts": {"private_person": 1, "private_email": 1},
  "original_url": "/api/files/uploads/ab12cd34ef56__memo.pdf",
  "redacted_url": "/api/files/redacted/ab12cd34ef56__redacted.pdf",
  "text_preview_original": "…",
  "text_preview_redacted": "…"
}
```

---

## 6. How redaction works per format

| Format        | Method                                                                                 |
|---------------|----------------------------------------------------------------------------------------|
| `.txt/.md/.csv` | Char-offset slice, replace span with `[REDACTED:LABEL]`                              |
| `.pdf`        | PyMuPDF `add_redact_annot` → black box + label, original text deleted, file flattened   |
| `.docx`       | Paragraph-level rewrite, span replaced with `[REDACTED:LABEL]` (preserves structure)    |
| Images        | Tesseract OCR with bounding boxes → black rectangles drawn over PII words               |
| `.dcm`        | PII DICOM tags overwritten with `REDACTED`; `PatientIdentityRemoved=YES` flag set       |

DICOM v1 does **not** OCR burned-in pixel annotations. Add a pixel-array OCR pass if you have scanners that burn patient name into the image.

---

## 7. Project structure

```
privacy-filter-app/
├── app/
│   ├── main.py            # FastAPI routes + lifespan
│   ├── model.py           # Privacy filter singleton
│   ├── redactor.py        # Format dispatcher
│   ├── schemas.py         # Pydantic models
│   ├── storage.py         # Local + GCS backends
│   └── extractors/        # Per-format extract+redact
├── frontend/              # index.html / app.js / styles.css
├── data/                  # uploads/ + redacted/  (gitignored content)
├── tests/
│   ├── conftest.py        # FakePrivacyFilter + fixtures
│   ├── unit/              # extractor + storage + model tests
│   └── integration/       # FastAPI TestClient + real-model smoke
├── scripts/deploy.sh
├── Dockerfile
├── cloudbuild.yaml
├── environment.yml        # conda env: privacy-filter
├── requirements.txt
├── requirements-dev.txt   # adds pytest, httpx, coverage
├── pytest.ini
├── Makefile
└── .env.example
```

---

## 8. Notes & next steps

- **Model size / cold start:** weights are baked into the Docker image. On Cloud Run with min-instances=0 expect ~5–10 s cold start; set `min-instances=1` if latency matters.
- **GPU:** Cloud Run now supports L4 GPUs. Replace CPU torch with `torch+cu121` and add `--gpu 1 --gpu-type nvidia-l4` to `deploy.sh` if needed (overkill for 50 M active params, but useful for batch).
- **Auth:** the deploy script enables `--allow-unauthenticated`. For internal use, drop that flag and front it with IAP or API Gateway.
- **Audit logging:** `data/uploads` retains originals — for compliance, swap to GCS with object versioning + retention.
- **DPDP / GDPR:** when running the service against real PHI, log only entity counts (already done) — never the matched text.
