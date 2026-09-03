# SmartAngio Reference Checker

Citation-verification tool on smartangio.com. Uploads a PDF, extracts its
cited sentences and reference list, retrieves the referenced articles from
legal open-access sources, and checks each claim against the source text
using an LLM.

## Layout

- `app/main.py` — FastAPI application (upload queue, status/stats, downloads).
- `app/reference_checker.py` — extraction, open-access retrieval, verification.
- `app/database.py` — SQLAlchemy request store.
- `app/static/index.html` — single-file frontend (nginx alias target).
- `requirements.txt` — pinned dependencies.

## Live deployment (voteladder-server)

- Path: `/var/www/html/website/reference_checker/reference-checker`
- Service: `reference-checker.service` (uvicorn on 127.0.0.1:8001)
- Nginx: `/reference_checker/` → `app/static/` alias; `/reference_checker/api/`
  → proxy to :8001
- Secrets live in `.env` at the app root (not committed). Never run without
  it: model and API keys are read from environment variables.

## Open-access retrieval order

Per reference, the app resolves a DOI (CrossRef fallback) and tries, in order:

1. Unpaywall — all open-access PDF locations
2. OpenAlex — open-access `pdf_url` locations
3. Europe PMC — DOI → PMCID resolution for confirmed open-access records
4. Europe PMC full-text XML — used when no open PDF is retrievable

Strict timeouts (8 s metadata, 20 s PDFs, one attempt, max 4 candidates per
reference) keep failed lookups cheap. Publisher paywalls and PMC
proof-of-work gates are never bypassed. Successful sources are recorded per
reference as `download_source` / `download_format` (`pdf` or `fulltext_xml`).

## Running locally

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
# create .env from the live server's .env (MAIN_*/VERIF_* keys/URLs/models)
venv/bin/uvicorn app.main:app --port 8001
```
