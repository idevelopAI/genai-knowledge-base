# End-to-End GenAI Knowledge Base (RAG)

[![CI](https://github.com/idevelopAI/genai-knowledge-base/actions/workflows/ci.yml/badge.svg)](https://github.com/idevelopAI/genai-knowledge-base/actions/workflows/ci.yml)

A containerized Retrieval-Augmented Generation system for asking questions over
internal PDF documents. The project combines a FastAPI backend, LangChain,
ChromaDB, Google Gemini models, and a Streamlit chat UI.

The system covers document ingestion, vector search, grounded answer
generation, API access, Docker-based packaging, CI checks, and an Azure
Terraform infrastructure draft.

The Streamlit UI and sample questions use German because the included sample
documents and demo workflow are German-focused.

## Why This Project Matters

Companies often have useful information locked inside PDFs, manuals, policies,
and internal guides. This project turns those documents into a searchable
knowledge base where users can ask natural language questions and receive
context-grounded answers with source references.

## Architecture

```mermaid
flowchart LR
    PDFs["PDF documents"] --> Loader["LangChain PDF loader"]
    Loader --> Splitter["Recursive text splitter"]
    Splitter --> Embeddings["Gemini embeddings"]
    Embeddings --> Chroma["ChromaDB vector store"]

    User["User in Streamlit UI"] --> API["FastAPI backend"]
    API --> Retriever["Retriever top-k chunks"]
    Retriever --> Chroma
    Retriever --> Prompt["Grounded prompt with context"]
    Prompt --> LLM["Gemini chat model"]
    LLM --> Answer["Answer with source citations"]
    Answer --> User

    Upload["PDF upload / reindex"] --> API
    API --> Loader
```

## Screenshots

Chat answers include expandable source citations:

![Streamlit chat answer with source citations](docs/screenshots/chat-with-sources.png)

The sidebar supports document discovery, PDF upload, and manual reindexing:

![Streamlit sidebar upload and reindex controls](docs/screenshots/upload-reindex-sidebar.png)

## Tech Stack

- Backend: Python, FastAPI, Pydantic
- RAG: LangChain, ChromaDB, Google Gemini
- Frontend: Streamlit
- DevOps: Docker, GitHub Actions
- Infrastructure: Terraform for an Azure deployment draft

## Tradeoffs

- ChromaDB keeps the prototype simple: it runs locally, persists embeddings on
  disk, and avoids external database setup while still showing real vector
  retrieval behavior.
- Google Gemini is used for both embeddings and answer generation to keep the
  model stack consistent and easy to configure through environment variables.
- In production, the vector store should move to managed infrastructure with
  backups, access controls, monitoring, and a clear migration path for schema or
  embedding-model changes.
- Local vector storage is convenient for demos, but it does not solve multi-user
  concurrency, centralized persistence, auditability, or horizontal scaling.
- The built-in rate limiter is process-local. For production with multiple app
  instances, use shared infrastructure such as Redis, API Management, or an
  ingress-level rate limiter.

## Features

- Ingests PDFs from the local `data/` folder.
- Supports PDF upload and reindexing from the Streamlit UI.
- Validates PDF structure and enforces configurable file-size and page limits.
- Stores document chunks in a local ChromaDB vector database.
- Rebuilds indexes through versioned collections so a failed build does not
  replace the active index.
- Exposes a FastAPI `/ask` endpoint for RAG answers.
- Returns source document names, page numbers, and excerpts with each answer.
- Provides a Streamlit chat interface for interactive use.
- Includes a lightweight RAG evaluation runner.
- Includes Docker packaging and a GitHub Actions CI workflow.
- Supports optional bearer-token API protection with `RAG_API_TOKEN`.
- Includes configurable in-memory rate limiting for expensive API routes.
- Adds `X-Request-ID` headers for request tracing across API responses and
  logs.
- Logs request timing, retrieval timing, LLM timing, and indexing metrics.
- Keeps secrets out of tracked code and Docker build context.

## Current Scope

This repository provides a complete local RAG workflow. It is not yet intended
for production workloads that require distributed storage, horizontal scaling,
or managed cloud operations.

Current hardening work includes:

- API health check.
- Safe API error responses.
- Configurable model, vector store, and retrieval settings.
- Basic test coverage and CI validation.
- Source-cited RAG responses.
- PDF upload and reindex workflow.
- Basic JSONL-driven RAG eval checks.
- Optional API token protection.
- Configurable in-memory rate limiting.
- Structured observability logs for RAG and indexing paths.
- Request IDs for correlating client errors, server logs, and API responses.
- Transactional index activation and upload rollback on indexing failure.
- Bounded, parser-validated PDF uploads and concurrent rebuild protection.

Planned improvements:

- Add more RAG evaluation cases.
- Complete Azure container deployment settings.
- Replace in-memory rate limiting with shared infrastructure for multi-instance
  deployments.

## Local Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

`requirements.txt` lists the direct project dependencies. `requirements.lock.txt`
pins the full resolved dependency set used by CI and Docker for reproducibility.

To refresh the lock file after dependency changes:

```bash
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

Create a local `.env` file from the safe example file, then add your real API
key only in your local `.env` file:

```bash
cp example.env .env
```

Run ingestion to build the local vector database:

```bash
python ingest.py
```

Load the local environment variables and start the backend:

```bash
set -a
source .env
set +a
uvicorn main:app --reload
```

Start the frontend in a second terminal:

```bash
streamlit run frontend.py
```

The Streamlit sidebar lets you upload a PDF, rebuild the index, and see the
documents currently available to the backend.

PDF uploads default to a 10 MiB file limit and a 200-page limit. Override them
with `RAG_MAX_UPLOAD_BYTES` and `RAG_MAX_PDF_PAGES`.

## Demo Script

After starting the app, try this flow:

1. Open `http://localhost:8501`.
2. Confirm the sidebar lists the PDFs from `data/`.
3. Ask this German sample question: `Welche Informationen stehen im Dokument?`
4. Expand the German `Quellen` section under the answer and check the source
   filename/page.
5. Upload another PDF from the sidebar, click the German
   `Hochladen und indexieren` button, then ask a question about the new
   document.
6. Run `python evals/run_eval.py` while the backend is running to see the basic
   RAG quality checks.

## Docker

Run both the backend and frontend with Docker Compose:

```bash
docker compose up --build
```

Compose passes the complete local environment only to the backend. The
frontend receives the backend URLs and optional `RAG_API_TOKEN`, but not the
Gemini API key.

Then open:

```text
http://localhost:8501
```

Build only the backend image:

```bash
docker build -t rag-backend:latest .
```

Run the backend with local environment variables:

```bash
docker run --rm --env-file .env -p 8000:8000 rag-backend:latest
```

Then start the Streamlit frontend locally:

```bash
streamlit run frontend.py
```

## Evaluation

Run the app first, then execute the lightweight eval suite:

```bash
python evals/run_eval.py
```

The eval runner reads `evals/questions.jsonl`, sends each question to `/ask`,
and checks simple expectations such as source presence, expected source names,
and refusal behavior.

## API

If `RAG_API_TOKEN` is set, protected endpoints require:

```text
Authorization: Bearer <RAG_API_TOKEN>
```

The expensive endpoints are rate-limited by default using
`RAG_RATE_LIMIT_REQUESTS=60` per `RAG_RATE_LIMIT_WINDOW_SECONDS=60`. Set
`RAG_RATE_LIMIT_REQUESTS=0` to disable this locally.

Every API response includes an `X-Request-ID` header. Clients can send their own
`X-Request-ID` value, or the API will generate one automatically. The same value
is included in structured logs to make debugging easier.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask a German sample question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Welche Informationen stehen im Dokument?"}'
```

List available documents:

```bash
curl http://127.0.0.1:8000/documents
```

Upload and index a PDF:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@data/sample_document.pdf"
```

Rebuild the index from all PDFs in `data/`:

```bash
curl -X POST http://127.0.0.1:8000/reindex
```

Example response shape:

```json
{
  "answer": "Antwort basierend auf dem gefundenen Kontext.",
  "sources": [
    {
      "source": "sample_document.pdf",
      "page": 1,
      "excerpt": "Relevant retrieved text..."
    }
  ]
}
```

## Security

- Real secrets must stay in local environment variables or `.env`.
- `.env` and `.env.*` are ignored by Git and Docker.
- `example.env` contains placeholder values only.
- Set `RAG_API_TOKEN` to require bearer-token access for `/ask`, `/documents`,
  and `/reindex`.
- `RAG_RATE_LIMIT_REQUESTS` and `RAG_RATE_LIMIT_WINDOW_SECONDS` limit repeated
  calls to `/ask`, `/documents`, and `/reindex`.
- `RAG_MAX_UPLOAD_BYTES` and `RAG_MAX_PDF_PAGES` bound uploaded PDF processing.
- Uploaded files are parser-validated and removed if indexing fails.
- Index rebuilds activate a new version only after embedding succeeds; the
  previous collection remains active on failure.
- API errors avoid returning raw internal exception text to users.
- Bearer tokens are compared in constant time and represented only by a hashed
  fingerprint in rate-limiter state and logs.

If a real API key is ever committed, rotate the key immediately and remove it
from Git history.

## Observability

The API logs structured events for the important paths:

- `rag_question_answered`: question length, source count, retrieval time, LLM
  time, total request time, and request ID.
- `rag_index_rebuilt`: PDF count, loaded pages, generated chunks, and indexing
  time, plus request ID when triggered through the API.
- `rag_document_uploaded`: saved filename, upload size, page count, resulting
  PDF count, chunk count, total upload/index time, and request ID.
- `rag_rate_limited`: client key, configured limit, window size, and retry delay
  when a request receives HTTP 429, plus request ID.
- `http_request_completed`: method, path, status code, elapsed time, and request
  ID for every API request.

Use `LOG_LEVEL=INFO` locally or in deployment to keep these metrics visible.

## CI

GitHub Actions validates the project by installing dependencies, running tests,
and building the Docker image.

## Dependency Management

- `requirements.txt` contains direct dependencies that describe what the project
  intentionally uses.
- `requirements.lock.txt` pins the full resolved environment used by CI and
  Docker.
- After changing direct dependencies, regenerate the lock file with
  `pip freeze > requirements.lock.txt` from a freshly installed environment.

## Terraform

The `terraform/` folder contains an Azure infrastructure draft for the future
deployment path. Before production use, it should be extended with container
image settings, secret management, managed identity, and deployment outputs.
