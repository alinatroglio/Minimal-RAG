# Minimal RAG API (FastAPI + Chroma + llama.cpp)

A minimal Retrieval-Augmented Generation (RAG) service using [Haystack](https://haystack.deepset.ai/) with two endpoints: /ingest for adding documents and /query for asking questions retrieving those docs. It uses open-source embeddings and a local LLM (via [llama.cpp](https://github.com/ggml-org/llama.cpp)) so it runs offline using.

## Project Structure

```bash
Minimal-RAG/
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
├─ README.md
├─ main.py                     # FastAPI app (ingest + query endpoints)
├─ config.py                   # Model paths & prompt template
├─ vector_database.py          # Chroma-based vector store (persistent)
├─ index_pipeline.py           # Document splitter + embedder + writer
├─ rag_pipeline.py             # Query embedder + retriever + prompt + LLM + answer
├─ chroma_db/                  # (persisted Chroma data; created at runtime)
└─ model/
   └─ openchat-3.5-1210.Q3_K_S.gguf   
```

## How it works
### 1. Ingest
- POST a text file to `\ingest`
- The pipeline splits the text into chunks, embeds them with [SentenceTransformers](https://www.sbert.net/), and stores vectors + metadata in [Chroma](https://www.trychroma.com/).

### 2. Query
- POST a query string to `\query`
- The query is embedded and Chroma returns the most similar chunks.
- A prompt is built from those chunks and sent to local llama.cpp model.
- The answer is returned considering the most relevant documents.

### 3. Build and Run
#### Prerequisites
Before running the project, make sure you have the following installed:

- [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) (or Docker Engine + Docker Compose if you’re on Linux)
- [Git](https://git-scm.com/downloads/win) – to clone the repository

Once Docker Desktop is running, you can proceed in the repository folder with:
```bash
docker compose build
docker compose up
```

The compose file runs a single service: `rag_api`.
Chroma data persists under ./chroma_db.


### 4. Try it out
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 1. Ingest a document

Endpoint: POST /ingest

Upload a small UTF-8 text file.
The server will split, embed, and store it.

#### 2. Ask a question

Endpoint: POST /query

## Design Choices
- Chroma is simple, lightweight vector DB with persistence. Easy to use for minimal RAG without managing external services.
- SentenceTransformers embeddings (Qwen/Qwen3-Embedding-0.6B) is an open, high-quality embeddings, usable without API keys.
- llama.cpp generator as model, since it is a full local LLM that works on CPU

## Known limitations
- **No automated tests** unit/integration yet
- **Minimal error handling** file size/encoding checks only
- **Single-process** though Chroma embedding
