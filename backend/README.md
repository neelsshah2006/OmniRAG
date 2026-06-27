# OmniRAG — Backend

**Deep-Dive Technical Reference**

> Python 3.12 · FastAPI · Qdrant · Groq · MinIO · Redis · Unstructured

This directory contains the complete OmniRAG backend: a single async Python service built on FastAPI that exposes a document ingestion pipeline and a RAG (Retrieval-Augmented Generation) query interface.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Folder Structure](#2-folder-structure)
3. [Startup Sequence](#3-startup-sequence)
4. [Request Flow](#4-request-flow)
   - [Ingestion Pipeline](#ingestion-pipeline)
   - [Retrieval Pipeline](#retrieval-pipeline)
5. [Core Modules](#5-core-modules)
6. [Tech Stack](#6-technology-stack)
7. [Installation](#7-installation)
8. [Environment Variables](#8-environment-variables)
9. [Running](#9-running)
10. [API Reference](#10-api-reference)
11. [Code Organisation](#11-code-organisation)
12. [Logging](#12-logging)
13. [Error Handling](#13-error-handling)
14. [Background Jobs](#14-background-jobs)

---

## 1. Overview

### Purpose

The backend is responsible for:

1. Accepting raw document uploads (PDF, DOCX, PPTX, TXT, Markdown)
2. Parsing, enriching, chunking, and indexing document content into a hybrid vector store
3. Answering natural language questions by retrieving relevant chunks and generating grounded answers via an LLM

### Responsibilities

- Document lifecycle management (upload → process → ready / failed)
- File storage via MinIO object storage
- Document parsing via the Unstructured library
- Multimodal enrichment: image description (vision LLM) and table summarisation (text LLM)
- Dense and sparse embedding generation
- Qdrant vector indexing (cosine + BM25)
- Dense and hybrid retrieval with cross-encoder reranking
- Query rewriting before retrieval
- LLM-powered answer generation with cited sources
- Redis caching for dense embeddings

---

## 2. Folder Structure

```
backend/
├── app/
│   ├── ai/                     # All AI model abstractions
│   │   ├── embeddings/         # Dense embedding service + SentenceTransformers
│   │   ├── llm/                # LLM abstraction layer + Groq provider
│   │   ├── reranker/           # Cross-encoder reranking service
│   │   ├── sparse/             # Sparse (BM25) embedding service + FastEmbed
│   │   ├── vision/             # Vision LLM service + Groq vision provider
│   │   └── prompts/            # All prompt templates (RAG, query, vision, table)
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── documents.py    # Upload endpoint
│   │       │   └── query.py        # Query endpoint
│   │       └── router.py           # API router assembly
│   ├── cache/                  # Redis-backed caching
│   │   ├── base.py             # Generic CacheService wrapper
│   │   ├── embedding.py        # Embedding-specific cache
│   │   ├── keys.py             # Cache key generation
│   │   └── constants.py        # TTL constants
│   ├── core/                   # Application-wide infrastructure
│   │   ├── config.py           # Pydantic settings (all env vars)
│   │   ├── exceptions.py       # OmniRAGException + FastAPI handler
│   │   ├── logging.py          # Loguru setup
│   │   └── redis.py            # Redis connection manager
│   ├── middleware/
│   │   └── request.py          # Request ID + latency logging middleware
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── document.py         # Document model + DocumentStatus enum
│   │   └── document_event.py   # DocumentEvent model + event type enum
│   ├── rag/                    # Core RAG pipeline
│   │   ├── chain/              # End-to-end RAG chain
│   │   ├── chunking/           # Chunking strategies
│   │   ├── ingestion/          # Ingestion pipeline + internal domain models
│   │   ├── loaders/            # Document format loaders
│   │   ├── processors/         # Pre-chunking enrichment processors
│   │   ├── query/              # Query preprocessing (rewriting)
│   │   └── retrieval/          # Retrieval pipeline, retrievers, context builder
│   ├── repositories/           # Database access layer
│   │   ├── document_repository.py
│   │   └── document_event_repository.py
│   ├── schemas/                # Pydantic API schemas (request/response)
│   │   ├── document.py
│   │   └── query.py
│   ├── services/               # Business logic layer
│   │   └── document_service.py
│   ├── storage/
│   │   ├── database/           # SQLAlchemy engine, session, schema init
│   │   ├── object/             # MinIO object storage
│   │   └── vector/             # Qdrant vector store
│   ├── tasks/
│   │   └── document.py         # Background task entry point
│   ├── utils/
│   │   ├── batch.py            # Iterable batching utility
│   │   ├── concurrency.py      # Semaphore-bounded asyncio.gather
│   │   ├── hash.py             # SHA-256 streaming hash
│   │   ├── retry.py            # Exponential backoff with jitter
│   │   └── time.py             # UTC datetime helper
│   └── main.py                 # FastAPI app, lifespan hooks, middleware
├── pyproject.toml
├── Dockerfile
└── .env.example
```

---

## 3. Startup Sequence

The `lifespan` async context manager in `app/main.py` executes on every server start:

| Step | Action                                                                             | Failure                              |
| ---- | ---------------------------------------------------------------------------------- | ------------------------------------ |
| 1    | `redis_manager.connect()` — ping                                                   | `RedisError` → server fails to start |
| 2    | `init_object_storage()` — MinIO connect, create bucket if absent                   | Connection error                     |
| 3    | `init_database()` — SQLAlchemy `create_all()`                                      | Bad `DATABASE_URL`                   |
| 4    | `init_embedding_service()` — load SentenceTransformer                              | Missing / inaccessible model         |
| 5    | `sparse_embedding_service.initialize()` — load BM25                                | Missing model                        |
| 6    | `init_vector_service(dimension=384)` — connect Qdrant, create collection if absent | Qdrant unreachable                   |
| 7    | `retrieval_service.initialize()` — select retriever from `RETRIEVAL_MODE`          | `ValueError` on unknown mode         |
| 8    | `reranker_service.initialize()` — load CrossEncoder                                | Missing model                        |
| 9    | `llm_service.initialize()` — create `AsyncGroq` client                             | Missing `GROQ_API_KEY`               |
| 10   | `vision_service.initialize()` — create Groq vision client                          | Missing `GROQ_API_KEY`               |

Shutdown (reverse order): close Qdrant → close Redis.

---

## 4. Request Flow

### Ingestion Pipeline

#### Upload

```
Client
  │  POST /api/v1/documents/upload
  ▼
documents.py (endpoint)
  │  SHA-256 dedup check
  │  Create document record
  │  Upload file to MinIO
  ▼
BackgroundTasks.add_task(process_document_task)
  ▼
process_document_task  (background)
  ▼
DocumentService.process_document()
  │  Download file from MinIO → temp dir
  ▼
LoaderService.load()          ← parse to DocumentElements
  ▼
ProcessorPipeline.process()   ← ImageProcessor + TableProcessor (concurrent)
  ▼
IngestionPipeline.ingest()
  │  ChunkingService → select chunker
  │  Chunker.chunk()
  │  EmbeddingService.embed_batch() + SparseEmbeddingService.embed_batch()  (concurrent)
  ▼
QdrantVectorStore.add_documents()
  ▼
DocumentRepository.update_status(READY)
```

#### Step 1 — Upload Handler

`POST /api/v1/documents/upload` accepts multipart form data. Before any DB write, `sha256_stream()` hashes the raw file. `DocumentRepository.get_by_content_hash()` checks for duplicates — if found, returns immediately (idempotent). Otherwise calls `DocumentService.create_document()`.

#### Step 2 — `create_document()`

1. DB transaction: create `Document` row (status=`UPLOADED`) + emit `UPLOADED` event
2. Upload stream to MinIO at `documents/{id}/{filename}`
3. Update `document.file_path` in DB
4. On MinIO failure: delete partial upload, delete DB record, re-raise
5. Return `Document` (FastAPI serialises to `DocumentResponse`)
6. `BackgroundTasks.add_task(process_document_task, document.id)` — returns HTTP response immediately

#### Step 3 — `process_document()` (background)

Runs in a separate `AsyncSession` scoped to the background task.

1. Update status → `PROCESSING`, emit `PROCESSING_STARTED` + `PARSING_STARTED`
2. Download file from MinIO to `TemporaryDirectory`
3. `loader_service.load(path)` — selects loader by extension
4. `UnstructuredLoader`: `partition(strategy="hi_res", infer_table_structure=True, extract_image_block_to_payload=True)` — produces structured elements with table HTML and base64 image payloads
5. `processor_pipeline.process(document)`:
   - `ImageProcessor`: calls Groq Vision for each image element (semaphore-limited), appends `[Image Analysis]` block to content
   - `TableProcessor`: calls Groq LLM for each table element (semaphore-limited), appends `[Table Analysis]` block to content
6. `chunking_service.get_chunker(document)`: `UnstructuredTitleChunker` for Unstructured-parsed docs, `RecursiveChunker` otherwise
7. Filter empty chunks
8. `asyncio.gather(embedding_service.embed_batch(texts), sparse_service.embed_batch(texts))` — dense + sparse in parallel
9. Build `VectorDocument` list with payload: `{text, document_id, chunk_index, element_ids, content_length, ...metadata}`
10. `vector_service.add_documents()` → batched upsert to Qdrant
11. Update `element_count`, `chunk_count`, status → `READY`, emit `COMPLETED`
12. On any exception: rollback → `FAILED` status → `FAILED` event

---

### Retrieval Pipeline

#### Query

```
Client
  │  POST /api/v1/query/
  ▼
query.py (endpoint)
  ▼
RAGChain.ask()
  ▼
RetrievalPipeline.retrieve()
  │  QueryService.process()          ← LLM query rewriting
  │  RetrievalService.retrieve()     ← dense or hybrid search
  │    EmbeddingService.embed_text() ← Redis cache hit/miss
  │    [SparseEmbeddingService.embed()]   ← hybrid only
  │    QdrantVectorStore.search()
  │  RerankerService.rerank()        ← CrossEncoder scoring
  │  select top-K chunks
  ▼
ContextBuilder.build()             ← format chunks into LLM context
  ▼
LLMService.generate()              ← Groq chat completion
  ▼
QueryResponse (answer + sources + usage + latency)
```

#### `RAGChain.ask(question)`

Entry point for all queries.

```
question
  → RetrievalPipeline.retrieve(question)
  → ContextBuilder.build(result)
  → LLMService.generate([system_prompt, user_prompt], temperature=0.2)
  → RAGResponse
```

#### `RetrievalPipeline.retrieve(query)`

```
query
  → QueryService.process()            # LLM query rewrite at temperature=0
  → RetrievalService.retrieve()       # dense OR hybrid
      DenseRetriever:
        embed_text(query) → Qdrant cosine search (top-20)
      HybridRetriever:
        gather(embed_text, sparse_embed) → Qdrant RRF (top-20)
  → RerankerService.rerank()          # CrossEncoder.predict for all pairs
      filter: score >= MIN_RELEVANCE_SCORE
      select: top RERANK_TOP_K
  → attach rerank_score to chunks
  → return RetrievalResult
```

#### Context Format

Each source chunk is formatted as:

```
[File: report.pdf
Metadata: {"filename": "report.pdf", "page_number": 3, ...}]

Vector Score:
0.847

Rerank Score:
0.912

Content:
... chunk text here ...
```

---

## 5. Core Modules

### API Layer (`app/api/`)

Two route groups mounted under `/api/v1`:

- **`/documents/upload`** (`POST`) — multipart file upload with background ingestion trigger
- **`/query/`** (`POST`) — question answering against indexed content
- **`/health`** (`GET`) — service liveness check

The router is assembled in `app/api/v1/router.py` and registered in `main.py` with the `/api/v1` prefix.

### Configuration (`app/core/config.py`)

`Settings` is a `pydantic-settings` `BaseSettings` subclass. All configuration is loaded from environment variables (or `.env`) with typed defaults. A cached singleton is provided via `get_settings()` (LRU-cached).

`REDIS_URL` is a computed field derived from host/port/db components.

### Logging (`app/core/logging.py`)

Loguru is configured with two sinks:

- **stdout** — colourised, `INFO` level, human-readable format
- **`logs/omnirag.log`** — JSON serialised, `DEBUG` level, 10 MB rotation, 7-day retention, gzip compression

### Exception Handling (`app/core/exceptions.py`)

`OmniRAGException` carries a `message`, HTTP `status_code`, and application-level `error_code`. A FastAPI exception handler serialises it to:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```

### Middleware (`app/middleware/request.py`)

`RequestMiddleware` (Starlette `BaseHTTPMiddleware`) assigns a UUID `request_id` to every request and appends:

- `X-Request-ID` response header
- `X-Process-Time` response header (seconds, 4 decimal places)

Both the incoming request and completed response are logged with method, path, status code, duration, and request ID.

### Storage — Relational Database (`app/storage/database/`)

SQLAlchemy 2 async engine with `aiosqlite` (SQLite default) or `asyncpg` (PostgreSQL). Schema is created on startup via `Base.metadata.create_all`. `SessionLocal` is an `async_sessionmaker`; sessions are yielded per request via `get_session()`.

Two ORM models:

**`Document`** — tracks a file from upload through ingestion:

| Column          | Type        | Notes                                              |
| --------------- | ----------- | -------------------------------------------------- |
| `id`            | UUID string | Primary key                                        |
| `filename`      | String      | Original file name                                 |
| `file_path`     | String      | Object storage key                                 |
| `file_size`     | Integer     | Bytes                                              |
| `content_type`  | String      | MIME type                                          |
| `content_hash`  | String(64)  | SHA-256; unique index for dedup                    |
| `status`        | Enum        | `uploaded · processing · ready · failed · deleted` |
| `element_count` | Integer     | Parsed elements                                    |
| `chunk_count`   | Integer     | Indexed vectors                                    |
| `error_message` | String      | Set on failure                                     |

**`DocumentEvent`** — immutable audit log:

| Event type                                  | Meaning               |
| ------------------------------------------- | --------------------- |
| `uploaded`                                  | File received         |
| `processing_started`                        | Background task began |
| `parsing_started / completed`               | Loader finished       |
| `enrichment_started / completed`            | Processors finished   |
| `chunking / embedding / indexing completed` | Pipeline stages       |
| `completed`                                 | Document ready        |
| `failed`                                    | Terminal error        |

### Storage — Object Storage (`app/storage/object/`)

`MinIOStorage` wraps the synchronous `minio` SDK. Key operations:

- `upload_stream()` — uploads a raw IO stream with content type
- `upload_path()` — reads and uploads an existing file path
- `download()` — downloads to a destination `Path`
- `delete()` — removes an object by key

The bucket is created automatically on startup if it does not exist. Document files are stored under the key pattern `documents/{document_id}/{filename}`.

### Storage — Vector Database (`app/storage/vector/`)

`QdrantVectorStore` uses the async Qdrant client. The collection is created on startup with:

- A **named dense vector** (`"dense"`, cosine distance, configurable dimension)
- A **named sparse vector** (`"sparse"`, BM25 indices/values)

Key operations:

- `upsert()` — batched upsert (configurable batch size, default 256) with timing log
- `search()` — dense-only ANN search
- `hybrid_search()` — prefetch dense + sparse, fuse with Reciprocal Rank Fusion
- `delete()` — remove points by ID list

### RAG — Document Loaders (`app/rag/loaders/`)

`LoaderService` dispatches by file extension:

| Extension                | Loader                                                  |
| ------------------------ | ------------------------------------------------------- |
| `.txt`, `.md`            | `TextLoader` — reads raw text, creates a single element |
| `.pdf`, `.docx`, `.pptx` | `UnstructuredLoader`                                    |

`UnstructuredLoader` calls `unstructured.partition.auto.partition` with `strategy="hi_res"`, table structure inference enabled, and image blocks extracted as base64 into element metadata. Files larger than 100 MB are rejected.

### RAG — Document Processors (`app/rag/processors/`)

`ProcessorPipeline` runs all processors concurrently via `asyncio.gather`:

**`ImageProcessor`** — finds elements with a `image_base64` metadata key, sends each image to the vision LLM via `vision_service.describe_image()` (rate-limited by `MAX_CONCURRENT_AI_REQUESTS` semaphore), and appends the description to the element's text content under an `[Image Analysis]` header.

**`TableProcessor`** — finds elements with `text_as_html` metadata, sends the table text to the LLM for summarisation (rate-limited), and appends the summary under a `[Table Analysis]` header.

Both processors modify elements in place and return the updated `Document`.

### RAG — Chunking (`app/rag/chunking/`)

`ChunkingService` selects the strategy based on whether any element carries a `source_element` (i.e., was parsed by Unstructured):

**`UnstructuredTitleChunker`** — calls `unstructured.chunking.title.chunk_by_title` with overlap and size controls. Preserves structural boundaries (headings, table isolation, multipage sections). Chunk IDs are deterministic UUID5s derived from `document_id + index + text`.

**`RecursiveChunker`** — uses `langchain_text_splitters.RecursiveCharacterTextSplitter` (default 1000 chars, 200 overlap). Joins all element texts, then splits. Chunk IDs use the same UUID5 scheme.

Both produce `DocumentChunk` records carrying the chunk text, position index, parent element IDs, and metadata including `chunk_strategy`.

### RAG — Ingestion Pipeline (`app/rag/ingestion/pipeline.py`)

Coordinates the full ingestion flow after loading:

1. Run `ProcessorPipeline.process()` (image + table enrichment)
2. Select chunker, produce `DocumentChunk` list
3. Filter out empty chunks
4. Generate dense and sparse embeddings concurrently (`asyncio.gather`)
5. Validate that both embedding services return the same count as the chunk list
6. Build `VectorDocument` list with dense vector, sparse vector, and payload (text, document ID, chunk index, element IDs, metadata)
7. Upsert to Qdrant in batches

### RAG — Query Processing (`app/rag/query/`)

`QueryService.process()` calls `QueryRewriter.rewrite()`, which sends the user's question to the Groq LLM with the `QUERY_REWRITE_PROMPT` system prompt (temperature 0). The rewritten query is used for all downstream retrieval. If the LLM returns the original unchanged, no rewrite occurred.

### RAG — Retrieval (`app/rag/retrieval/`)

**`RetrievalService`** initialises either `DenseRetriever` or `HybridRetriever` based on `RETRIEVAL_MODE`.

**`DenseRetriever`**: embeds the query → Qdrant dense ANN search (top-K).

**`HybridRetriever`**: embeds dense + sparse concurrently → Qdrant prefetch both, fuse with RRF → single ranked result set.

**`RetrievalPipeline`** wraps retrieval with reranking:

1. Retrieve candidates (up to `RETRIEVAL_TOP_K`)
2. Rerank all candidates with `CrossEncoderReranker`
3. Select top `RERANK_TOP_K` by rerank score
4. Attach `rerank_score` to each chunk

**`ContextBuilder`** formats the selected chunks into a single string containing file name, metadata, vector score, rerank score, and content per chunk.

### RAG — Chain (`app/rag/chain/service.py`)

`RAGChain.ask()` is the single entry point for query answering:

1. Retrieve and rerank via `RetrievalPipeline`
2. Build context string via `ContextBuilder`
3. Construct system + user messages using `RAG_SYSTEM_PROMPT` and `build_rag_user_prompt()`
4. Generate answer via `LLMService` (temperature 0.2)
5. Return `RAGResponse` with answer, source references (with scores), model name, token usage, and total latency

### AI Services (`app/ai/`)

All AI services follow the same pattern: an abstract base class (`Base`), a concrete provider, and a service wrapper that handles caching, batching, or concurrency controls.

| Service                  | Provider                      | Model                                     |
| ------------------------ | ----------------------------- | ----------------------------------------- |
| `EmbeddingService`       | `SentenceTransformerProvider` | BAAI/bge-small-en-v1.5 (configurable)     |
| `SparseEmbeddingService` | `FastEmbedProvider`           | Qdrant/bm25                               |
| `LLMService`             | `GroqProvider`                | llama-3.1-8b-instant (configurable)       |
| `VisionService`          | `GroqVisionProvider`          | meta-llama/llama-4-scout-17b-16e-instruct |
| `RerankerService`        | `CrossEncoderReranker`        | cross-encoder/ms-marco-MiniLM-L-6-v2      |

`EmbeddingService` checks the Redis embedding cache before calling the provider for both single and batch requests. Cache misses are written back after the provider responds.

`GroqProvider` and `GroqVisionProvider` both use `retry_async` with exponential backoff on `RateLimitError`, `InternalServerError`, `APIConnectionError`, and `APITimeoutError`.

### Prompts

#### RAG System Prompt (`prompts/rag.py`)

Enforces strict grounding:

- Answer only from provided context
- Never invent facts, numbers, names, dates, or statistics
- Surface conflicting information between sources
- Preserve technical terminology from source documents
- Prefer bullet points and step-by-step explanations for technical answers
- If context is insufficient: say so explicitly

#### Query Rewrite Prompt (`prompts/query.py`)

Instructs the LLM to:

- Resolve vague references and expand abbreviations
- Add related technical terms
- Preserve original meaning

Explicitly forbids: answering the question, adding assumptions, inventing entities.

#### Vision Analysis Prompt (`prompts/vision.py`)

Five-section extraction framework:

1. Visual content — what the image represents, purpose of diagrams/charts
2. Text extraction — all visible labels, captions, annotations, equations (exact)
3. Structural relationships — flows, hierarchies, dependencies
4. Data interpretation — axes/trends for charts, rows/columns for tables, components for architecture diagrams
5. Retrieval summary — concise semantic description for natural-language findability

#### Table Analysis Prompt (`prompts/table.py`)

Extracts: table purpose, column meanings, key values/metrics, inter-row trends and relationships. Explicitly preserves all numbers exactly as they appear.

### Cache (`app/cache/`)

`CacheService` is a thin Redis wrapper with JSON serialisation, namespace-prefixed keys, and configurable TTL. `EmbeddingCache` uses it with a per-model/text cache key generated by hashing model name and content.

### Repositories (`app/repositories/`)

`DocumentRepository` and `DocumentEventRepository` are the exclusive database access points. They accept an `AsyncSession` and expose only typed, documented query methods. No business logic lives here.

### Services (`app/services/`)

`DocumentService` orchestrates the upload and processing workflows. It delegates to the repository for persistence, to `ObjectStorage` for file IO, and to `IngestionPipeline` for RAG indexing. It never touches the database directly.

### Utilities (`app/utils/`)

| Utility             | Purpose                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `retry_async`       | Generic async exponential backoff with configurable retries, base delay, and exception types       |
| `gather_with_limit` | `asyncio.gather` with a semaphore to cap in-flight coroutines                                      |
| `batched`           | Splits an iterable into fixed-size chunks                                                          |
| `sha256_stream`     | Reads a binary stream and returns its SHA-256 hex digest without loading the full file into memory |

---

## 6. Technology Stack

| Layer              | Library / Service                             |
| ------------------ | --------------------------------------------- |
| API framework      | FastAPI + Uvicorn                             |
| Async runtime      | Python 3.12 asyncio                           |
| ORM                | SQLAlchemy 2 (async)                          |
| Relational DB      | SQLite via aiosqlite · PostgreSQL via asyncpg |
| Vector DB          | Qdrant (`qdrant-client`)                      |
| Object Storage     | MinIO (`minio`)                               |
| Cache              | Redis (`redis[asyncio]`)                      |
| LLM / Vision       | Groq (`groq`)                                 |
| Dense Embeddings   | SentenceTransformers                          |
| Sparse Embeddings  | FastEmbed                                     |
| Reranker           | SentenceTransformers CrossEncoder             |
| Document Parsing   | Unstructured (`unstructured[all-docs]`)       |
| Text Splitting     | LangChain Text Splitters                      |
| Validation         | Pydantic v2 + pydantic-settings               |
| Logging            | Loguru                                        |
| Package manager    | uv                                            |
| Linter / Formatter | Ruff                                          |
| Type checker       | mypy                                          |
| Testing            | pytest + pytest-asyncio + httpx               |

---

## 7. Installation

```bash
# From the repository root
make install

# Or directly
cd backend
uv sync
```

---

## 8. Environment Variables

| Variable                     | Default                                     | Required | Description                            |
| ---------------------------- | ------------------------------------------- | -------- | -------------------------------------- |
| `GROQ_API_KEY`               | —                                           | **YES**  | Groq API key                           |
| `GROQ_MODEL`                 | `llama-3.1-8b-instant`                      | no       | LLM for generation and query rewriting |
| `GROQ_VISION_MODEL`          | `meta-llama/llama-4-scout-17b-16e-instruct` | no       | Vision model for image captioning      |
| `MAX_CONCURRENT_AI_REQUESTS` | `5`                                         | no       | Semaphore limit for parallel AI calls  |
| `AI_MAX_RETRIES`             | `5`                                         | no       | Max retries on Groq transient errors   |
| `AI_RETRY_BASE_DELAY`        | `2`                                         | no       | Base seconds for exponential backoff   |
| `EMBEDDING_MODEL`            | `BAAI/bge-small-en-v1.5`                    | no       | SentenceTransformer model              |
| `EMBEDDING_DIMENSION`        | `384`                                       | no       | Must match model output dimension      |
| `EMBEDDING_BATCH_SIZE`       | `64`                                        | no       | Texts per embedding batch              |
| `SPARSE_EMBEDDING_MODEL`     | `Qdrant/BM25`                               | no       | FastEmbed sparse model                 |
| `RETRIEVAL_MODE`             | `dense`                                     | no       | `dense` or `hybrid`                    |
| `RETRIEVAL_TOP_K`            | `20`                                        | no       | Candidates before reranking            |
| `RERANKER_MODEL`             | `cross-encoder/ms-marco-MiniLM-L-6-v2`      | no       | CrossEncoder model                     |
| `RERANK_TOP_K`               | `5`                                         | no       | Chunks after reranking                 |
| `MIN_RELEVANCE_SCORE`        | `0.3`                                       | no       | Minimum reranker score                 |
| `QDRANT_HOST`                | `localhost`                                 | no       | Qdrant host                            |
| `QDRANT_PORT`                | `6333`                                      | no       | Qdrant REST port                       |
| `QDRANT_COLLECTION`          | `omnirag_chunks`                            | no       | Collection name                        |
| `VECTOR_UPSERT_BATCH_SIZE`   | `256`                                       | no       | Points per upsert batch                |
| `DATABASE_URL`               | `sqlite+aiosqlite:///./omnirag.db`          | no       | SQLAlchemy async connection string     |
| `REDIS_HOST`                 | `localhost`                                 | no       | Redis host                             |
| `REDIS_PORT`                 | `6379`                                      | no       | Redis port                             |
| `REDIS_DB`                   | `0`                                         | no       | Redis database index                   |
| `MINIO_ENDPOINT`             | `localhost:9000`                            | no       | MinIO endpoint (host:port)             |
| `MINIO_ACCESS_KEY`           | `minioadmin`                                | no       | MinIO access key                       |
| `MINIO_SECRET_KEY`           | `minioadmin`                                | no       | MinIO secret key                       |
| `MINIO_BUCKET`               | `documents`                                 | no       | MinIO bucket for uploads               |
| `MINIO_SECURE`               | `False`                                     | no       | Use HTTPS for MinIO                    |
| `LOG_LEVEL`                  | `INFO`                                      | no       | `DEBUG / INFO / WARNING / ERROR`       |
| `ENVIRONMENT`                | `development`                               | no       | `development / staging / production`   |

Copy `.env.example` to `.env` and populate the required values:

```bash
cp .env.example .env
```

The only required variable without a default is:

```dotenv
GROQ_API_KEY=your_key_here
```

See the full variable table in the [root README](../README.md#environment-variables).

---

## 9. Running

### Development (hot-reload)

```bash
# From repository root
make backend

# Or directly
cd backend
uv run uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Production

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set `ENVIRONMENT=production` and use a PostgreSQL `DATABASE_URL` for production deployments.

### Docker

```bash
# From repository root
docker compose build
docker run --env-file backend/.env -p 8000:8000 omnirag-backend
```

The Dockerfile uses `python:3.12-slim` and copies the `uv` binary from the official image for fast, reproducible installs.

---

## 10. API Reference

### `POST /api/v1/documents/upload`

Multipart form upload. Returns immediately; ingestion runs as a background task.

**Request:** `multipart/form-data`, field `file`

**Response (`200 OK`):**

```json
{
  "id": "uuid",
  "filename": "report.pdf",
  "status": "uploaded",
  "element_count": 0,
  "chunk_count": 0,
  "error_message": null,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

Uploading an identical file (same SHA-256) returns the existing document without re-processing.

### `POST /api/v1/query/`

**Request:**

```json
{ "question": "What are the main findings?" }
```

**Response (`200 OK`):**

```json
{
  "question": "What are the main findings?",
  "answer": "The report identifies three key findings...",
  "sources": [
    {
      "content": "chunk text...",
      "metadata": {
        "filename": "report.pdf",
        "chunk_strategy": "unstructured_title"
      },
      "vector_score": 0.87,
      "rerank_score": 4.21
    }
  ],
  "model": "llama-3.1-8b-instant",
  "usage": { "prompt_tokens": 512, "completion_tokens": 128 },
  "latency_ms": 1234.56
}
```

### `GET /health`

```json
{
  "status": "running",
  "environment": "development",
  "version": "v1",
  "services": { "redis": "healthy" }
}
```

---

## 11. Code Organisation

The backend follows a **Layered Architecture** with clear responsibility boundaries:

```
Endpoint (HTTP / schema validation)
    ↓
Service (business logic, workflow orchestration)
    ↓
Repository (database queries only)
    ↓
ORM Model (SQLAlchemy)
```

The RAG pipeline runs as a parallel hierarchy:

```
Task (background entry point)
    ↓
Service (coordinates pipeline + status updates)
    ↓
Pipeline (IngestionPipeline / RetrievalPipeline / RAGChain)
    ↓
Specialised modules (Loader / Processor / Chunker / Retriever / Reranker)
    ↓
AI Services + Storage Abstractions (provider-agnostic interfaces)
    ↓
Concrete Providers (Groq, SentenceTransformers, Qdrant, MinIO)
```

Each layer depends only on the layer below it. Concrete provider classes are never referenced outside the `ai/` and `storage/` packages.

---

## 12. Logging

Loguru is initialised in `app/core/logging.py` and called at startup:

- **stdout**: colourised, `INFO+`, format includes timestamp, level, module, function, line, and message
- **`logs/omnirag.log`**: JSON (fully serialised), `DEBUG+`, rotated at 10 MB, retained for 7 days, compressed with gzip

All modules import `logger` directly from `app.core.logging`.

---

## 13. Error Handling

`OmniRAGException` is the single application exception type. It carries:

- `message` — human-readable
- `status_code` — HTTP status (default 500)
- `error_code` — machine-readable string (e.g. `INTERNAL_ERROR`)

The FastAPI handler in `main.py` converts these to a consistent JSON error envelope. Unhandled exceptions propagate as standard FastAPI 500 responses.

AI provider calls use `retry_async` for transient failures. Ingestion failures are caught at the service level, which sets `status=FAILED`, records an error event, and logs the exception without crashing the process.

---

## 14. Background Jobs

Document processing runs as a FastAPI `BackgroundTask` (not a task queue). The task uses its own database session (`SessionLocal`) independent of the request session, ensuring the HTTP response is returned before processing begins.

For large-scale deployments, this could be replaced by a proper task queue (Celery, ARQ, or similar) with minimal changes to `DocumentService.process_document()`.

---
