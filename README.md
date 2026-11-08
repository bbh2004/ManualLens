# ManualLens 🔍📄

> **Multimodal Retrieval-Augmented Generation (RAG) for Technical PDF Manuals**  
> *A portfolio GenAI application delivering grounded answers, page citations, and diagram retrieval without hallucinations.*

---

## 🎯 What Is ManualLens?

ManualLens is an end-to-end multimodal RAG system designed for complex technical documentation (service manuals, datasheets, installation guides). When a user uploads a PDF manual and asks natural-language or exact-identifier questions (e.g., *"What is the operating voltage?"* or *"Error E17 appears, how do I fix it?"*), the system:
1. Runs **semantic vector search** over token-bounded chunks via `all-MiniLM-L6-v2` and **ChromaDB**.
2. Runs **exact keyword search** via pure-Python **BM25Okapi** to catch specific part numbers, beep codes, and model identifiers.
3. Fuses both retrieval paths using **Reciprocal Rank Fusion (RRF)**.
4. Performs **visual page retrieval** via **ColPali** (`vidore/colpali-v1.2` using `byaldi`), returning high-relevance schematic diagrams, pinouts, and tables.
5. Synthesizes a grounded, structured answer via **Groq API** with strict `[Page N]` citations and explicit uncertainty disclaimers when information is absent.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite + TS)                      │
│   UploadPanel  │  DocumentList  │  ChatInterface  │  AnswerCard        │
│                │                │                 │  PageViewer Modal  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST (JSON) / Vite Proxy (:5173 -> :8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        FastAPI Backend (:8000)                         │
│                                                                        │
│   ┌───────────────────────────┐    ┌───────────────────────────────┐   │
│   │        API Layer          │    │      Ingestion Pipeline       │   │
│   │  /upload     /status      │───▶│  PyMuPDF -> Chunker (200 tok)  │   │
│   │  /query      /documents   │    │  -> all-MiniLM -> ChromaDB    │   │
│   │  /pages      /health      │    │  -> BM25Okapi -> Pickle       │   │
│   └─────────────┬─────────────┘    │  -> 150 DPI PNGs -> ColPali   │   │
│                 │                  └───────────────────────────────┘   │
│                 ▼                                                      │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    Retrieval Orchestrator                    │     │
│   │  ChromaDB (top-5)  +  BM25 (top-5)  -> RRF Fusion (top-5)    │     │
│   │  ColPali Multi-Vector Search (top-3 visual diagram pages)    │     │
│   └─────────────────────────────┬────────────────────────────────┘     │
│                                 │                                      │
│                                 ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    Grounded Generation                       │     │
│   │  Evidence Assembly -> Strict Grounding Prompt -> Groq Client  │     │
│   │  Response Parser & Citation Validator -> Structured JSON     │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                       Storage Layer                          │     │
│   │  SQLite (metadata) │ ChromaDB (vectors) │ Qdrant/Byaldi      │     │
│   │  storage/pdfs/     │ storage/page_images/                     │     │
│   └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                            Groq Cloud API (Fast Inference)
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python:** 3.11, 3.12, or 3.13 (Standard GIL build)
- **Node.js:** v18+ (tested on Node 24 / npm 11)
- **Groq API Key:** Free key from [console.groq.com](https://console.groq.com)

### 2. Backend Setup
From the project directory:
```bash
# Activate virtual environment
# Windows:
..\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and enter your GROQ_API_KEY
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run build   # Verifies TypeScript & Vite build
```

### 4. Running Locally

**Terminal 1 — Backend:**
```bash
# In manuallens directory with venv activated:
uvicorn backend.main:app --reload --port 8000
```
API Documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🔬 Core RAG Technical Concepts

| Concept | Implementation in ManualLens |
|---|---|
| **Token-Bounded Chunking** | 200 tokens with 20 token overlap via `tiktoken`. Chunks strictly respect page boundaries to guarantee citation integrity. |
| **Hybrid Search (Dense + Sparse)** | Combines semantic proximity (`all-MiniLM-L6-v2`) with lexical BM25 matching (`rank-bm25`) for precision on error codes and component part numbers. |
| **Reciprocal Rank Fusion (RRF)** | Rank-based scoring `$score(d) = \sum \frac{1}{k + r_i}$` with `$k=60$`. Merges heterogeneous score distributions without manual calibration. |
| **ColPali Visual Retrieval** | Vision-language model treating entire PDF pages as images. Preserves schematics, tables, and exploded assembly diagrams that OCR collapses. |
| **Citation Grounding** | Programmatic validator checks every cited page against retrieved evidence. Rejects ungrounded claims and highlights manual coverage gaps. |

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Accepts multipart PDF upload; initiates asynchronous background ingestion pipeline. |
| `GET` | `/api/status/{doc_id}` | Returns real-time ingestion stage (`extracting_text`, `indexing_text`, `rendering_pages`, `ready`). |
| `POST` | `/api/query` | Submits question; returns grounded answer, citations, and visual diagram thumbnails. |
| `GET` | `/api/documents` | Lists all uploaded manuals and their statuses. |
| `DELETE` | `/api/documents/{doc_id}` | Cascading deletion across SQLite, ChromaDB, BM25, byaldi index, and filesystem. |
| `GET` | `/api/pages/{doc_id}/{page}` | Streams rendered 150 DPI page PNG for visual inspection in modal lightbox. |
| `GET` | `/health` | Health-check endpoint. |

---

## 🎙️ 3-Minute Interview Pitch

> *"ManualLens is a multimodal RAG system specifically engineered for technical manuals and service guides. Traditional text RAG often fails on technical documents because manuals rely heavily on exact identifiers (like error codes or part numbers) and complex visual layouts (like exploded assembly diagrams, wiring schematics, and pinouts).*
>
> *To solve this, I designed a multi-retrieval pipeline: dense embeddings using SentenceTransformers for conceptual queries, BM25 for exact codes, and Reciprocal Rank Fusion to synthesize the results. For visual content, I integrated ColPali, which embeds full page images using late-interaction patch representations so diagrams and tables are directly searchable by natural language.*
>
> *Answers are synthesized with strict citation grounding, mapping every statement directly back to verified page numbers with an inline page lightbox inspector."*

---

## 🗺️ Part 2 Roadmap (Confirmed Scope)
- **GraphRAG (Neo4j):** Knowledge graph representation of manual entities (`ErrorCode`, `Component`, `Procedure`) to enable multi-hop reasoning (`E17 -> Fan -> Cleaning Procedure -> Page 84`).
- **User Authentication:** JWT-based user authentication scoping document catalogs per user.
- **Cloud Deployment:** Migration to Railway (backend + managed PostgreSQL) + Vercel (frontend) + Cloudflare R2 (storage) + Qdrant Cloud.

