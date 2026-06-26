# Telecom RAG Assistant

## Overview

Telecom RAG Assistant is a Retrieval-Augmented Generation (RAG) system for Radio Access Networks (RAN). It combines telecom standards, network telemetry, and a hybrid retrieval pipeline with a local LLM to automate network diagnosis and troubleshooting.

The knowledge base includes **3GPP Release 16/18**, **TeleQnA**, **O-RAN logs**, and telecom simulation datasets, enabling accurate and explainable responses to telecom-specific queries.

---

## Features

* Hybrid retrieval using ChromaDB and BM25
* Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking
* Local LLM for privacy-preserving inference
* Explainable responses with source traceability
* React frontend with FastAPI backend

---

## Project Structure

```text
Telecom-RAG/
├── backend/
│   ├── chunking_retrieval/
│   ├── retrieval_index/
│   └── dataset/
├── frontend/
└── README.md
```

---

## Tech Stack

**Backend:** Python, FastAPI, ChromaDB, BM25, Sentence Transformers
**Frontend:** React, Vite
**AI:** BGE Embeddings, Hybrid Retrieval, RRF, Cross-Encoder, Local LLM

---

## Datasets

* 3GPP Release 16 & 18
* TeleQnA
* O-RAN Anomaly Logs
* Telecom Simulation Data

---

## Installation

### Backend

```bash
cd backend/chunking_retrieval
pip install -r requirements.txt
python main.py
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: `http://localhost:5173`

---

## Usage

1. Start the backend and frontend.
2. Enter a telecom query or network log.
3. The system retrieves relevant documents and generates an explainable diagnostic report.

Example queries:

* What causes an RRC Connection Setup failure?
* Analyze this O-RAN telemetry log.
* Diagnose high packet loss in a 5G RAN.

---

## Retrieval Pipeline

1. Dense retrieval (ChromaDB)
2. Sparse retrieval (BM25)
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder re-ranking

---


