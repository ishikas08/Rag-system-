# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a production-ready Retrieval-Augmented Generation (RAG) system
for AI research papers. It uses a hybrid retrieval approach by combining keyword-based
search (BM25) and vector-based semantic search (FAISS) to retrieve relevant documents
and generate context-aware responses.

---

## Project Structure
rag-system/
├── src/
├── data/
├── logs/
├── requirements.txt
├── README.md
└── .gitignore

---

## Architecture
User Query → Hybrid Search → Top-K Documents → Response Generator

Hybrid Search:
- BM25 for keyword relevance
- FAISS for semantic similarity
- Weighted score-based re-ranking

---

## Setup Instructions
```bash
pip install -r requirements.txt
python src/main.py
