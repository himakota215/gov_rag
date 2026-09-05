# Government Scheme RAG Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that helps users find information from government scholarship scheme documents using natural-language questions.

The system combines semantic search, keyword-based search, OCR, hybrid retrieval, and an LLM to provide answers grounded in government scheme documents.

---

## 🚀 Features

- 📄 PDF document ingestion
- 🔍 OCR support for scanned/image-based PDFs
- ✂️ Intelligent document chunking
- 🧠 Sentence Transformer embeddings
- ⚡ FAISS vector similarity search
- 🔤 BM25 keyword-based retrieval
- 🔀 Reciprocal Rank Fusion (RRF) for hybrid retrieval
- 🏷️ Metadata and category-based filtering
- 🤖 LLM-powered question answering
- 💬 Streamlit chatbot interface
- 📊 Retrieval evaluation using Precision@1 and Recall@5

---

## 🏗️ System Architecture

```text
                  Government Scheme PDFs
                           │
                           ▼
                  PDF Text Extraction
                           │
                    ┌──────┴──────┐
                    │             │
              Normal PDF      Scanned PDF
                    │             │
                    │           OCR
                    │             │
                    └──────┬──────┘
                           ▼
                  Text Cleaning
                           │
                           ▼
                       Chunking
                           │
                           ▼
                  Sentence Transformers
                  all-MiniLM-L6-v2
                           │
                           ▼
                     FAISS Index
                           │
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
              FAISS Search      BM25 Search
                  │                 │
                  └────────┬────────┘
                           ▼
                    RRF Hybrid Search
                           │
                           ▼
                     Top-K Chunks
                           │
                           ▼
                         LLM
                           │
                           ▼
                   Streamlit Chatbot