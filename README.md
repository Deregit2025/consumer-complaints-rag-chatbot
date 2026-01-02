# RAG Chatbot Project – Task 1 & Task 2

## Project Overview

This project implements the **data preparation and vectorization foundation** for a Retrieval-Augmented Generation (RAG) chatbot. The chatbot is designed to answer user questions based on historical customer complaint data.

The work completed so far focuses on:

* Cleaning and preparing raw complaint data
* Designing a reproducible data pipeline
* Chunking text for semantic search
* Generating embeddings
* Building and persisting a FAISS vector store

These steps establish a **production-ready retrieval layer** that can later be connected to an LLM for question answering.

---

## Repository Structure (Relevant)

```
rag_chatbot/
│
├── data/
│   ├── raw/                  # Original raw complaint data
│   └── processed/            # Cleaned & sampled data (not committed)
│
├── src/
│   ├── data_pipeline.py      # Task 1: data cleaning & preprocessing
│   └── vector_pipeline.py    # Task 2: chunking, embeddings, vector store
│
├── vector_store/             # FAISS index (ignored in git)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Task 1 – Data Preparation & Cleaning

### Objective

Prepare raw complaint data so it is suitable for semantic retrieval and downstream NLP tasks.

### Key Steps

1. **Load Raw Data**

   * Input: raw complaints dataset (CSV)
   * Large-scale, multi-column data with mixed types

2. **Column Selection & Renaming**

   * Retained only relevant textual and categorical fields
   * Removed identifiers and unused metadata

3. **Text Cleaning**

   * Lowercasing
   * Removal of extra whitespace
   * Handling missing or null complaint narratives

4. **Filtering Invalid Records**

   * Dropped rows with empty or unusable text
   * Ensured only meaningful complaints remain

5. **Output**

   * Cleaned CSV saved under `data/processed/`
   * This file is **derived** and therefore ignored from Git

### Outcome

A clean, consistent dataset suitable for chunking and embedding, with reduced noise and improved semantic quality.

---

## Task 2 – Vectorization & Indexing Pipeline

### Objective

Transform cleaned complaint text into searchable vectors and store them in a persistent vector database.

---

### Step 1: Stratified Sampling

* Applied **category-based stratified sampling**
* Ensured balanced representation across complaint categories
* Targeted ~12,000 complaints for efficient experimentation

This prevents dominance of high-frequency categories and improves retrieval fairness.

---

### Step 2: Text Chunking

* Used `RecursiveCharacterTextSplitter`
* Chunk size and overlap chosen to:

  * Preserve semantic coherence
  * Avoid truncation of important context

**Result:**

* ~12,000 complaints
* ~22,651 text chunks

Each chunk becomes an independent retrievable unit.

---

### Step 3: Embedding Generation

* Model used: `sentence-transformers/all-MiniLM-L6-v2`
* Properties:

  * Lightweight
  * Open-source
  * Optimized for semantic similarity

Each text chunk is converted into a dense numerical vector suitable for similarity search.

---

### Step 4: FAISS Vector Store Creation

* Vector database: **FAISS**
* Index type: default flat index (exact search)

Stored artifacts:

```
vector_store/faiss_index/
├── index.faiss   # Vector index
└── index.pkl     # Metadata & document mapping
```

> ⚠️ These files are **not committed to Git** because they are large, derived artifacts and can be regenerated at any time.

---

### Execution Output (Successful Run)

* Cleaned data loaded successfully
* Stratified sampling applied
* 22,651 chunks created
* Embeddings generated
* FAISS index saved locally

✅ Vector store creation completed successfully

---

## Dependency Management Notes

### LangChain Versioning

* The vector pipeline is compatible with **LangChain pre-1.0 (e.g., 0.0.214)**
* LangChain ≥1.0 introduces modular package splits that break legacy imports:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
```

For stability and reproducibility, the project currently pins a **pre-1.0 LangChain version**.

---

## Reproducibility Instructions

To rebuild everything from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data cleaning (Task 1)
python -m src.data_pipeline

# 3. Build vector store (Task 2)
python -m src.vector_pipeline
```

This will regenerate the FAISS vector store locally.

---

## What Has Been Achieved So Far

✔ Clean and reliable complaint dataset
✔ Deterministic preprocessing pipeline
✔ Balanced sampling strategy
✔ Semantic chunking strategy
✔ Open-source embedding generation
✔ Persistent FAISS vector index

These components together form the **retrieval backbone** of the RAG system.

---

## Next Steps (Preview)

* Task 3: Retrieval layer (query → relevant chunks)
* Task 4: LLM integration for answer generation
* Task 5: UI (Gradio) and evaluation

---

## Key Design Principles Followed

* Reproducibility over artifacts
* Open-source tooling
* Modular pipelines
* Production-aligned best practices

---

