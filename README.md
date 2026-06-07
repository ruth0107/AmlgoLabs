# 🔍 PolicyAI — RAG Chatbot

> **Amlgo Labs · Junior AI Engineer Assignment**  
> A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions from policy and legal documents using a locally-hosted LLM with real-time streaming responses.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    Streamlit (app.py)                           │
│         [Chat Input] → [Streaming Response] → [Sources]         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │   RAG Pipeline     │
                  │  src/pipeline.py   │
                  └────┬──────────┬────┘
                       │          │
           ┌───────────▼──┐  ┌────▼──────────────┐
           │  Retriever   │  │    Generator       │
           │ (+ Reranker) │  │  generator.py      │
           └───────┬──────┘  └────────┬───────────┘
                   │                  │
     ┌─────────────▼──┐    ┌──────────▼──────────┐
     │   ChromaDB     │    │  Mistral 7B Instruct │
     │  (vectordb/)   │    │  via Ollama (local)  │
     └─────────────┬──┘    └─────────────────────-┘
                   │
     ┌─────────────▼──────────────────┐
     │  BAAI/bge-small-en-v1.5        │
     │  sentence-transformers          │
     └─────────────────────────────────┘
                   │
     ┌─────────────▼──────────────────┐
     │  policies.pdf → chunks.json    │
     │  (data/ → chunks/)             │
     └─────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | `mistral:7b-instruct` via Ollama | Instruction-tuned, named in spec, runs locally |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | High retrieval accuracy, ~130MB, no GPU cost |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Second-stage cross-encoder for high-precision retrieval |
| **Vector DB** | ChromaDB (persistent) | Native Python, persistent storage, cosine similarity |
| **RAG Framework** | LangChain | Text splitting, orchestration |
| **UI** | Streamlit | Required by assignment, real-time streaming |
| **PDF Parsing** | PyMuPDF | Reliable text extraction, header/footer cleaning |

---

## 📁 Folder Structure

```
Amlgo Labs/
│
├── data/                        # Source documents & evaluation data
│   ├── policies.pdf
│   ├── eval_dataset.json        # Generated Q&A pairs for evaluation
│   └── rag_evaluation_report.md # RAGAS evaluation results
│
├── chunks/                      # Processed text segments
│   ├── chunks.json              # All chunks with metadata
│   └── chunk_distribution.png  # Visual analysis
│
├── vectordb/                    # ChromaDB persistent store
│   └── chroma_db/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb  # Chunking + index building
│   └── 02_evaluation.ipynb     # Hit@K, MRR evaluation
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # PDF loading + cleaning
│   ├── chunker.py              # Sentence-aware chunking
│   ├── embedder.py             # bge-small-en-v1.5 wrapper
│   ├── vectorstore.py          # ChromaDB operations
│   ├── retriever.py            # Semantic search + scoring
│   ├── reranker.py             # Cross-encoder reranking
│   ├── generator.py            # Mistral streaming via Ollama
│   ├── pipeline.py             # Full RAG orchestrator
│   └── evaluation/             # Evaluation scripts (RAGAS)
│       ├── dataset_generator.py
│       ├── metrics.py
│       └── run_eval.py
│
├── app.py                       # Streamlit chatbot
├── config.py                    # Central configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
- NVIDIA GPU recommended (CPU works but slower)

### Step 1 — Clone & Install

```bash
git clone <your-repo-url>
cd "Amlgo Labs"
pip install -r requirements.txt
```

### Step 2 — Pull the LLM

```bash
ollama pull mistral:7b-instruct
```

### Step 3 — Add Your Document

Place your PDF in the `data/` folder:
```
data/policies.pdf
```

### Step 4 — Run Preprocessing (Option A: Notebook)

```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```
Run all cells to chunk the document and build the ChromaDB index.

### Step 4 — Run Preprocessing (Option B: Script)

```bash
python src/vectorstore.py
```

### Step 5 — Launch the Chatbot

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  
Click **"Build Index"** in the sidebar if the index hasn't been built yet.

---

## 💬 Using the Chatbot

1. **Ask a question** in the chat input box
2. **Watch streaming** — the response appears token by token
3. **View sources** — click "📄 Sources Used" below each response to see retrieved chunks with relevance scores
4. **Clear chat** — use the sidebar button to reset the conversation

### Sample Queries

| Query | Expected Behavior |
|-------|------------------|
| `What personal data is collected?` | Retrieves data collection clauses |
| `How long is my data retained?` | Returns retention period info |
| `Can I delete my account?` | Shows data deletion rights |
| `What third parties receive my data?` | Lists data sharing practices |
| `What are my rights under GDPR?` | Returns rights section |

---

## 📐 Chunking Strategy

- **Chunk size**: ~1000 characters (~250 words)
- **Overlap**: 150 characters (~40 words)
- **Splitter**: `RecursiveCharacterTextSplitter` with sentence-boundary aware separators
- **Metadata**: Each chunk stores `chunk_id`, `source`, `word_count`, `char_count`

This ensures context is preserved at chunk boundaries while keeping each chunk semantically coherent.

---

## 🔢 Embeddings & Two-Stage Retrieval

To maximize relevance, the pipeline employs a two-stage retrieval process:

**Stage 1: Dense Vector Retrieval**
- **Model:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384-dimensional)
- **Vector Search:** ChromaDB fetches the top 15 candidate chunks using cosine similarity.
- **Query Prefix:** `"Represent this sentence for searching relevant passages: {query}"`

**Stage 2: Cross-Encoder Reranking**
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Function:** Scores the initial 15 candidates against the query to significantly boost precision.
- **Output:** Returns the final top-K (default 5) most relevant chunks to the generator.

---

## ⚡ Streaming Implementation

Streaming uses the Ollama Python client directly:

```python
stream = ollama.chat(
    model="mistral:7b-instruct",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
    options={"num_ctx": 2048, "temperature": 0.1}
)
for chunk in stream:
    token = chunk["message"]["content"]
    yield token
```

Streamlit renders tokens progressively via `st.markdown(full_response + "▌")`.

---

## 🧠 Prompt Format

```
You are a knowledgeable and precise assistant specializing in policy and legal documents.

Your task is to answer the user's question using ONLY the information from the provided context sections below.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information, respond with: "The provided documents do not contain sufficient information..."
3. Do not fabricate or infer beyond what is explicitly stated.

--- CONTEXT START ---
[Section 1]
{retrieved_chunk_1}

[Section 2]
{retrieved_chunk_2}
--- CONTEXT END ---

Question: {user_question}

Answer:
```

---

## 📊 Evaluation

The RAG pipeline is evaluated using a two-pronged approach:

### 1. RAGAS Metrics (End-to-End)
Run `python src/evaluation/run_eval.py` to evaluate the pipeline against the autogenerated Q&A dataset. Below are the current pipeline scores:

| Metric | Score | Description |
|--------|-------|-------------|
| **Context Recall** | `1.00` | How well retrieved context covers the ground truth. |
| **Groundedness** | `1.00` | Faithfulness of the generated answer to the retrieved context. |
| **Context Precision** | `0.52` | Proportion of retrieved chunks that are relevant. |
| **Hallucination Rate** | `0.00` | Rate of fabricated facts not present in context. |

*(Detailed analysis is available in `data/rag_evaluation_report.md`)*

### 2. Retrieval Metrics
Run `notebooks/02_evaluation.ipynb` for vector-search specific metrics:

| Metric | Description |
|--------|-------------|
| **Hit@1/3/5** | Was the correct chunk in the top K results? |
| **MRR** | Mean Reciprocal Rank across all queries |

---

## ⚠️ Known Limitations

| Issue | Description |
|-------|-------------|
| **Hallucination** | If retrieved chunks are marginally relevant, Mistral may still attempt an answer |
| **VRAM constraint** | On 4GB GPUs, context window is limited to 2048 tokens |
| **Long answers** | Responses capped at 512 tokens to prevent VRAM overflow |
| **Table extraction** | Complex PDF tables may not chunk cleanly with text splitters |

---

## 🖥️ Hardware Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM | 4 GB | 8 GB |
| Storage | 10 GB free | 20 GB free |
| CPU | 4 cores | 8+ cores |

---

*Built for Amlgo Labs Junior AI Engineer Assignment · June 2026*
