# RAG from Scratch

Progressive **Retrieval-Augmented Generation** examples on a small public text file (`sample.txt` — biography of Ratan Tata). No proprietary documents are included.

## Pipeline overview

```
sample.txt → chunk → embed → retrieve → prompt → LLM answer
```

## Scripts (in suggested learning order)

| File | What it demonstrates |
|------|----------------------|
| `basic_rag.py` | Fixed-size character chunks, cosine similarity with SentenceTransformers, top-k context, OpenRouter LLM |
| `text_sentence_chunk.py` | Sentence-based grouping (manual chunking strategy) before embed + retrieve |
| `rag_chroma.py` | Same text indexed in **ChromaDB** with manual embeddings |
| `full_rag_chroma.py` | **Chonkie** `TokenChunker` + Chroma + full RAG in one script |

## Setup

```bash
cd rag-from-scratch
pip install -r requirements.txt
cp .env.example .env
# Add OPENROUTER_API_KEY to .env
```

Chroma folders (`chroma_db/`) are created when you run the Chroma scripts. They are gitignored.

## Run examples

```bash
python basic_rag.py
python text_sentence_chunk.py
python rag_chroma.py
python full_rag_chroma.py
```

## Sample questions (match `sample.txt`)

- What was Ratan Tata's leadership mantra?
- What is the Tata Nano and why was it introduced?
- Who is Ratan Tata and what is he known for?

## Notes for interviews

- **Chunking trade-offs:** character windows vs sentence groups vs token chunker (overlap, boundary issues).
- **Retrieval:** embedding similarity vs vector DB query.
- **Grounding:** prompts ask the model to answer only from context.
