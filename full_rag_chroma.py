import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
from chonkie import TokenChunker
from pathlib import Path
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parent / ".env")

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunker = TokenChunker(
     tokenizer = "character",
     chunk_size = 300,
     chunk_overlap = 50
)

chunks = chunker.chunk(text)

texts = []
for c in chunks:
    texts.append(c.text)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text).tolist()

ids = []
embeddings = []
for i in range(len(chunks)):
    ids.append("doc_" + str(i))
    embeddings.append(embed(texts[i]))

client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="sample_docs")

if collection.count() == 0:
    collection.add(
        ids = ids,
        documents = texts,
        embeddings = embeddings
    )

query = "What was Ratan Tata's leadership mantra?"
query_embedding = embed(query)

results = collection.query(
    query_embeddings = [query_embedding],
    n_results = 3
)

top_chunks = results["documents"][0]
context = ""

for chunk in top_chunks:
    context = context + chunk + "\n\n"

prompt = f"""Answer ONLY using the context below.
If not in context, say "I don't know."

Context:
{context}

Question: {query}
"""

llm_client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
)

llm_response = llm_client.chat.completions.create(
    model = "openrouter/free",
    messages = [
        {"role": "system", "content": "Answer only from the provided context."},
        {"role": "user", "content": prompt}
    ]
)

print(llm_response.choices[0].message.content)
