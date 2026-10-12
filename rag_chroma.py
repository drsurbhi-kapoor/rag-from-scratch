import chromadb
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="sample_docs")
print("files in DB: ", collection.count())

text_path = Path(__file__).resolve().parent / "sample.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

def make_chunk(text, size = 500, overlap = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start+size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = make_chunk(text)
print("total_chunks: ", len(chunks))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(text):
    return model.encode(text).tolist()

if collection.count() == 0:
    print("indexing...")

    ids = []
    embeddings = []

    for i in range(len(chunks)):
        ids.append("doc_"+ str(i))
        embeddings.append(embed(chunks[i]))

    collection.add(
        ids = ids,
        documents = chunks,
        embeddings = embeddings
    )
    print("Done! Saved", collection.count(), "chunks")
else:
    print("Already indexed:", collection.count(), "chunks")

query = "Who is Ratan Tata and what is he known for?"
query_embedding = embed(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

top_chunks = results["documents"][0]
context = ""

for chunk in top_chunks:
    context += chunk+ "\n\n"

prompt = f"""Answer ONLY using the context below.
If not in context, say "I don't know."

context: {context}
Question: {query}
"""

llm_client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
    )

response = llm_client.chat.completions.create(
    model = "openrouter/free",
    messages = [{"role":"user", "content": prompt}]
)
print(response.choices[0].message.content)
