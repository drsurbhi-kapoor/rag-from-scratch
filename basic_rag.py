import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import math

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

from sentence_transformers import SentenceTransformer

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

def chunk_text(text, chunk_size = 300, overlap= 50):
    chunks = []
    start = 0
    while start<len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end-overlap
    return chunks

chunks = chunk_text(text)
print(f"total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i} ---\n{c[:100]}...")

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text).tolist()

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b)

chunk_embedding = []
for c in chunks:
    chunk_embedding.append(get_embedding(c))

query = "What was Ratan Tata's leadership mantra?"
quer_embed = get_embedding(query)

scores = []
for emb in chunk_embedding:
    scores.append(cosine_similarity(quer_embed, emb))

top_idx = scores.index(max(scores))
for i, s in enumerate(scores):
    print(f"Chunk {i}: score = {s:.4f}")

top_k = 3
pairs = []
for i in range(len(scores)):
    pairs.append((scores[i], i))

pairs.sort(reverse=True)

top_indices = []
for j in range(top_k):
    top_indices.append(pairs[j][1])
print("Top 3 indices:", top_indices)

context = ""
for idx in top_indices:
    context = context + chunks[idx] + "\n\n"

prompt = f"""Answer ONLY using the context below.
If not in context, say "I don't know."
Context:
{context}
Question: {query}
"""
response = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)
