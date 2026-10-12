import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import math

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

text_path = Path(__file__).resolve().parent / "sample.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

from sentence_transformers import SentenceTransformer

def split_sentences(text):
    text = text.replace("\n", " ")
    sentences = text.split(". ")
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned.append(s + ".")
    return cleaned

sentences = split_sentences(text)
print(f"Total sentences: {len(sentences)}")

def chunk_by_sentence_count(sentences, n=3):
    chunks = []
    i = 0
    while i < len(sentences):
        group = sentences[i : i + n]
        chunk = " ".join(group)
        chunks.append(chunk)
        i = i + n
    return chunks

chunks = chunk_by_sentence_count(sentences, n=3)

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

query = "What is the Tata Nano and why was it introduced?"
quer_embed = get_embedding(query)

scores = []
for emb in chunk_embedding:
    scores.append(cosine_similarity(quer_embed, emb))

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
