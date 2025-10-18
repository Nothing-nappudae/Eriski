# app.py
import os
import glob
import math
import argparse
from typing import List, Tuple
import ollama
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

#  load notes --------


def load_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def load_all_notes(notes_dir: str = "notes") -> List[Tuple[str, str]]:
    docs = []
    for p in glob.glob(os.path.join(notes_dir, "**", "*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in [".txt", ".md"]:
                docs.append((p, load_txt_md(p)))
            elif ext in [".pdf"]:
                docs.append((p, load_pdf(p)))
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return docs

#  simple text chunking


def chunk_text(text: str, chunk_size=900, overlap=150) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

#  embeddings via Ollama


def embed_texts(texts: List[str], model="nomic-embed-text") -> List[List[float]]:
    vectors = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)
        vectors.append(r["embedding"])
    return vectors

#  vector store (Chroma)


def get_collection(persist_dir="chroma_db", name="physics_notes"):
    client = chromadb.PersistentClient(
        path=persist_dir, settings=Settings(allow_reset=True))
    try:
        return client.get_collection(name=name)
    except:
        return client.create_collection(name=name)


def build_index(notes_dir="notes", persist_dir="chroma_db", rebuild=False):
    col = get_collection(persist_dir)
    if rebuild:
        # wipe and recreate
        client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(allow_reset=True))
        client.reset()
        col = client.create_collection(name="physics_notes")

    docs = load_all_notes(notes_dir)
    if not docs:
        print("No notes found in ./notes (supported: .txt, .md, .pdf)")
        return

    ids, texts, metas = [], [], []
    print(f"Indexing {len(docs)} files...")
    idx = 0
    for path, content in docs:
        for chunk in chunk_text(content):
            ids.append(f"{os.path.basename(path)}-{idx}")
            texts.append(chunk)
            metas.append({"source": path})
            idx += 1

    BATCH = 64
    for s in range(0, len(texts), BATCH):
        batch = texts[s:s+BATCH]
        embs = embed_texts(batch)
        col.add(ids=ids[s:s+BATCH], metadatas=metas[s:s+BATCH],
                documents=batch, embeddings=embs)
        print(f"Added {min(s+BATCH, len(texts))}/{len(texts)} chunks")
    print("Index built ✅")


def retrieve(query: str, k=5, persist_dir="chroma_db"):
    col = get_collection(persist_dir)
    q_emb = embed_texts([query])[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    pairs = [(d, m.get("source", "unknown")) for d, m in zip(docs, metas)]
    return pairs


# sys prom
SYSTEM_PROMPT = (
    "You are a careful Physics study assistant. "
    "Use ONLY the given context to answer. If the context is insufficient, say so. "
    "Show step-by-step derivations for calculations. Keep answers concise and correct."
)


def answer(query: str, k=5, model="llama3.1:8b"):
    context_pairs = retrieve(query, k=k)
    context_text = "\n\n---\n\n".join(
        [f"[{i+1}] {src}\n{txt}" for i, (txt, src) in enumerate(context_pairs)])
    user_prompt = f"""Answer the question using the context.

# Context
{context_text}

# Question
{query}

# Instructions
- Cite sources inline like [1], [2] using the bracket numbers above.
- If math is involved, derive clearly and keep it brief.
- If not enough info, say what’s missing and suggest which note to read.
"""
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"]

# Cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the index from notes/")
    args = parser.parse_args()

    if args.rebuild:
        build_index(rebuild=True)

    print("\nType your physics question (or 'exit'): ")
    while True:
        q = input(">> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\nThinking...\n")
        print(answer(q))
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
