# app.py — Eriski v2.2 (Fast Mode, better OCR, stability fixes)
# --------------------------------------------------------------

import os
import re
import glob
import time
import shlex
import subprocess
from typing import List, Tuple, Dict, Any
import numpy as np
import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="Eriski — Physics RAG", page_icon="⚡", layout="wide")

# -------------------- Defaults --------------------
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
FAST_EMBED_MODEL = "mxbai-embed-large"

NOTES_DIR = "notes"
DB_DIR = "chroma_db"
OCR_CACHE_DIR = "ocr_cache"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# -------------------- Theme --------------------
with st.sidebar:
    theme_choice = st.radio("Theme", ["Light", "Dark"], index=0)
if theme_choice == "Dark":
    st.markdown("""
        <style>
            body, .stApp { background: #0f1116; color: #f0f2f6; }
            .stTextInput>div>div>input, .stTextArea textarea { background:#161a22; color:#f0f2f6; }
            .stButton>button { background:#1f2430; color:#f0f2f6; border:1px solid #2b3243; }
            .stSelectbox>div>div>div { background:#161a22; color:#f0f2f6;}
            .block-container { padding-top: 1.2rem; }
        </style>
    """, unsafe_allow_html=True)

# -------------------- Model helpers --------------------
def ensure_model(model, is_embed=False):
    try:
        if is_embed:
            _ = ollama.embeddings(model=model, prompt="test")
        else:
            _ = ollama.chat(model=model, messages=[{"role": "user", "content": "test"}])
        return True
    except Exception:
        return False

# -------------------- OCR --------------------
def set_tesseract_path(path):
    if path and os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path

def ocr_image_to_text(img_path, lang="eng", fast_mode=False):
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    base = os.path.basename(img_path)
    cache_txt = os.path.join(OCR_CACHE_DIR, base + ".txt")
    if os.path.exists(cache_txt):
        with open(cache_txt, "r", encoding="utf-8") as f:
            return f.read()

    try:
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        if fast_mode:
            img.thumbnail((2000, 2000))
        img = img.filter(ImageFilter.MedianFilter(size=2))
        img = ImageOps.autocontrast(img)
        config = "--oem 1 --psm 6" if fast_mode else "--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        with open(cache_txt, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        return f"[OCR error: {e}]"

# -------------------- Load Notes --------------------
def load_txt_md(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf_pages(path):
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        yield page.extract_text() or "", (i + 1), path

def load_all_notes(notes_dir="notes", fast_mode=False):
    docs = []
    for p in glob.glob(os.path.join(notes_dir, "**", "*"), recursive=True):
        if os.path.isdir(p): continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in [".txt", ".md"]:
                docs.append((load_txt_md(p), None, {"source": p, "kind": "text"}))
            elif ext == ".pdf":
                for txt, page_num, path in load_pdf_pages(p):
                    docs.append((txt, f"p{page_num}", {"source": path, "page": page_num, "kind": "pdf"}))
            elif ext in IMAGE_EXTS:
                txt = ocr_image_to_text(p, fast_mode=fast_mode)
                docs.append((txt, None, {"source": p, "kind": "image"}))
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return docs

# -------------------- Chunking --------------------
def chunk_text(text, chunk_size=850, overlap=150):
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= chunk_size:
            cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur)
            tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
            cur = (tail + " " + s).strip()
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]

# -------------------- Vector Store --------------------
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=True))
    try:
        return client.get_collection("physics_notes")
    except:
        return client.create_collection("physics_notes")

def embed_texts(texts, model):
    vecs = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)
        vecs.append(r["embedding"])
    return vecs

def build_index(embed_model, fast_mode=False, rebuild=False):
    if not ensure_model(embed_model, is_embed=True):
        st.error(f"Embedding model `{embed_model}` missing. Run: ollama pull {embed_model}")
        return
    col = get_collection()
    if rebuild:
        client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=True))
        client.reset()
        col = client.create_collection("physics_notes")

    raw_docs = load_all_notes(NOTES_DIR, fast_mode=fast_mode)
    if not raw_docs:
        st.warning("No notes found (.txt, .pdf, .png, .jpg).")
        return

    ids, texts, metas = [], [], []
    idx = 0
    chunk_sz = 1200 if fast_mode else 850
    overlap = 80 if fast_mode else 150

    for content, id_suffix, meta in raw_docs:
        for chunk in chunk_text(content, chunk_sz, overlap):
            ids.append(f"{os.path.basename(meta['source'])}-{id_suffix or idx}-{idx}")
            texts.append(chunk)
            metas.append(meta)
            idx += 1

    BATCH = 64
    for s in range(0, len(texts), BATCH):
        batch = texts[s:s+BATCH]
        embs = embed_texts(batch, model=embed_model)
        col.add(ids=ids[s:s+BATCH], metadatas=metas[s:s+BATCH], documents=batch, embeddings=embs)
        st.write(f"Indexed {min(s+BATCH, len(texts))}/{len(texts)} chunks…")
    st.success("✅ Index built successfully!")

# -------------------- Retrieval --------------------
def retrieve(query, embed_model, k=5):
    col = get_collection()
    q_vec = embed_texts([query], model=embed_model)[0]
    res = col.query(query_embeddings=[q_vec], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]

# -------------------- Answering --------------------
SYSTEM_PROMPT = (
    "You are Eriski, a precise physics assistant. "
    "Use only the provided context. "
    "If info is missing, say what’s missing. "
    "Keep derivations short, clean, and correct."
)

def answer(query, llm_model, embed_model, k=5):
    if not ensure_model(llm_model):
        raise RuntimeError(f"LLM `{llm_model}` missing. Run: ollama pull {llm_model}")
    ctx = retrieve(query, embed_model, k=k)
    if not ctx:
        return "No relevant context found. Try rebuilding the index.", []
    context = "\n\n---\n\n".join([f"[{i+1}] {c.get('source','')}:\n{c['text']}" for i, c in enumerate(ctx)])
    prompt = f"""Answer the question using ONLY the context below.

# Context
{context}

# Question
{query}

# Guidelines
- Cite sources inline like [1], [2].
- Use brief step-by-step math when needed.
"""
    r = ollama.chat(model=llm_model, messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}], options={"temperature":0.2})
    return r["message"]["content"], ctx

# -------------------- UI --------------------
st.title(" Eriski — Fast Physics RAG")
st.caption("Offline AI assistant that reads your physics notes, PDFs, and images (OCR).")

with st.sidebar:
    st.subheader("Controls")
    fast_mode = st.checkbox("⚡ Fast Mode (faster, slightly less precise)", value=True)
    llm_model = st.text_input("LLM model", value=DEFAULT_LLM_MODEL)
    embed_model = FAST_EMBED_MODEL if fast_mode else DEFAULT_EMBED_MODEL
    top_k = st.slider("Top-k sources", 3, 10, 5)
    tess_path = st.text_input("Path to tesseract.exe", value="C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    if tess_path:
        set_tesseract_path(tess_path)

    st.markdown("---")
    st.markdown("**Upload notes** (.pdf, .txt, .md, .png, .jpg, .jpeg)")
    uploaded = st.file_uploader("Drop files", type=["pdf","txt","md","png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded:
        os.makedirs(NOTES_DIR, exist_ok=True)
        for f in uploaded:
            with open(os.path.join(NOTES_DIR, f.name), "wb") as fh:
                fh.write(f.getbuffer())
        st.success(f"Uploaded {len(uploaded)} file(s).")

    if st.button("🔁 Rebuild Index"):
        with st.spinner("Rebuilding index..."):
            build_index(embed_model, fast_mode=fast_mode, rebuild=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.pop("history", None)
        st.rerun()

# -------------------- Chat --------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Chat")
query = st.text_input("Ask a physics question", key="query")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            st.markdown(turn["content"])
            if "sources" in turn:
                with st.expander("📚 Sources used"):
                    for i, c in enumerate(turn["sources"], 1):
                        src = os.path.basename(c.get("source","?"))
                        kind = c.get("kind","text")
                        st.markdown(f"**[{i}] {src}** — kind: `{kind}`")
                        if kind == "image" and os.path.exists(c["source"]):
                            st.image(c["source"], caption=src, use_container_width=True)
                        snippet = c["text"][:700] + "..." if len(c["text"])>700 else c["text"]
                        st.text_area(f"Preview {i}", snippet, height=120, key=f"p_{i}_{src}_{time.time_ns()}")
        else:
            st.markdown(turn["content"])

if query and not st.session_state.get("just_answered", False):
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        try:
            ans, ctx = answer(query, llm_model, embed_model, k=top_k)
        except Exception as e:
            ans, ctx = f"⚠️ Error: {e}", []
    st.session_state.history.append({"role": "assistant", "content": ans, "sources": ctx})

    # 🧩 Mark that we just answered, so rerun won't re-trigger
    st.session_state.just_answered = True
    st.rerun()

# 🧹 After rerun, reset the flag so new queries can work
elif st.session_state.get("just_answered", False):
    st.session_state.just_answered = False

