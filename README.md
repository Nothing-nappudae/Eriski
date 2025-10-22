# Eriski — Local Physics RAG Assistant

**Eriski** is a privacy-first, offline AI assistant that reads my physics notes (PDFs, text, and even handwritten images!) and answers conceptual or derivation-based questions using locally-hosted LLMs.

Built for students who want an AI that actually understands _their own_ notes — no cloud, no data leaks.

---

## Core Features

- **Local RAG (Retrieval-Augmented Generation)** — powered by [Ollama](https://ollama.ai) models.
- **Private Vector Database** — stores note embeddings using [ChromaDB](https://www.trychroma.com).
- **Image-to-Text (OCR)** — extracts text from handwritten or printed notes via Tesseract.
- **Multi-format Support** — `.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`.
- **Step-by-Step Physics Derivations** — concise explanations with citations from your notes.
- **Chat History Memory** — maintains your conversation across sessions.
- **Dark / Light Mode UI** — sleek and minimal Streamlit interface.
- **Fast Mode Toggle** — uses smaller, faster embeddings (MXBAI) for quick indexing.
- **100% Offline** — nothing ever leaves your machine.

---

## 🛠 Tech Stack

| Component  | Tool                                                        |
| ---------- | ----------------------------------------------------------- |
| Interface  | [Streamlit](https://streamlit.io)                           |
| Vector DB  | [ChromaDB](https://www.trychroma.com)                       |
| Embeddings | `nomic-embed-text` / `mxbai-embed-large`                    |
| LLM        | `llama3.1:8b` via [Ollama](https://ollama.ai)               |
| OCR        | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) |
| Language   | Python 3.11+                                                |

---

## 🧪 How It Works

1. Upload your notes (PDF / text / image).
2. Eriski extracts text, chunks it, and builds a local embedding index.
3. Ask any physics question — it retrieves relevant notes and cites sources inline.
4. You can verify every derivation with references from your own material.

---

## ⚙️ Installation

```bash
git clone https://github.com/Nothing-nappudae/physics-rag
cd physics-rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
