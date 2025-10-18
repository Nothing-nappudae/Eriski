# 🧠 Eriski — My Physics AI Assistant

Eriski started as a random midnight idea —  
I just wanted an AI that could actually *read my own physics notes* instead of throwing random textbook answers at me.  

So I built one.

Eriski is a **local Retrieval-Augmented Generation (RAG)** bot that reads my notes, indexes them, and answers my physics questions directly from what I’ve studied.  
No cloud, no external API — just me, my PC, and my messy notes.

---

# ⚙️ What It Does
- Reads `.txt`, `.md`, and `.pdf` notes from a `notes/` folder  
- Breaks them into small chunks and stores them in a local database  
- Uses **Ollama (Llama 3.1)** + **LangChain** + **ChromaDB**  
- Lets me chat with my own notes through the terminal  
- Works 100% offline 🧾  

---

#  Why I Made It:
I was tired of flipping through old notebooks before every physics test.  
I wanted something that understood my explanations, my way of writing, and my confusion points.  
Eriski became that — a little assistant that speaks *my version* of physics.

---

🛠️ Tech Stack:
- **Python**
- **Ollama** for local LLMs (Llama 3.1)
- **LangChain** for RAG pipeline
- **ChromaDB** for vector search
- **FAISS embeddings** via `nomic-embed-text`

---

###  How to Run
```bash
git clone https://github.com/<yourusername>/eriski.git
cd eriski
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --rebuild   # builds database from your notes/
python app.py             # start chatting
