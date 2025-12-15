# 📄 Exact-Text RAG using LangChain, Ollama & Chroma

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)**
pipeline that retrieves **exact word-to-word text** from a PDF document.

Unlike typical RAG systems that paraphrase or summarize,
this implementation forces the LLM to behave as a **text extractor**.

---

## 🎯 Objective

- Index a PDF document into a vector database
- Retrieve only relevant content using semantic search
- Ensure the output matches the document **exactly as written**
- Prevent hallucination or paraphrasing

---

## 🧠 Architecture

PDF
↓
Text Chunking (paragraph-based)
↓
Embeddings (mxbai-embed-large)
↓
Chroma Vector Store
↓
Retriever (Top-K semantic search)
↓
LLM (llama3, temperature=0)
↓
Exact text output


---

## 📦 Tech Stack

| Component | Purpose |
|---------|--------|
| LangChain | Orchestration |
| Ollama | Local LLM + Embeddings |
| Chroma | Vector database |
| PyPDF | PDF text extraction |

---

## ⚙️ Setup Instructions

1. Install Python dependencies

```bash
python3 -m venv ragenv

source  ragenv/bin/activate

pip install -r requirements.txt
```
2. Install Ollama

    https://ollama.com

3. Pull required models

```bash
ollama pull llama3
ollama pull mxbai-embed-large
```

4. Place/replace your PDF and modify the PDF_PATH in main.py

5. Run the application

```bash
python main.py
```

## ❗Limitations

If text spans multiple chunks, partial answers may appear

OCR-scanned PDFs may not work correctly

No reasoning or summarization (by design)

## 📌 Use Cases

Employee handbooks
Legal policies
Compliance documents
Audit & governance systems
HR knowledge bases


## ⭐ Example
```bash
Enter your search text (or type 'exit'): What do we mean by most likely?

The “most likely” completion is learned by experiencing examples of real 
text.
During “pre-training”, the LLM is exposed to hundreds of millions of examples of real 
text, which it attempts to complete (like fill in the _GAP_) 
When it is wrong, it is penalised, and its internal state is updated to reflect that
This continues until it is really good at predicting real text

```