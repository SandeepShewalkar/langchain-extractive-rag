"""
RAG pipeline to extract EXACT word-to-word text from a PDF
using Ollama + Chroma + LangChain.

Goal:
- Index a PDF document
- Retrieve relevant chunks using embeddings
- Force the LLM to copy text exactly as it appears in the document
"""

import os
from pypdf import PdfReader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# CONFIGURATION
# -----------------------------

PDF_PATH = "AI-for-Education-RAG.pdf"
CHROMA_DB_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "ai-for-education"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3"
TOP_K_RESULTS = 3


# -----------------------------
# LOAD PDF
# -----------------------------
pdf_not_exists = not os.path.exists(PDF_PATH)

if pdf_not_exists:
     print("Please enter valid pdf path")
     exit()

reader = PdfReader(PDF_PATH)

# Initialize embedding model (used for vector search)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Check whether vector DB already exists
is_new_db = not os.path.exists(CHROMA_DB_DIR)


# -----------------------------
# CREATE DOCUMENT CHUNKS
# -----------------------------

documents = []

if is_new_db:
    for page_no, page in enumerate(reader.pages):
        text = page.extract_text()

        # Skip empty pages
        if not text:
            continue

        # Split text by paragraphs to preserve structure
        chunks = text.split("\n\n")

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={"page": page_no}
                )
            )


# -----------------------------
# VECTOR STORE (CHROMA)
# -----------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# Add documents only on first run
if is_new_db:
    vector_store.add_documents(documents=documents)


# -----------------------------
# RETRIEVER
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": TOP_K_RESULTS}
)


# -----------------------------
# LLM SETUP
# -----------------------------

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.0  # IMPORTANT: disables paraphrasing
)

output_parser = StrOutputParser()


# -----------------------------
# PROMPT (EXTRACTIVE, NOT GENERATIVE)
# -----------------------------

prompt = ChatPromptTemplate.from_template("""
You are a text extraction system.

Rules:
- Answer ONLY using the provided context
- Copy text EXACTLY as it appears
- Do NOT rephrase, summarize, or rename
- Preserve bullets, punctuation, and line breaks
- If exact text is not present, reply: "NOT FOUND"

Context:
{context}

Question:
{question}
""")


# -----------------------------
# RAG CHAIN
# -----------------------------

chain = prompt | llm | output_parser


# -----------------------------
# QUERY LOOP
# -----------------------------


while True:
    question = input("\nEnter your search text (or type 'exit'):")
    if question.lower() == "exit":
            break
    
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Combine retrieved chunks into context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run the chain
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n--- Answer ---")
    print(response)
