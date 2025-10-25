import os
import warnings

# --- Updated and Modern Imports ---
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants ---
DOCS_DIR = "docs/"
INDEX_PATH = "faiss_index_cli"

def load_document(file_path):
    """Loads a document from a file path based on its extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        # PDFPlumberLoader is often more robust for text extraction
        return PDFPlumberLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        print(f"Unsupported file type: {ext}. Skipping.")
        return None

# --- Embeddings and LLM ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
llm = Ollama(model="mistral")

# --- Load or Create Vector Store ---
if os.path.exists(INDEX_PATH):
    print("✅ Loading existing FAISS index...")
    vector_store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("✨ Creating new FAISS index...")
    all_docs = []
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.isfile(file_path):
            print(f"  - Processing {filename}...")
            docs = load_document(file_path)
            if docs:
                all_docs.extend(docs)

    if not all_docs:
        print("\n❌ No documents found or processed. Please add files to the 'docs' directory.")
        exit()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(INDEX_PATH)
    print(f"\n✅ Index created from {len(all_docs)} document(s) and saved to '{INDEX_PATH}'.")

# --- Modern RAG Pipeline using LCEL ---
retriever = vector_store.as_retriever()
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Q&A loop ----
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    response = chain.invoke(query)
    print("\nAnswer:", response)
