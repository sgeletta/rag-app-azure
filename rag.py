import os
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=FutureWarning)

def load_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---- File path ----
file_path = "docs/example.pdf"

# ---- Load and split ----
docs = load_document(file_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ---- Embeddings and FAISS ----
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vector_store = FAISS.from_documents(chunks, embedding_model)

# ---- Local LLM via Ollama ----
llm = OllamaLLM(model="mistral")

# ---- RAG pipeline ----
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# ---- Q&A loop ----
print(f"\n✅ Loaded {len(chunks)} document chunks from {file_path}")
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    response = qa.invoke({"query": query})
    result = response.get("result", "Sorry, I could not find an answer.")
    print("\nAnswer:", result)
