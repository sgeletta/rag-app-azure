\# RAG App with Streamlit + LangChain



This is a local Retrieval-Augmented Generation (RAG) application that allows document uploading, semantic search, Q\&A over documents, session tracking, and analytics.



\## Features



\- Upload multiple document formats (PDF, DOCX, TXT, CSV, Markdown)

\- Chunk documents and build FAISS vector index

\- Ask questions using an embedded LLM (via Ollama)

\- Track queries with per-session logging

\- Interactive analytics dashboard with visual summaries

\- Real-time document preview and logging



\## Requirements



\- Python 3.9+

\- Install dependencies:

  ```bash

  pip install -r requirements.txt



\# 🚀 Deployment Guide



This application is designed to be run using Docker. Follow these steps to get the entire application stack running on your local machine.



\### Prerequisites



\-   \*\*Git:\*\* \[https://git-scm.com/downloads](https://git-scm.com/downloads)

\-   \*\*Docker Desktop:\*\* \[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)



\### Step 1: Clone the Repository



Open a terminal and clone this repository to your local machine.



```bash

git clone https://github.com/sgeletta/rag-app.git

cd rag-app



