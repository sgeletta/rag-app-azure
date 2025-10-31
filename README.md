# Local RAG Application with Ollama and Streamlit

This project provides a fully containerized, local-first Retrieval-Augmented Generation (RAG) application. It uses Streamlit for the user interface, Ollama to run local language models like Mistral, and FAISS for efficient document searching.

The entire application and its dependencies are managed through Docker, making it easy to deploy and run on any machine with Docker installed.

## Features

- **Project-Based Organization**: Keep your documents and chat histories organized by project.
- **Multiple Document Types**: Upload and process PDFs, Word documents, CSVs, and Markdown files.
- **Local First**: Your documents and the LLM run entirely on your local machine. No data is sent to external services.
- **Interactive Q&A**: Chat with your documents through an intuitive Streamlit interface.
- **Feedback & Logging**: Track conversation history and provide feedback on model responses.
- **Dashboard**: Visualize query statistics and user feedback.

---

## Prerequisites

Before you begin, you will need to have the following software installed on your workstation.

**Docker & Docker Compose**: The application runs in Docker containers. The easiest way to get both is by installing [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system (Windows, macOS, or Linux).

---

## 🚀 Deployment

There are two ways to run the application. The Quick Start is recommended for most users.

---

### Option 1: Quick Start (Recommended)

This method uses pre-built Docker images from Docker Hub for the fastest and simplest setup. You do not need to clone any source code.

**Step 1: Create a Project Folder**

Create a new folder on your computer for the project and navigate into it.

**Step 2: Create Required Directories**

Inside your new project folder, create the following sub-directories. These will be used to persist your data and configuration.

- `docs`
- `faiss_indexes`
- `logs`

**Step 3: Create an Environment File for Authentication**

This application uses an environment file to manage local login credentials. In the root of your project folder, create a new file named `.env` and paste the following content into it.

```
# This file configures the login credentials for the local application.
# These variables are loaded by docker-compose.yml.
RAG_APP_USERNAME=admin
RAG_APP_PASSWORD=changeme
```
> **Important**: Be sure to change the default password `changeme` to a strong, secure password of your choice.

**Step 4: Create a `docker-compose.yml` File**

In the root of your project folder, create a file named `docker-compose.yml` and paste the following content into it:

```yaml
version: '3.8'

services:
  rag-app:
    image: sgeletta/rag-app:latest
    container_name: rag-app-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./docs:/app/docs
      - ./faiss_indexes:/app/faiss_indexes
      - ./logs:/app/logs
      - ./.streamlit:/app/.streamlit
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: sgeletta/rag-app-ollama:latest
    container_name: rag-app-ollama
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

### Step 2: Navigate to the Project Directory



Change into the newly created directory:

```bash
cd rag-app
```

### Step 3: Build and Run the Application

Execute the following command from the root of the project directory. This command tells Docker Compose to build the necessary Docker images and start the services.

```bash
docker-compose up --build
```

> **Note:** The first time you run this command, it may take several minutes. Docker will download the base images and, most importantly, the `mistral` language model (which is several gigabytes). Subsequent launches will be much faster.

You will see a lot of log output from both the `rag-app` and `ollama` services. This is normal.

### Step 4: Access the Application

Once the services are running, open your web browser and navigate to:

**http://localhost:8501**

You should now see the Streamlit application interface, ready for you to create your first project.

---

## How It Works

The `docker-compose up` command starts two services that work together:

1.  `ollama`: This service runs the Ollama server with the `mistral` model pre-packaged. It exposes the LLM on a network that is only accessible by the other Docker containers.
2.  `rag-app`: This is the main Streamlit application you interact with. It handles document uploads, vector index creation, and communicates with the `ollama` service to answer your questions.

## Managing the Application

### Stopping the Application

To stop the application, press `Ctrl + C` in the terminal where `docker-compose` is running. To ensure the containers are fully stopped and removed, you can run:

```bash
docker-compose down
```

### Data Persistence

Your data is safe! The `docker-compose.yml` file is configured to store all your important data on your host machine in the project directory:

- `docs/`: Contains the documents you upload for each project.
- `faiss_indexes/`: Contains the vector search indexes.
- `logs/`: Contains the query and feedback logs.

This means you can stop and restart the application without losing your work.

## Advanced: GPU Acceleration

If you have an NVIDIA GPU and have installed the NVIDIA Container Toolkit, you can significantly speed up the language model.

1.  Open the `docker-compose.yml` file.
2.  Uncomment the `deploy` section under the `ollama` service.
3.  Restart the application with `docker-compose up --build`.

Ollama will automatically detect and use the GPU.