# Project Context Summary: RAG Application

*Last Updated: After completing CI/CD setup.*

This document provides a comprehensive overview of the project's current state, architecture, and deployment configuration. It serves as a reference to quickly re-establish context for future development sessions.

## 1. Project Overview

-   **Purpose**: A containerized, local-first Retrieval-Augmented Generation (RAG) application.
-   **Core Technologies**:
    -   **Frontend**: Streamlit
    -   **LLM Service**: Ollama (running the `mistral` model)
    -   **Vector Store**: FAISS
    -   **Embeddings**: HuggingFace (`BAAI/bge-small-en`)
    -   **Containerization**: Docker

## 2. Architecture & File Structure

-   **`rag_app.py`**: The main Streamlit application file containing all UI and backend logic.
-   **`Dockerfile`**: A multi-stage Dockerfile that builds the Python environment and creates a final image running as a non-root user (`appuser`).
-   **Data Directories**:
    -   `docs/`: Stores uploaded source documents.
    -   `faiss_indexes/`: Stores the generated FAISS vector indexes.
    -   `logs/`: Contains SQLite databases for query history and feedback.
-   **CI/CD Workflows**: Located in `.github/workflows/`.
    -   `docker-publish.yml`: Pushes the `rag-app` image to Docker Hub for public/local use.
    -   `azure-deploy.yml`: Deploys the application to Azure Container Apps.

## 3. Deployment & Configuration

The application has two primary deployment targets: local development and Azure production.

### 3.1. Local Deployment

-   **Method**: Uses `docker-compose.yml` as described in `README.md`.
-   **Authentication**: Reads `RAG_APP_USERNAME` and `RAG_APP_PASSWORD` from a local `.env` file.

### 3.2. Azure Production Deployment

-   **Platform**: Azure Container Apps.
-   **Resources**:
    -   **Resource Group**: `rag-app-rg`
    -   **Container Apps Environment**: `rag-app-env`
    -   **Container Apps**:
        -   `rag-app`: The public-facing Streamlit application.
        -   `ollama-app`: The internal LLM service.
    -   **Container Registry (ACR)**: `ragappacr9626`
    -   **Storage**: An Azure File Share named `ragappstorage` is mounted to both container apps to persist the `docs`, `faiss_indexes`, `logs`, and `ollama` model data.

-   **Authentication**:
    -   The `rag-app` container app is configured with secure secrets (`rag-app-username`, `rag-app-password`).
    -   These secrets are injected into the container as environment variables (`RAG_APP_USERNAME`, `RAG_APP_PASSWORD`).
    -   The application code (`rag_app.py`) reads these environment variables for user authentication.

-   **CI/CD Pipeline (GitHub Actions)**:
    -   **Trigger**: Automatically runs on any `git push` to the `main` branch.
    -   **Process**:
        1.  Builds the `rag-app` Docker image.
        2.  Pushes the image to the private Azure Container Registry (`ragappacr9626`).
        3.  Updates the `rag-app` Container App to pull the new image, triggering a new revision.
    -   **Authentication**: The workflow authenticates with Azure using a Service Principal whose credentials are stored in GitHub Actions secrets (`AZURE_CREDENTIALS`).

## 4. Project Status

-   **Secure Secret Management**: **Complete**. The application no longer uses insecure files and pulls credentials from the Azure environment.
-   **Automated Deployments (CI/CD)**: **Complete**. A GitHub Actions workflow fully automates the deployment to Azure.
-   **Documentation**:
    -   `README.md` has been updated to reflect local and cloud deployment processes.
    -   `TODO.md` tasks are all addressed.

## How to Use This Document

When resuming work, provide this document as context to ensure a full understanding of the project's current state.
