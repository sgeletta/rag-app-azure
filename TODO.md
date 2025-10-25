# Future Enhancements for RAG Application

This document outlines potential next steps to improve the security and maintainability of the deployed RAG application.

## 1. Secure Secret Management

Currently, user credentials are managed via a `secrets.toml` file that is part of the Docker image. This is not ideal for production environments.

**Task**: Refactor the application to use a more secure, cloud-native secret store.

-   **Option A (Recommended)**: Use Azure Container Apps' built-in secret store. This involves creating secrets at the container app level and injecting them as environment variables.
-   **Option B (More Advanced)**: Integrate with Azure Key Vault. This provides centralized, enterprise-grade secret management and is ideal if you plan to expand the application with more services.

This change will decouple sensitive data from your application code, improving security and making credential rotation easier.

## 2. Automated Deployments (CI/CD)

The current workflow for updating the application is manual (edit code, build, push, restart). This is time-consuming and prone to error.

**Task**: Implement a Continuous Integration/Continuous Deployment (CI/CD) pipeline.

-   **Recommended Tool**: GitHub Actions.
-   **Pipeline Steps**:
    1.  **Trigger**: On a `git push` to the `main` branch.
    2.  **Build**: Build the Docker images for `rag-app` and `ollama-app`.
    3.  **Push**: Push the newly built images to your Azure Container Registry (ACR).
    4.  **Deploy**: Trigger an update to the Azure Container Apps to use the new images. This can be done by updating the container app revision or by using the `az containerapp update` command.

Automating this process will significantly speed up development cycles and ensure that every change is deployed consistently and reliably.