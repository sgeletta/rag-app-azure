# Deploying the RAG Application to Azure

This guide provides step-by-step instructions to deploy the entire RAG (Retrieval-Augmented Generation) application stack to your own Azure subscription using a single PowerShell script.

The deployment is fully automated using Azure Bicep for Infrastructure as Code (IaC).

---

## Prerequisites

Before you begin, ensure you have the following tools installed and configured on your machine:

1.  **Azure CLI**: You must have the Azure CLI installed. You can download it from [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
    -   After installation, run `az login` from your terminal to authenticate with your Azure account.

2.  **Docker Desktop**: The script uses Docker to build and push the application container images. Docker Desktop is required. You can download it from here.
    -   Ensure the Docker engine is running before executing the deployment script.

3.  **PowerShell**: The deployment script is a PowerShell script (`.ps1`). It is installed by default on Windows. On macOS or Linux, you may need to install it.

---

## Deployment Steps

The entire deployment process is handled by the `deploy.ps1` script.

1.  **Open a PowerShell Terminal**: Navigate to the root directory of this project (`C:\Users\SimonGeletta\Documents\rag-app`).

2.  **Run the Deployment Script**: Execute the script from the project root directory as follows:

    ```powershell
    .\deploy\deploy.ps1
    ```

3.  **Follow the Prompts**: The script will prompt you for the following information:
    -   **Resource Prefix**: A short, unique, alphanumeric string (3-10 characters, e.g., `myragapp`) that will be used to name all the resources created in Azure.
    -   **Application Password**: A password (at least 8 characters) that you will use to log in to the web application's UI.

4.  **Wait for Deployment**: The script will now create all the Azure resources, build the Docker images, push them to the Azure Container Registry, and deploy the applications. This process can take several minutes.

5.  **Access Your Application**: Once the script completes, it will output the public URL of your running application. You can click this URL to access the RAG application in your browser.

---

## What Gets Deployed?

The script provisions the following resources into a new resource group (`<your-prefix>-rg`) in your Azure subscription:

-   **Log Analytics Workspace**: For collecting logs and monitoring the container apps.
-   **Virtual Network (VNet)**: A private network to securely host the application components.
-   **Azure Container Registry (ACR)**: A private registry to store your application's Docker images.
-   **Azure Storage Account**: Provides an Azure File Share to persist the downloaded language model, preventing re-downloads on application restart.
-   **Container Apps Environment**: A secure, serverless environment to run the containerized applications, connected to your VNet.
-   **`ollama-app` Container App**: The backend service that runs the language model. It is internal and not exposed to the public internet.
-   **`rag-app` Container App**: The frontend Streamlit web application that you interact with. It is publicly accessible via the URL provided upon completion.

---

## Cleaning Up

To avoid ongoing costs, you can easily delete all the resources created by this deployment.

The script creates a single resource group that contains everything. To remove it, run the following Azure CLI command, replacing `<your-prefix>` with the same prefix you provided during deployment.

```bash
# Replace <your-prefix> with the one you used for deployment
az group delete --name <your-prefix>-rg --yes --no-wait
```

This command will delete the resource group and all the resources within it. The `--no-wait` flag returns control to your terminal immediately while the deletion happens in the background.