# TODO: Create Azure Distribution Package

This document outlines the tasks required to package the RAG application so it can be easily deployed by other users into their own Azure subscriptions and virtual networks.

The goal is to create a self-contained, automated deployment using Infrastructure as Code (IaC).

---

## Task 1: Convert to Infrastructure as Code (IaC) with Bicep

-   [ ] **Decompile Existing ARM Template**: Use `az bicep decompile` on `azure-deploy.json` as a starting point to understand the Bicep syntax for the container apps.
-   [ ] **Create `main.bicep`**: Create a new `main.bicep` file in a new `/deploy` directory.
-   [ ] **Define All Azure Resources**: Add the following resource definitions to `main.bicep`:
    -   [ ] Log Analytics Workspace
    -   [ ] Virtual Network (VNet) with at least two subnets (one for the Container Apps Environment).
    -   [ ] Azure Container Registry (ACR).
    -   [ ] Azure Storage Account and an Azure File Share.
    -   [ ] Azure Container Apps Environment (configured to be internal, using the VNet).
    -   [ ] `ollama-app` Container App definition.
    -   [ ] `rag-app` Container App definition.

---

## Task 2: Parameterize the Bicep Template

-   [ ] **Add Parameters**: In `main.bicep`, define parameters to make the template reusable.
    -   [ ] `location` (string, with a default value).
    -   [ ] `resourcePrefix` (string, for creating unique resource names).
    -   [ ] `vnetAddressPrefix` (string, for the VNet's IP address space).
    -   [ ] `ragAppUsername` (string).
    -   [ ] `ragAppPassword` (string, marked with `@secure()`).
-   [ ] **Use Parameters and Variables**: Refactor the resource definitions to use the `resourcePrefix` and other parameters instead of hardcoded names (e.g., `'rag-app-rg'`).

---

## Task 3: Create a Master Deployment Script

-   [ ] **Create `deploy.ps1`**: Create a new PowerShell script in the `/deploy` directory.
-   [ ] **Script Logic**: Add the following functionality to the script:
    -   [ ] Prompt the user for required inputs (resource prefix, password, etc.).
    -   [ ] Create the resource group (`az group create`).
    -   [ ] Deploy the `main.bicep` file (`az deployment group create`).
    -   [ ] Build the `rag-app` and `ollama-app` Docker images from the source code.
    -   [ ] Push the images to the newly created Azure Container Registry.
    -   [ ] Output the final FQDN (URL) of the `rag-app`.

---

## Task 4: Create User-Facing Documentation

-   [ ] **Create `DEPLOY_TO_AZURE.md`**: Create a new documentation file explaining the process.
-   [ ] **Document Prerequisites**: List required tools (Azure CLI, Docker Desktop).
-   [ ] **Provide Step-by-Step Instructions**: Clearly explain how to run the `deploy.ps1` script.
-   [ ] **Explain Created Resources**: Briefly describe what the script creates in the user's subscription.
-   [ ] **Add Cleanup Instructions**: Provide the `az group delete` command so users can easily remove all resources and avoid further costs.
