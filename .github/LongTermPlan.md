# Deployment Strategy and Planning TODO

This document outlines the two-phased approach for deploying the RAG application. Plan A focuses on achieving a functional deployment quickly, while Plan B details the steps for a more secure, production-grade architecture.

---

## Plan A: Immediate Goal (Public Ingress)

### Objective
Achieve a reliable and working deployment by exposing the `ollama-app` externally and using its public URL for communication. This bypasses the complexities of internal VNet DNS that have been causing connection issues.

- [x] **Consolidate Infrastructure Code:** All Bicep files (`main.bicep`, `ragApp.bicep`) are in a single `deploy` directory.
- [x] **Configure External Environment:** The `main.bicep` file has been updated to create a Container Apps Environment without a VNet configuration, ensuring it is a true external environment.
- [x] **Configure `ollama-app` for External Ingress:** The `ollama-app` in `main.bicep` is configured with `external: true` to receive a public-facing URL.
- [x] **Update `OLLAMA_BASE_URL`:** The `ragAppModule` in `main.bicep` is configured to pass the public FQDN of the `ollama-app` to the `rag-app` container.
- [x] **Implement Two-Stage Deployment Workflow:** The `azure-deploy.yml` workflow is updated to use a robust two-stage deployment, ensuring the Azure Container Registry is created before the images are built and pushed.
- [ ] **Execute a Clean Deployment:**
    - Ensure the target Azure resource group is empty to allow the new external environment to be created correctly.
    - Push the latest changes to the `rag-app-azure` repository to trigger the CI/CD pipeline.
- [ ] **Verify and Test:**
    - After deployment, confirm that the `ollama-app` has a public URL (without `.internal.` in the name).
    - Perform an end-to-end test by asking a question in the `rag-app` UI to confirm that the connection timeout error is resolved.

---

## Plan B: Production Goal (NAT Gateway Architecture)

### Objective
Enhance security by moving back to a VNet-integrated environment and using a NAT Gateway to provide a static outbound IP. This allows the `ollama-app` to be firewalled, accepting traffic *only* from our application.

- [ ] **Re-introduce VNet and Subnets:**
    - Modify `main.bicep` to create a Virtual Network with two subnets: one for the Container Apps Environment and one for the NAT Gateway.
- [ ] **Deploy NAT Gateway:**
    - Add a `Microsoft.Network/natGateways` resource to `main.bicep`.
    - Create a `Microsoft.Network/publicIPAddresses` resource and associate it with the NAT Gateway.
    - Associate the NAT Gateway with the Container Apps Environment's subnet.
- [ ] **Deploy VNet-Integrated Environment:**
    - Update the `containerAppsEnvironment` resource in `main.bicep` to include the `vnetConfiguration` block, linking it to the dedicated subnet.
- [ ] **Implement Strict Firewall Rule:**
    - Keep `external: true` for the `ollama-app`'s ingress.
    - Update the `ipSecurityRestrictions` block to remove the "Allow all" rule.
    - Add a new rule that explicitly allows traffic **only** from the static public IP address of the NAT Gateway.
- [ ] **Deploy and Test:**
    - Push the updated Bicep file to trigger a deployment of the new, more secure architecture.
    - Perform end-to-end testing to confirm that communication is still successful and that the `ollama-app` is inaccessible from other internet locations.

This phased approach allows us to get the application working reliably first (Plan A) and then build upon that success to implement a more hardened, production-ready architecture (Plan B) when ready.