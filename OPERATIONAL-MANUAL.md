# Operational Manual for RAG Application

This document provides the essential commands for accessing and monitoring your deployed RAG application on Azure Container Apps.

## 1. Accessing Your Application

Your `rag-app` is now running and has been assigned a public URL by Azure. You can find this URL by running the following command in your terminal:

```bash
# --- Set variables ---
RESOURCE_GROUP="rag-app-rg"

# --- Get the application URL ---
az containerapp show \
  --name "rag-app" \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv
```

This command will output a URL similar to `rag-app.some-region.azurecontainerapps.io`. Copy this URL and paste it into your web browser.

## 2. Monitoring Your Application

If you encounter any issues or want to see what's happening inside your containers, you can easily view their logs.

### View `rag-app` (Streamlit UI) Logs

To view the logs for the main Streamlit application:

```bash
az containerapp logs show --name "rag-app" --resource-group "rag-app-rg" --tail 200
```

### View `ollama-app` (LLM Service) Logs

To view the logs for the Ollama service, which is useful for checking model download progress or errors:

```bash
az containerapp logs show --name "ollama-app" --resource-group "rag-app-rg" --tail 200
```

> **Note**: The `--tail 200` flag shows the last 200 lines. You can remove it to see all logs or change the number.