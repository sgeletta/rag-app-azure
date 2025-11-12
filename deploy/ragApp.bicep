// This module defines the frontend RAG application.
// It is designed to be called by main.bicep.

@description('The location for the resource.')
param location string

@description('The name of the RAG application.')
param ragAppName string

@description('The resource ID of the parent Container Apps Environment.')
param containerAppsEnvironmentId string

@description('The Docker image for the RAG application.')
param ragAppImage string

@description('The username for the application UI.')
@secure()
param ragAppUsername string

@description('The password for the application UI.')
@secure()
param ragAppPassword string

@description('The login server for the Azure Container Registry.')
param acrLoginServer string

@description('The username for the Azure Container Registry.')
param acrUsername string

@description('The admin password for the Azure Container Registry.')
@secure()
param acrPassword string

@description('The fully qualified base URL for the internal Ollama service.')
param ollamaBaseUrl string

// --- RAG App (Frontend) ---
resource ragApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: ragAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    template: {
      containers: [
        {
          name: 'rag-app'
          image: ragAppImage
          resources: {
            cpu: 1
            memory: '2.0Gi'
          }
          env: [
            {
              name: 'RAG_APP_USERNAME'
              secretRef: 'rag-app-username'
            }
            {
              name: 'RAG_APP_PASSWORD'
              secretRef: 'rag-app-password'
            }
            {
              name: 'OLLAMA_BASE_URL'
              value: ollamaBaseUrl
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
    configuration: {
      secrets: [
        { name: 'rag-app-username', value: ragAppUsername }
        { name: 'rag-app-password', value: ragAppPassword }
        { name: 'acr-password', value: acrPassword }
      ]
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
        allowInsecure: true
      }
      registries: [
        {
          server: acrLoginServer
          username: acrUsername
          passwordSecretRef: 'acr-password'
        }
      ]
    }
  }
}
