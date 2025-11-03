@description('The location where all resources should be created.')
param location string = resourceGroup().location

@description('A prefix that will be used to name all the Azure resources.')
@minLength(3)
@maxLength(10)
param resourcePrefix string

@description('The address space for the new Virtual Network.')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('The username for the RAG application login.')
@secure()
param ragAppUsername string

@description('The password for the RAG application login. It must be at least 8 characters long.')
@secure()
@minLength(8)
param ragAppPassword string

// Variables for resource naming
var logAnalyticsWorkspaceName = '${resourcePrefix}-logs'
var vnetName = '${resourcePrefix}-vnet'
var containerAppsEnvSubnetName = 'cae-subnet'
var otherSubnetName = 'default-subnet'
var acrName = '${resourcePrefix}acr'
var storageAccountName = '${resourcePrefix}storage'
var fileShareName = 'ragappdata'
var containerAppsEnvName = '${resourcePrefix}-cae'
var ollamaAppName = '${resourcePrefix}-ollama-app'
var ollamaImage = '${acrName}.azurecr.io/rag-app-ollama:latest'
var ragAppName = '${resourcePrefix}-rag-app'
var ragAppImage = '${acrName}.azurecr.io/rag-app:latest'


/*
  RESOURCE DEFINITIONS
  The following sections define the Azure resources required for the application.
*/

// 1. Log Analytics Workspace
// All Container Apps environments require a Log Analytics workspace.
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 2. Virtual Network (VNet)
// The VNet provides a private network for the Container Apps, enhancing security.
resource virtualNetwork 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: containerAppsEnvSubnetName
        properties: {
          addressPrefix: '10.0.0.0/23'
        }
      }
      {
        name: otherSubnetName
        properties: {
          addressPrefix: '10.0.2.0/24'
        }
      }
    ]
  }
}

// 3. Azure Container Registry (ACR)
// The ACR will store the Docker images for the ollama-app and rag-app.
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 4. Azure Storage Account and File Share
// The storage account provides a file share to persist data across container restarts.
resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

// We need to get a reference to the 'default' file service that exists within the storage account.
resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2022-09-01' existing = {
  name: 'default'
  parent: storageAccount
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2022-09-01' = {
  // The parent is the file service, not the storage account itself.
  parent: fileService
  name: fileShareName
}

// 5. Azure Container Apps Environment
// This provides the secure, isolated environment to run our container apps.
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppsEnvName
  location: location
  properties: {
    // This makes the environment internal, meaning apps are only accessible from inside the VNet.
    vnetConfiguration: {
      internal: true
      infrastructureSubnetId: virtualNetwork.properties.subnets[0].id // Reference the 'cae-subnet'
    }
    // Link to the Log Analytics workspace for logging and monitoring.
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// 5a. Container Apps Environment Storage
// This defines the link between the environment and the Azure File Share.
// It is a child resource of the environment, not a property within it.
resource environmentStorage 'Microsoft.App/managedEnvironments/storages@2023-05-01' = {
  parent: containerAppsEnvironment
  name: 'ollama-storage' // This name is referenced by the volume in the ollama-app.
  properties: {
    azureFile: {
      shareName: fileShareName
      accountName: storageAccountName
      accountKey: storageAccount.listKeys().keys[0].value
    }
  }
}

// 6. Ollama Container App
// This app runs the Ollama service. It's internal and uses a file share for model persistence.
resource ollamaApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: ollamaAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    template: {
      // Define the container that will run in the app
      containers: [
        {
          name: 'ollama'
          image: ollamaImage
          resources: {
            cpu: 2
            memory: '4.0Gi'
          }
          // Mount the file share to persist the LLM data
          volumeMounts: [
            {
              volumeName: 'ollama-data-volume'
              mountPath: '/root/.ollama'
            }
          ]
        }
      ]
      // Define the scaling rules for the app
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
      // Define the volume that connects to our Azure File Share
      volumes: [
        {
          name: 'ollama-data-volume' // This name is referenced in volumeMounts
          storageType: 'AzureFile'
          storageName: 'ollama-storage' // This name must match a storage definition below
        }
      ]
    }
    configuration: {
      // Allow the app to pull images from our ACR by providing registry credentials.
      // The password is a secret defined below.
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'acr-password'
        }
      ]
      // Allow the app to be reached from other apps in the environment
      ingress: {
        targetPort: 11434
        transport: 'http'
        // Because the environment is internal, this ingress is also internal.
        // It will not be accessible from the public internet.
      }
      // Define secrets for the app. This includes the ACR password.
      secrets: [
        {
          name: 'acr-password'
          // Get the admin password from the ACR resource.
          value: containerRegistry.listCredentials().passwords[0].value
        }
      ]
    }
  }
}

// 7. RAG App (Frontend)
// This is the main Streamlit application that users will interact with.
resource ragApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: ragAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    template: {
      containers: [
        {
          name: 'rag-app'
          image: ragAppImage
          resources: {
            cpu: 1
            memory: '2.0Gi'
          }
          // Set environment variables for the application
          env: [
            {
              name: 'RAG_APP_USERNAME'
              secretRef: 'rag-app-username' // Reference to a secret defined below
            }
            {
              name: 'RAG_APP_PASSWORD'
              secretRef: 'rag-app-password' // Reference to a secret defined below
            }
            {
              name: 'OLLAMA_BASE_URL'
              // Use the internal FQDN of the ollama-app
              value: 'http://${ollamaApp.properties.configuration.ingress.fqdn}:${ollamaApp.properties.configuration.ingress.targetPort}'
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
        {
          name: 'rag-app-username'
          value: ragAppUsername
        }
        {
          name: 'rag-app-password'
          value: ragAppPassword
        }
        {
          name: 'acr-password'
          // Use the same secret value for the ACR password.
          value: containerRegistry.listCredentials().passwords[0].value
        }
      ]
      ingress: {
        external: true // Make this app accessible from the public internet
        targetPort: 8501
        transport: 'http'
        allowInsecure: true // Allow HTTP traffic, since we're not setting up a custom domain/cert
      }
      // Allow the app to pull images from our ACR
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'acr-password'
        }
      ]
    }
  }
}
