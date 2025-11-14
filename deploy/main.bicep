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

@description('A flag to control whether the container apps should be deployed. Used for two-stage deployment.')
param deployApps bool = true

@description('The unique tag for the Docker images to be deployed.')
param imageTag string = 'latest' // Default to 'latest' for safety, but script will override.

// Variables for resource naming
var logAnalyticsWorkspaceName = '${resourcePrefix}-logs'
var vnetName = '${resourcePrefix}-vnet'
var containerAppsEnvSubnetName = 'cae-subnet'
var otherSubnetName = 'default-subnet'
var acrName = toLower('${resourcePrefix}acr')
// Storage account names must be globally unique, 3-24 characters long, and use only lowercase letters and numbers.
// Appending a unique string derived from the resource group ID ensures uniqueness.
var storageAccountName = substring(replace(toLower('${resourcePrefix}storage${uniqueString(resourceGroup().id)}'), '-', ''), 0, 24)
var fileShareName = 'ragappdata'
var containerAppsEnvName = '${resourcePrefix}-cae'
var ollamaAppName = '${resourcePrefix}-ollama-app'
var ollamaImage = '${acrName}.azurecr.io/rag-app-ollama:${imageTag}'
var ragAppName = '${resourcePrefix}-rag-app'
var ragAppImage = '${acrName}.azurecr.io/rag-app:${imageTag}'
// This variable safely resolves the OLLAMA_BASE_URL, avoiding the BCP318 warning.

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
      internal: false // This creates an External environment, allowing apps to be exposed to the internet.
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
      accessMode: 'ReadWrite'
    }
  }
}

// 6. Ollama Container App
// This app runs the Ollama service. It's internal and uses a file share for model persistence.
resource ollamaApp 'Microsoft.App/containerApps@2023-05-01' = if (deployApps) {
  name: ollamaAppName
  location: location
  dependsOn: [
    environmentStorage
  ]
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
          // --- FIX: Use a patient startup probe to allow for long model download times ---
          probes: [
            // Startup probe: Gives the container up to 16 minutes to start before being killed.
            // A TCP probe is used because the Ollama server doesn't respond to HTTP GET on '/'.
            // This simply checks if the port is open, which is sufficient to know the service is starting.
            // Total timeout = 60s initial delay + (30s period * 30 failures) = 960s = 16 minutes.
            {
              type: 'startup'
              tcpSocket: {
                port: 11434
              }
              initialDelaySeconds: 60
              periodSeconds: 30
              failureThreshold: 30
            }
            // Liveness probe: After startup, periodically checks if the port is still open.
            // If this fails (e.g., the process crashes), the container is restarted.
            {
              type: 'liveness'
              tcpSocket: {
                port: 11434
              }
              initialDelaySeconds: 60
              periodSeconds: 30
            }
            // Readiness probe: After startup, checks if the app is ready to accept traffic.
            // If this fails, the container is temporarily removed from service discovery until it passes again.
            {
              type: 'readiness'
              tcpSocket: {
                port: 11434
              }
              initialDelaySeconds: 60
              periodSeconds: 30
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
        external: false
        // CRITICAL FIX: Expose port 80 for service discovery, and let ACA handle the mapping to the container's target port.
        // The internal DNS resolver only works on standard ports (80/443).
        targetPort: 11434 // The port the container is listening on.
        transport: 'http' // The protocol for the ingress.
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

// 7. RAG App Module
// Conditionally deploy the RAG frontend app using a separate module.
// This completely isolates the resource and its dependencies, resolving the BCP318 warning.
module ragAppModule 'ragApp.bicep' = if (deployApps) {
  name: 'ragAppDeployment'
  params: {
    location: location
    ragAppName: ragAppName
    containerAppsEnvironmentId: containerAppsEnvironment.id
    ragAppImage: ragAppImage
    ragAppUsername: ragAppUsername
    ragAppPassword: ragAppPassword
    acrLoginServer: containerRegistry.properties.loginServer
    acrUsername: containerRegistry.name
    acrPassword: containerRegistry.listCredentials().passwords[0].value
    // CRITICAL FIX: For internal service discovery, we must use the app's name, not its FQDN.
    // The Container Apps DNS resolver will automatically map this short name to the correct internal IP.
    ollamaBaseUrl: 'http://${ollamaAppName}'
  }
}
