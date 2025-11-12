<#
.SYNOPSIS
    Deploys the RAG application infrastructure to Azure.
.DESCRIPTION
    This script automates the entire deployment process for the RAG application. It prompts the user for necessary inputs,
    creates a resource group, deploys the Bicep template, builds and pushes Docker images, and outputs the final application URL.
    It is designed to be run from the root of the project directory.
.PARAMETER resourcePrefix
    A unique prefix (3-10 characters) used to name all Azure resources. If not provided, the script will prompt for it.
.PARAMETER location
    The Azure region where the resources will be deployed. Defaults to 'eastus'.
.PARAMETER ragAppUsername
    The username for the web application login. Defaults to 'ragapp'.
#>
param (
    [Parameter(Mandatory = $false)]
    [string]$resourcePrefix,

    [Parameter(Mandatory = $false)]
    [string]$location = 'eastus',

    [Parameter(Mandatory = $false)]
    [string]$ragAppUsername = 'ragapp'
)

# --- Helper Functions ---

<#
.SYNOPSIS
    Securely converts a SecureString to a plain text string.
.DESCRIPTION
    This function is required for compatibility with older PowerShell versions that do not have the -AsPlainText parameter.
    It uses .NET marshalling to handle the conversion in memory.
#>
function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][System.Security.SecureString]$SecureString)
    $pointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        return [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}
# --- Main Script Logic ---

Write-Host "Starting the Azure RAG application deployment." -ForegroundColor Cyan
Write-Host "This script will guide you through deploying all necessary resources."
Write-Host "--------------------------------------------------------------------"

# --- 1. Gather and Validate User Inputs ---

# Prompt for Resource Prefix if not provided as a parameter
while ([string]::IsNullOrWhiteSpace($resourcePrefix) -or $resourcePrefix.Length -lt 3 -or $resourcePrefix.Length -gt 10 -or $resourcePrefix -notmatch '^[a-zA-Z0-9]+$') {
    $resourcePrefix = Read-Host -Prompt "Enter a unique Resource Prefix (3-10 alphanumeric characters, e.g., 'rag123')"
    if ([string]::IsNullOrWhiteSpace($resourcePrefix) -or $resourcePrefix.Length -lt 3 -or $resourcePrefix.Length -gt 10 -or $resourcePrefix -notmatch '^[a-zA-Z0-9]+$') {
        Write-Warning "Invalid prefix. It must be 3-10 alphanumeric characters long and contain no special characters."
    }
}

# Prompt for the application password
$ragAppPassword = $null
while ($null -eq $ragAppPassword -or $ragAppPassword.Length -lt 8) {
    $ragAppPassword = Read-Host -Prompt "Enter a password for the application login (at least 8 characters)" -AsSecureString
    if ($ragAppPassword.Length -lt 8) {
        Write-Warning "Password must be at least 8 characters long."
    }
}

Write-Host "`nDeployment parameters confirmed:" -ForegroundColor Green
Write-Host " - Resource Prefix: $resourcePrefix"
Write-Host " - Location:        $location"
Write-Host " - App Username:    $ragAppUsername"
Write-Host " - App Password:    (Provided securely)"
Write-Host "--------------------------------------------------------------------"

# --- 2. Create Resource Group and Deploy Bicep Template ---

$resourceGroupName = "${resourcePrefix}-rg"

Write-Host "Creating resource group '$resourceGroupName' in location '$location'..." -ForegroundColor Cyan
az group create --name $resourceGroupName --location $location --output none

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create resource group. Aborting."
    exit 1
}
Write-Host "Resource group created successfully." -ForegroundColor Green

# --- 3. Build and Push Docker Images ---

$acrName = "${resourcePrefix}acr"

# We need the ACR to exist before we can build and push images.
# We will deploy the Bicep template in two stages.

Write-Host "`nStarting Bicep deployment (Stage 1: Foundational Resources)..." -ForegroundColor Cyan
# In the first stage, we deploy everything EXCEPT the container apps.
# We do this by setting a parameter that the Bicep template will use to conditionally skip them.
az deployment group create `
    --resource-group $resourceGroupName `
    --template-file "deploy/main.bicep" `
    --parameters resourcePrefix=$resourcePrefix `
                 ragAppUsername=$ragAppUsername `
                 ragAppPassword="$(Convert-SecureStringToPlainText -SecureString $ragAppPassword)" `
                 deployApps=false `
    -c

if ($LASTEXITCODE -ne 0) {
    Write-Error "Bicep deployment (Stage 1) failed. Please check the output above for details. Aborting."
    exit 1
}
Write-Host "Bicep Stage 1 deployment completed successfully." -ForegroundColor Green
Write-Host "--------------------------------------------------------------------"

# --- 4. Build and Push Docker Images ---

# Generate a unique tag for this build to ensure Azure pulls the new image.
$uniqueTag = Get-Date -Format 'yyyyMMddHHmmss'
Write-Host "Generated unique image tag: $uniqueTag" -ForegroundColor Yellow

Write-Host "Logging in to Azure Container Registry '$acrName'..." -ForegroundColor Cyan
$acrLoginServer = az acr show --name $acrName --query "loginServer" --output tsv
az acr login --name $acrName

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to log in to ACR. Aborting."
    exit 1
}

# Define image names with the unique tag
$ragAppImage = "$($acrLoginServer)/rag-app:$($uniqueTag)"
$ollamaAppImage = "$($acrLoginServer)/rag-app-ollama:$($uniqueTag)"

Write-Host "`nBuilding and pushing rag-app image: '$ragAppImage'..." -ForegroundColor Cyan
docker build --no-cache -t $ragAppImage -f Dockerfile .
docker push $ragAppImage
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push rag-app image. Aborting."
    exit 1
}

Write-Host "`nBuilding and pushing ollama-app image: '$ollamaAppImage'..." -ForegroundColor Cyan
# Build the ollama image from its specific Dockerfile
docker build -t $ollamaAppImage -f Dockerfile.ollama .
docker push $ollamaAppImage
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push ollama-app image. Aborting."
    exit 1
}


Write-Host "Docker images pushed successfully." -ForegroundColor Green
Write-Host "--------------------------------------------------------------------"

# --- 5. Deploy Container Apps and Get URL ---

Write-Host "`nStarting Bicep deployment (Stage 2: Application Services)..." -ForegroundColor Cyan
# In the second stage, we re-run the deployment, this time with deployApps=true.
# Bicep will see that the other resources already exist and only create the container apps.
az deployment group create `
    --resource-group $resourceGroupName `
    --template-file "deploy/main.bicep" `
    --parameters resourcePrefix=$resourcePrefix `
                 ragAppUsername=$ragAppUsername `
                 ragAppPassword="$(Convert-SecureStringToPlainText -SecureString $ragAppPassword)" `
                 deployApps=true `
                 imageTag=$uniqueTag `
    -c

if ($LASTEXITCODE -ne 0) {
    Write-Error "Bicep deployment (Stage 2) failed. Please check the output above for details. Aborting."
    exit 1
}
Write-Host "Bicep Stage 2 deployment completed successfully." -ForegroundColor Green

$ragAppName = "${resourcePrefix}-rag-app"

Write-Host "Retrieving application URL..." -ForegroundColor Cyan
$appUrl = az containerapp show `
    --name $ragAppName `
    --resource-group $resourceGroupName `
    --query "properties.configuration.ingress.fqdn" `
    --output tsv

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to retrieve application URL. Please check the Azure portal for the '$ragAppName' container app in the '$resourceGroupName' resource group."
    exit 1
}

Write-Host "--------------------------------------------------------------------" -ForegroundColor Green
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "Your application is available at: https://$appUrl" -ForegroundColor White
Write-Host "--------------------------------------------------------------------"