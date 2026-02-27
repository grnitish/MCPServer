// ──────────────────────────────────────────────
// Parameters
// ──────────────────────────────────────────────
@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Base name used to derive resource names')
param appName string = 'newsrag'

@description('Container image tag (set by CI/CD)')
param imageTag string = 'latest'

// Secrets — passed from pipeline / parameter file
@secure()
param sqlPassword string
@secure()
param serperApiKey string
@secure()
param azureOpenaiApiKey string

param azureOpenaiEndpoint string
param azureOpenaiApiVersion string = '2024-02-01'
param azureOpenaiDeployment string

// ──────────────────────────────────────────────
// Azure Container Registry
// ──────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${appName}acr'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// ──────────────────────────────────────────────
// Log Analytics  →  Container Apps Environment
// ──────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ──────────────────────────────────────────────
// MCP Server  Container App  (internal only)
// ──────────────────────────────────────────────
resource mcpServer 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${appName}-mcp-server'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: false          // internal — only reachable by other apps in the env
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password';      value: acr.listCredentials().passwords[0].value }
        { name: 'sql-password';       value: sqlPassword }
        { name: 'serper-api-key';     value: serperApiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'mcp-server'
          image: '${acr.properties.loginServer}/${appName}-mcp-server:${imageTag}'
          resources: { cpu: json('0.5'); memory: '1Gi' }
          env: [
            { name: 'SQL_PASSWORD';  secretRef: 'sql-password' }
            { name: 'SERPER_API';    secretRef: 'serper-api-key' }
            { name: 'MCP_HOST';      value: '0.0.0.0' }
            { name: 'MCP_PORT';      value: '8000' }
          ]
        }
      ]
      scale: { minReplicas: 1; maxReplicas: 3 }
    }
  }
}

// ──────────────────────────────────────────────
// Client / Streamlit  Container App  (external)
// ──────────────────────────────────────────────
resource client 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${appName}-client'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true           // public-facing
        targetPort: 8501
        transport: 'http'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password';          value: acr.listCredentials().passwords[0].value }
        { name: 'azure-openai-api-key';   value: azureOpenaiApiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'client'
          image: '${acr.properties.loginServer}/${appName}-client:${imageTag}'
          resources: { cpu: json('0.5'); memory: '1Gi' }
          env: [
            { name: 'MCP_SERVER_URL';              value: 'http://${mcpServer.name}:8000/mcp' }
            { name: 'AZURE_OPENAI_ENDPOINT';        value: azureOpenaiEndpoint }
            { name: 'AZURE_OPENAI_API_KEY';         secretRef: 'azure-openai-api-key' }
            { name: 'AZURE_OPENAI_API_VERSION';     value: azureOpenaiApiVersion }
            { name: 'AZURE_OPENAI_DEPLOYMENT';      value: azureOpenaiDeployment }
          ]
        }
      ]
      scale: { minReplicas: 1; maxReplicas: 3 }
    }
  }
}

// ──────────────────────────────────────────────
// Outputs
// ──────────────────────────────────────────────
output acrLoginServer string = acr.properties.loginServer
output clientFqdn string = client.properties.configuration.ingress.fqdn
output mcpServerFqdn string = mcpServer.properties.configuration.ingress.fqdn
