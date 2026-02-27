using './main.bicep'

// Fill in or override via pipeline variables
param appName = 'newsrag'
param imageTag = 'latest'
param azureOpenaiEndpoint = ''
param azureOpenaiApiVersion = '2024-02-01'
param azureOpenaiDeployment = ''

// Secrets — supply via Azure DevOps pipeline variables (secret)
param sqlPassword = ''
param serperApiKey = ''
param azureOpenaiApiKey = ''
