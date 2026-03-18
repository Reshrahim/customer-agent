extension radius

@description('Radius-provided context for the recipe')
param context object

@description('Azure region for provisioned resources')
param location string = resourceGroup().location

// ── Resolve properties from Radius context ──────────────────

var name = context.resource.name
var containerName = context.resource.properties.?container ?? 'documents'

var uniqueSuffix = uniqueString(context.resource.id, resourceGroup().id)
var storageAccountName = 'st${take(replace(name, '-', ''), 14)}${take(uniqueSuffix, 6)}'

var tags = {
  'radius-resource': name
  'radius-resource-type': 'Radius.Storage/blobStorages'
}

// ── Storage Account ─────────────────────────────────────────

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: containerName
  properties: {
    publicAccess: 'None'
  }
}

// ── Output ──────────────────────────────────────────────────

output result object = {
  values: {
    endpoint: storageAccount.properties.primaryEndpoints.blob
    accountName: storageAccount.name
    accountKey: storageAccount.listKeys().keys[0].value
    container: containerName
  }
  resources: [
    storageAccount.id
  ]
}
