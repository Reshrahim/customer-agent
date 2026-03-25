extension radius
extension radiusData
extension radiusStorage

resource env 'Applications.Core/environments@2023-10-01-preview' existing = {
  name: 'azure'
}

resource postgresql 'Radius.Data/postgreSqlDatabases@2025-08-01-preview' = {
  name: 'contoso-db'
  properties: {
    environment: env.id
    size: 'S'
  }
}

resource blobstorage 'Radius.Storage/blobStorages@2025-08-01-preview' = {
  name: 'contoso-knowledge-base'
  properties: {
    environment: env.id
    container: 'documents'
  }
}
