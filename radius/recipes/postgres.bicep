extension radius

@description('Radius-provided context for the recipe')
param context object

@description('Azure region for provisioned resources')
param location string = resourceGroup().location

@secure()
@description('Auto-generated admin password used only for seeding')
param administratorPassword string = newGuid()

// ── Resolve properties from Radius context ──────────────────

var name = context.resource.name
var database = context.resource.properties.?database ?? 'postgres_db'
var sizeValue = context.resource.properties.?size ?? 'S'
var port = 5432

var uniqueSuffix = uniqueString(context.resource.id, resourceGroup().id)
var serverName = '${name}-${take(uniqueSuffix, 6)}'

var skuMap = {
  S: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
    storageSizeGB: 32
  }
  M: {
    name: 'Standard_D2ds_v5'
    tier: 'GeneralPurpose'
    storageSizeGB: 64
  }
  L: {
    name: 'Standard_D4ds_v5'
    tier: 'GeneralPurpose'
    storageSizeGB: 128
  }
}

var tags = {
  'radius-resource': name
  'radius-resource-type': 'Radius.Data/postgreSqlDatabases'
}

// ── Azure Database for PostgreSQL Flexible Server ───────────

resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: serverName
  location: location
  tags: tags
  sku: {
    name: skuMap[sizeValue].name
    tier: skuMap[sizeValue].tier
  }
  properties: {
    version: '16'
    administratorLogin: 'pgadmin'
    administratorLoginPassword: administratorPassword
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Enabled'
    }
    storage: {
      storageSizeGB: skuMap[sizeValue].storageSizeGB
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

resource db 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgresServer
  name: database
}

// ── Firewall rule: allow Azure services ─────────────────────

resource allowAzureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: postgresServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ── Seed sales data ─────────────────────────────────────────

resource seedData 'Microsoft.Resources/deploymentScripts@2023-08-01' = {
  name: '${name}-seed-data'
  location: location
  kind: 'AzureCLI'
  dependsOn: [
    db
    allowAzureServices
  ]
  properties: {
    azCliVersion: '2.63.0'
    retentionInterval: 'PT1H'
    timeout: 'PT10M'
    environmentVariables: [
      { name: 'PGHOST', value: postgresServer.properties.fullyQualifiedDomainName }
      { name: 'PGPORT', value: '${port}' }
      { name: 'PGDATABASE', value: database }
      { name: 'PGUSER', value: 'pgadmin' }
      { name: 'PGPASSWORD', secureValue: administratorPassword }
      { name: 'PGSSLMODE', value: 'require' }
    ]
    scriptContent: '''#!/bin/bash
set -e

apk add --no-cache postgresql-client > /dev/null 2>&1 || true

echo "Creating orders table and seeding data..."
psql -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS orders (
  id SERIAL PRIMARY KEY,
  order_number VARCHAR(20) UNIQUE NOT NULL,
  customer_name VARCHAR(100) NOT NULL,
  customer_email VARCHAR(100) NOT NULL,
  order_date TIMESTAMP NOT NULL,
  status VARCHAR(30) NOT NULL,
  total_amount DECIMAL(10,2) NOT NULL,
  shipping_method VARCHAR(30) NOT NULL,
  tracking_number VARCHAR(50),
  estimated_delivery DATE,
  items JSONB NOT NULL
);

INSERT INTO orders (order_number, customer_name, customer_email, order_date, status, total_amount, shipping_method, tracking_number, estimated_delivery, items)
VALUES
  ('ORD-10001', 'Alice Johnson', 'alice@example.com', '2026-03-10 14:30:00', 'Shipped', 129.99, 'Standard', 'TRK-98765432', '2026-03-17', '[{"name": "Wireless Headphones", "qty": 1, "price": 79.99}, {"name": "Phone Case", "qty": 1, "price": 19.99}, {"name": "USB-C Cable", "qty": 2, "price": 15.00}]'),
  ('ORD-10002', 'Bob Smith', 'bob@example.com', '2026-03-12 09:15:00', 'Processing', 249.50, 'Express', NULL, '2026-03-15', '[{"name": "Bluetooth Speaker", "qty": 1, "price": 149.50}, {"name": "Smart Watch Band", "qty": 2, "price": 50.00}]'),
  ('ORD-10003', 'Carol Davis', 'carol@example.com', '2026-03-14 16:45:00', 'Delivered', 89.99, 'Standard', 'TRK-11223344', '2026-03-19', '[{"name": "Yoga Mat", "qty": 1, "price": 49.99}, {"name": "Water Bottle", "qty": 2, "price": 20.00}]'),
  ('ORD-10004', 'Dan Lee', 'dan@example.com', '2026-03-15 11:00:00', 'Shipped', 599.00, 'Overnight', 'TRK-55667788', '2026-03-16', '[{"name": "Laptop Stand", "qty": 1, "price": 89.00}, {"name": "Mechanical Keyboard", "qty": 1, "price": 159.00}, {"name": "Monitor Light Bar", "qty": 1, "price": 69.00}, {"name": "Webcam HD", "qty": 1, "price": 119.00}, {"name": "Desk Pad", "qty": 1, "price": 39.00}, {"name": "Cable Management Kit", "qty": 1, "price": 24.00}]'),
  ('ORD-10005', 'Eve Martinez', 'eve@example.com', '2026-03-16 08:30:00', 'Pending', 42.50, 'Standard', NULL, NULL, '[{"name": "Notebook Set", "qty": 1, "price": 22.50}, {"name": "Pen Pack", "qty": 1, "price": 12.00}, {"name": "Sticky Notes", "qty": 1, "price": 8.00}]')
ON CONFLICT (order_number) DO NOTHING;

SQL
echo "Seed data complete. $(psql -t -c 'SELECT count(*) FROM orders;') orders in table."
'''
  }
}

// ── Output ──────────────────────────────────────────────────

output result object = {
  values: {
    host: postgresServer.properties.fullyQualifiedDomainName
    port: port
    database: database
    user: 'pgadmin'
    password: administratorPassword
  }
  resources: [
    postgresServer.id
  ]
}
