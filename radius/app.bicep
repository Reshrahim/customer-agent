extension radius
extension radiusAi
extension radiusData
extension radiusStorage

@description('The Radius environment to deploy to')
param environment string

@description('Container registry for the application images')
param registry string = 'ghcr.io/reshrahim'

@description('Image tag')
param tag string = '3.0'

@description('Azure OpenAI model deployment name')
param model string = 'gpt-4.1-mini'

@description('System prompt for the agent')
param prompt string = '''You are the customer support agent for Contoso Online Store, a popular e-commerce retailer that sells electronics, home goods, clothing, and accessories.

Your job is to help customers with:
- **Order status**: Look up orders by order number (e.g., ORD-12345). Provide shipping updates, estimated delivery dates, and tracking info.
- **Returns & exchanges**: Explain the 30-day return policy, walk through the return process, and help initiate returns.
- **Shipping questions**: Standard shipping (5-7 business days), express (2-3 days), overnight available. Free shipping on orders over $50.
- **Billing & payments**: Help with payment issues, refund status, and billing questions. Refunds take 5-10 business days.
- **Product questions**: Help customers find products, compare options, and check availability.

Store policies:
- 30-day return window for most items (electronics have 15-day window)
- Free returns on defective items
- Price match guarantee within 14 days of purchase
- Loyalty members earn 2x points on all purchases

Be friendly, professional, and concise. If you don't have specific order data, provide helpful general guidance and let the customer know what information you'd need to look up their order.

IMPORTANT: You MUST only reference information that is explicitly present in the order data provided to you. Never fabricate or guess tracking numbers, item statuses, delivery dates, or any other order details. If the order data does not contain the information the customer is asking about, say so honestly and suggest next steps.

Always sign off warmly and ask if there's anything else you can help with.'''

resource app 'Applications.Core/applications@2023-10-01-preview' = {
  name: 'contoso-online-store-agent'
  location: 'global'
  properties: {
    environment: environment
  }
}

// ── Existing shared resource ────────────────────────────────
resource postgresql 'Radius.Data/postgreSqlDatabases@2025-08-01-preview' existing = {
// TODO: name: 'contoso-db'
  name: 'postgresql'
}

// ── Existing shared resource ────────────────────────────────
resource blobstorage 'Radius.Storage/blobStorages@2025-08-01-preview' existing = {
  // TODO: name: 'contoso-knowledge-base'
  name: 'blobstorage'
}

// ── Customer Service Agent : agents ─────────────────────────
resource agent 'Radius.AI/agents@2025-08-01-preview' = {
  name: 'support-agent'
  properties: {
    application: app.id
    environment: environment
    prompt: prompt
    model: model
    enableObservability: true
    connections: {
      postgres: {
        source: postgresql.id
      }
      blobstorage: {
        source: blobstorage.id
      }
    }
  }
}

// ── Web App : containers ────────────────────────────────────
resource webapp 'Applications.Core/containers@2023-10-01-preview' = {
  name: 'frontend-ui'
  location: 'global'
  properties: {
    application: app.id
    environment: environment
    container: {
      image: '${registry}/frontend-ui:${tag}'
      ports: {
        http: {
          containerPort: 3000
        }
      }
    }
    connections: {
      agent: {
        source: agent.id
      }
    }
  }
}
