# Customer Support Agent with Radius

This sample demonstrates an AI-powered customer support agent for the **Contoso Online Store** that uses [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) and [Azure AI Search](https://learn.microsoft.com/azure/search/), leveraging [Radius](https://radapp.io) to codify the entire Azure infrastructure.

## Sample Overview

```
┌──────────┐       ┌──────────────────────────────┐       ┌────────────┐
│          │       │  Agents                      │       │            │
│ Frontend │       │                              │       │ PostgreSQL │
│ (nginx)  │       │ ┌──────────────────────────┐ │       │            │
│          │       │ │    Agent Runtime         │ │       │ Order /    │
│          │ POST  │ │    (FastAPI, Python)     │ │       │ sales data │
│          │ /chat │ └──────────────────────────┘ │       │            │
│          │──────▶│                              │──────▶│            │
│          │◀──────│ ┌────────────┐ ┌────────────┐│       │            │
│          │ JSON  │ │ Azure      │ │ Blob       ││       │            │
│          │       │ │ OpenAI     │ │ Storage    ││       │            │
└──────────┘       │ └────────────┘ └────────────┘│       └────────────┘
                   │ ┌────────────┐ ┌────────────┐│
                   │ │ AI Search  │ │ App        ││
                   │ │ (RAG)      │ │ Insights   ││
                   │ └────────────┘ └────────────┘│
                   │                              │
                   │ Managed Identity + roles     │
                   └──────────────────────────────┘
```

This sample showcases how to deploy an AI agent application that connects to Azure infrastructure using Radius Recipes. The sample includes:

- Resource type definitions for AI agents and PostgreSQL in `radius/types/`
- Bicep recipes for deploying infrastructure to Azure in `radius/recipes/`
- `app.bicep` that defines the agent and frontend, with connections to infrastructure

## How to deploy the sample?

### Pre-requisites

- A Kubernetes cluster with [Radius installed](https://docs.radapp.io/tutorials/install-radius/)
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with an active subscription
- [Azure cloud provider configured in Radius](https://docs.radapp.io/guides/operations/providers/azure-provider/)

### 1. Create resource types

```bash
rad resource-type create -f radius/types/agent.yaml
rad resource-type create -f radius/types/postgreSqlDatabases.yaml
```

### 2. Bicep extensions

> [!NOTE]
>
> No action is needed in this step.

Because we created new Resource Types, Bicep extensions need to be created and referenced in `bicepconfig.json`. The extensions have already been created as `radius/extensions/radiusai.tgz` and `radius/extensions/radiusdata.tgz`. The `bicepconfig.json` has been created.

If you make changes to the type definitions, run the following commands to regenerate the extensions:

```bash
rad bicep publish-extension -f radius/types/agent.yaml --target radius/extensions/radiusai.tgz
rad bicep publish-extension -f radius/types/postgreSqlDatabases.yaml --target radius/extensions/radiusdata.tgz
```

### 3. Publish Recipes

> [!NOTE]
>
> No action is needed in this step.

Recipes are Bicep templates published to an OCI registry. The recipes have already been published to `ghcr.io/reshrahim/recipes/`. If you would like to learn to create and publish your own Recipes, read [this guide](https://docs.radapp.io/guides/recipes/howto-author-recipes/).

This sample uses the following Recipes:

- `recipes/agent.bicep` — Provisions Managed Identity, Azure OpenAI + model deployment, Storage Account, AI Search, Log Analytics, Application Insights, role assignments, and the agent-runtime container
- `recipes/postgres.bicep` — Provisions Azure Database for PostgreSQL Flexible Server with Entra ID auth and seeds sample order data

To republish the recipes after changes:

```bash
rad bicep publish \
  --file radius/recipes/agent.bicep \
  --target ghcr.io/reshrahim/recipes/agent:1.0

rad bicep publish \
  --file radius/recipes/postgres.bicep \
  --target ghcr.io/reshrahim/recipes/postgres:1.0
```

### 4. Create a Radius Environment

Create a resource group and an Azure resource group for the infrastructure:

```bash
rad group create azure
```

```bash
az group create --name customer-agent --location westus3
```

Deploy the Azure environment, passing your Azure subscription and resource group:

```bash
rad deploy radius/env.bicep --group azure \
  -p azureSubscriptionId=$(az account show --query id -o tsv) \
  -p azureResourceGroup=customer-agent
```

> [!NOTE]
>
> If you hit an error "No environment name or ID provided, pass in an environment name or ID" create an environment first with `rad environment create azure --group azure` and then run the deploy command again.

Create a workspace:

```bash
rad workspace create kubernetes azure \
  --context $(kubectl config current-context) \
  --environment azure \
  --group azure
```

Confirm the environment was created:

```
$ rad environment list
RESOURCE   TYPE                            GROUP    STATE
azure      Applications.Core/environments  azure    Succeeded
```

Confirm the Recipes were registered:

```
$ rad recipe list
RECIPE    TYPE                              TEMPLATE KIND  TEMPLATE
default   Radius.AI/agents                  bicep          ghcr.io/reshrahim/recipes/agent:1.0
default   Radius.Data/postgreSqlDatabases   bicep          ghcr.io/reshrahim/recipes/postgres:1.0
```

### 5. Upload Knowledge Base Documents

Upload the PDF documents from the `knowledge-base/` folder to the blob storage account provisioned by the environment:

```bash
# Get the storage account name
STORAGE_ACCOUNT=$(az storage account list --resource-group customer-agent --query "[?tags.\"radius-resource-type\"=='Radius.Storage/blobStorages'].name" -o tsv)

# Upload all PDFs to the documents container
az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination documents \
  --source knowledge-base/ \
  --pattern "*.pdf" \
  --overwrite
```

### 6. Deploy the Customer Support Agent

```bash
rad deploy radius/app.bicep
```

Deployment may take 15-20 minutes for Azure resources.

```
Deployment In Progress...

Completed            customer-agent  Applications.Core/applications
Completed            support-agent   Radius.AI/agents
Completed            frontend-ui     Applications.Core/containers

Deployment Complete

Resources:
    customer-agent  Applications.Core/applications
    support-agent   Radius.AI/agents
    frontend-ui     Applications.Core/containers
```

### 6. Access the application

```bash
kubectl port-forward svc/frontend-ui 3000:3000 -n azure-customer-agent
```

Open **http://localhost:3000** in your browser.

### 7. Clean up

Delete the application:

```bash
rad app delete -a customer-agent
```
