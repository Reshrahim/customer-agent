# Customer Support Agent with Radius

![Radius](https://img.shields.io/badge/Radius-0.55-blue?logo=data:image/svg+xml;base64,) ![Azure](https://img.shields.io/badge/Azure-OpenAI%20%7C%20Search%20%7C%20PostgreSQL-0078D4?logo=microsoftazure) ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)

This sample demonstrates how to build and deploy an **agentic AI application** using [Radius](https://radapp.io) — an open-source platform that enables developers and platform engineers to define, deploy, and manage cloud-native applications across any infrastructure.

The application is a customer support agent for the fictional **Contoso Online Store**. Unlike a simple chatbot, this agent autonomously reasons about customer requests, decides which tools to use, takes actions (like cancelling orders or initiating returns), and chains multiple operations together — all powered by [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) function calling.

### Why

Building AI applications today is hard. A developer who just wants to build a customer support agent goes through the toil of understanding infrastructure dependencies like Azure OpenAI deployments, configuring AI Search indexes, setting up managed identities with the right RBAC roles, provisioning storage accounts, current environments, and more over conforming to the requirements of the enterprise. This creates a high barrier to entry for developers and slows down innovation.

Radius solves this by enabling **platform engineers** codify all that infrastructure into reusable **Recipes**. The developer just declares what they need in a simple `app.bicep` (an AI agent, a database, a frontend) and everything else is handled behind the scenes.

Below is a high-level architecture diagram of the application.

```
┌──────────┐       ┌──────────────────────────────┐       ┌────────────┐
│          │       │  Agent                       │       │            │
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
                   │ │ AI Search  │ │ App        ││       ┌────────────┐
                   │ │ (RAG)      │ │ Insights   ││       │            │
                   │ └────────────┘ └────────────┘│       │ Blob       │
                   │                              │──────▶│ Storage    │
                   │ Managed Identity + roles     │       │ (docs)     │
                   └──────────────────────────────┘       └────────────┘
```

## 📁 Repository structure

>This application is purely vibe coded using Copilot CLI. Please ignore discrepancies in code formatting or structure — the focus is on demonstrating the Radius concepts and developer experience. The repo is organized into three main folders

```
├── knowledge-base/                    # PDF documents for the knowledge base
│   ├── contoso-shipping-policy.pdf
│   ├── contoso-return-refund-policy.pdf
│   └── contoso-loyalty-program.pdf
├── src/
│   ├── agent-runtime/                 # The agentic backend (FastAPI + OpenAI tool calling)
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── web/                           # Chat UI frontend (nginx)
│       ├── index.html
│       └── style.css
└── radius/
    ├── app.bicep                      # Application definition (what the developer writes)
    ├── env.bicep                      # Environment + shared resources
    ├── bicepconfig.json               # Bicep extension references
    ├── types/                         # Custom resource type schemas
    │   ├── agent.yaml                 # Radius.AI/agents
    │   ├── postgreSqlDatabases.yaml   # Radius.Data/postgreSqlDatabases
    │   └── blobStorages.yaml          # Radius.Storage/blobStorages
    ├── extensions/                    # Generated Bicep extensions (.tgz)
    └── recipes/                       # Infrastructure-as-code recipes
        ├── agent.bicep                # Provisions OpenAI, Search, Identity, containers, etc.
        ├── postgres.bicep             # Provisions PostgreSQL + seed data
        └── blobstorage.bicep          # Provisions Storage Account + blob container
```

## 🎯 Goals

By the end of this walkthrough, you will:

- Understand the Radius concepts of **Resource Types**, **Recipes**, **Environments**, and **Applications**
- See how custom abstractions (like `Radius.AI/agents`) simplify the developer experience building agentic applications
- Deploy a fully functional agentic AI application to Azure with Radius

## ✅ Prerequisites

Before you begin, you need:

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with an active subscription
- [Radius CLI](https://docs.radapp.io/tutorials/install-radius/)
- A Kubernetes cluster with [Radius installed](https://docs.radapp.io/tutorials/install-radius/)
  - If you are using an existing k8s cluster, you need to make sure you have cluster-admin permissions to install Radius and deploy applications

---

## 🚀 Walkthrough

### Step 0 : Verify all prerequisites are in place

Verify you are logged into Azure CLI

```bash
az account show
```

Verify Radius CLI is installed

```bash

rad --version
```

you should see an output similar to this:

```
CLI Version Information:
RELEASE   VERSION   BICEP     COMMIT
0.55.0    v0.55.0   0.41.2    4142176e438a8e81b21f5cd41b85f3ab73ed86b2

Control Plane Information:
STATUS     VERSION
Installed  0.55.0
```

Verify Radius is installed in your Kubernetes cluster

```bash
kubectl get pods -n radius-system
```
you should see an output similar to this:

```
NAME                READY   STATUS    RESTARTS   AGE
applications-rp      1/1     Running   0          1m
bicep-de             1/1     Running   0          1m
controller           1/1     Running   0          1m
dashboard            1/1     Running   0          1m 
dynamic-rp           1/1     Running   0          1m
ucp                  1/1     Running   0          1m
```

### Step 1: Create the resource types required by the application

Resource types are application abstractions — essentially teaching Radius what your application needs. They define the properties that developers can set when they declare resources in their `app.bicep`, and they map to recipes that provision the underlying infrastructure.

Each type definition in `radius/types/` is a YAML file that declares:
- A **namespace and type name** (e.g., `Radius.AI/agents`)
- A **schema** describing the properties developers can set (like `prompt`, `model`, `knowledgeBase`)
- **Read-only properties** that recipes output back (like `agentEndpoint`)

For example, `radius/types/agent.yaml` defines the `Radius.AI/agents` type. A developer using this type only needs to specify a prompt and model name — they don't need to know that behind the scenes, the recipe provisions 8 Azure resources with role assignments and networking.

Register all three types with Radius:

```bash
rad resource-type create -f radius/types/agent.yaml
rad resource-type create -f radius/types/postgreSqlDatabases.yaml
rad resource-type create -f radius/types/blobStorages.yaml
```

You can verify the types were created:

```bash
rad resource-type list
```

### Step 2: Bicep extensions

Bicep extensions are similar to your package libraries. An extension is generated from a resource type definition and contains the necessary metadata and type information for Bicep to provide type safety and autocompletion when developers write their `app.bicep`.

> [!NOTE]
>
> **No action is needed in this step.** The extensions have already been generated and are checked into the repo as `.tgz` files in `radius/extensions/`. The `bicepconfig.json` references them.

The `radius/bicepconfig.json` file maps extension names to their local `.tgz` files:

```json
{
  "extensions": {
    "radius": "br:biceptypes.azurecr.io/radius:edge",
    "radiusAi": "./extensions/radiusai.tgz",
    "radiusData": "./extensions/radiusdata.tgz",
    "radiusStorage": "./extensions/radiusstorage.tgz"
  }
}
```

This enables you to write `extension radiusAi` in your `app.bicep` and then use `resource agent 'Radius.AI/agents@2025-08-01-preview'` with full type safety.

If you modify a type definition, regenerate its extension:

```bash
rad bicep publish-extension -f radius/types/agent.yaml --target radius/extensions/radiusai.tgz
rad bicep publish-extension -f radius/types/postgreSqlDatabases.yaml --target radius/extensions/radiusdata.tgz
rad bicep publish-extension -f radius/types/blobStorages.yaml --target radius/extensions/radiusstorage.tgz
```

### Step 3: Understand the Recipes

A recipe defines *how* to provision a resource-type. Recipes are Bicep templates that Radius executes when you deploy a resource of a given type. They receive context from Radius (the resource name, properties, connections) and output infrastructure.

> [!NOTE]
>
> **No action is needed in this step.** The recipes have already been published to `ghcr.io/reshrahim/recipes/`.

This sample has three recipes:

| Recipe | Type | What it provisions |
|--------|------|--------------------|  
| `recipes/agent.bicep` | `Radius.AI/agents` | Managed Identity, Azure OpenAI + model deployment, AI Search (index + indexer + data source), Log Analytics, App Insights, role assignments, and the `agent-runtime` Kubernetes container |
| `recipes/postgres.bicep` | `Radius.Data/postgreSqlDatabases` | Azure Database for PostgreSQL Flexible Server, database, firewall rule, and seeds 5 sample orders |
| `recipes/blobstorage.bicep` | `Radius.Storage/blobStorages` | Azure Storage Account with a blob container |

The key insight is that the **developer never sees these recipes**. They just declare `resource agent 'Radius.AI/agents' = { ... }` in their `app.bicep`, and Radius automatically finds and executes the matching recipe configured by the platform engineer in the environment.

Recipes are published to OCI registries (like container images). If you make changes, republish them:

```bash
rad bicep publish \
  --file radius/recipes/agent.bicep \
  --target br:ghcr.io/reshrahim/recipes/agent:1.0

rad bicep publish \
  --file radius/recipes/postgres.bicep \
  --target br:ghcr.io/reshrahim/recipes/postgres:1.0

rad bicep publish \
  --file radius/recipes/blobstorage.bicep \
  --target br:ghcr.io/reshrahim/recipes/blobstorage:1.0
```

### Step 4: Create the Radius Environment

A Radius Environment is where you configure *which* recipes to use and *where* Azure resources should be provisioned. 

The `radius/env.bicep` file does three things:
1. Creates the environment with an Azure provider scope (subscription + resource group)
2. Registers recipes — mapping each resource type to its recipe template in the OCI registry
3. Creates **shared resources** (PostgreSQL and Blob Storage) that multiple applications can reference

Shared resources are deployed at the environment level because they are long-lived infrastructure that multiple applications can reference.

First, create a Radius resource group and an Azure resource group:

```bash
rad group create azure
```

Now create the Azure resource group where the environment will provision resources:

```bash
az group create --name customer-agent --location <location>
```

You need to grant the Radius control plane permission to deploy resources to this resource group. Run the following command and follow the instructions to assign the "Owner" role to the Radius service principal for the `customer-agent` resource group:

```bash
  az ad sp create-for-rbac --name "radius-sp" --role Owner \
  --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group>
```

```
{
  "appId": "myAppId",
  "displayName": "myServicePrincipalName",
  "password": "myServicePrincipalPassword",
  "tenant": "myTentantId"
}
```

Create a Radius credential with the service principal details:

```bash
rad credential register azure sp --client-id myClientId  --client-secret myClientSecret  --tenant-id myTenantId
``` 

Check if the credential is registered:

```bash
rad credential list
```

Now deploy the environment. This will create the environment, register the recipes, and provision the shared PostgreSQL and Blob Storage resources:

```bash
rad deploy radius/env.bicep --group azure \
  -p azureSubscriptionId=$(az account show --query id -o tsv) \
  -p azureResourceGroup=customer-agent
```

> [!NOTE]
>
> This step takes a 10 minutes because it provisions an Azure PostgreSQL Flexible Server and an Azure Storage Account.

Create a workspace so the `rad` CLI knows which environment and group to use by default:

```bash
rad workspace create kubernetes azure \
  --context $(kubectl config current-context) \
  --environment azure \
  --group azure
```

Confirm the environment was created:

```bash
rad environment list
```

```
RESOURCE   TYPE                            GROUP    STATE
azure      Applications.Core/environments  azure    Succeeded
```

Confirm the recipes were registered:

```bash
rad recipe list
```

```
RECIPE    TYPE                              TEMPLATE KIND  TEMPLATE
default   Radius.AI/agents                  bicep          ghcr.io/reshrahim/recipes/agent:1.0
default   Radius.Data/postgreSqlDatabases   bicep          ghcr.io/reshrahim/recipes/postgres:1.0
default   Radius.Storage/blobStorages       bicep          ghcr.io/reshrahim/recipes/blobstorage:1.0
```

**Recap** You just created a Radius environment scoped to your Azure subscription and resource group, registered three recipes that define how to provision Agents, PostgreSQL databases, and blob storage, and deployed shared PostgreSQL and Storage resources that can be referenced by applications.

### Step 5: Upload knowledge base documents

The agent uses Azure AI Search for knowledge retrieval (RAG). The search index is populated by an indexer that reads documents from the blob storage container. In this step, you upload the Contoso policy documents, so the agent can answer questions about shipping, returns, and the loyalty program.

The `knowledge-base/` folder contains three PDF documents:
- `contoso-shipping-policy.pdf` — Shipping methods, costs, delivery times, address changes
- `contoso-return-refund-policy.pdf` — Return windows, conditions, refund process, exchanges
- `contoso-loyalty-program.pdf` — Points system, tiers, rewards redemption

Upload them to the blob storage account that was provisioned in the previous step:

```bash
# Get the storage account name (provisioned by the blobstorage recipe)
STORAGE_ACCOUNT=$(az storage account list --resource-group customer-agent \
  --query "[?tags.\"radius-resource-type\"=='Radius.Storage/blobStorages'].name" -o tsv)

# Upload all PDFs to the 'documents' container
az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination documents \
  --source knowledge-base/ \
  --pattern "*.pdf" \
  --overwrite
```

Once uploaded, the AI Search indexer (created by the agent recipe in the next step) will automatically index these documents every 5 minutes.

### Step 6: Deploy the application

This is where the developer experience comes together. The `app.bicep` file is what a developer writes — it's simple and declarative. It references the shared PostgreSQL and Blob Storage resources, declares an AI agent with a prompt and model, connects everything together, and adds a frontend UI.

Take a look at `radius/app.bicep`:
- It references shared resources (`postgresql`, `blobstorage`) using the `existing` keyword
- It creates a `Radius.AI/agents` resource with a system prompt, model name, and **connections** to both shared resources
- It creates a `frontend-ui` container connected to the agent

Deploy the application:

```bash
rad deploy radius/app.bicep
```

> [!NOTE]
>
> Deployment takes 15-20 minutes. The agent recipe provisions Azure OpenAI (with a model deployment), AI Search (with index, data source, and indexer), Log Analytics, Application Insights, role assignments, and the agent-runtime container.

```
Deployment In Progress...

Completed            customer-agent  Applications.Core/applications
Completed            postgresql      Radius.Data/postgreSqlDatabases
Completed            blobstorage     Radius.Storage/blobStorages
Completed            support-agent   Radius.AI/agents
Completed            frontend-ui     Applications.Core/containers

Deployment Complete

Resources:
    customer-agent  Applications.Core/applications
    support-agent   Radius.AI/agents
    frontend-ui     Applications.Core/containers
```

### Step 7: Access the application

Port-forward the frontend service to your local machine:

```bash
kubectl port-forward svc/frontend-ui 3000:3000 -n azure-customer-agent
```

Open **http://localhost:3000** in your browser.

### Step 8: Try it out

Here are some things to try that demonstrate the agentic behavior:

**Simple order lookup** (single tool call):
> "What's the status of ORD-10001?"

The agent will call `lookup_order` and respond with the order details from the database.

**Policy question** (knowledge base search):
> "What's your return policy for electronics?"

The agent will call `search_knowledge_base` and respond using information from the uploaded PDF documents.

**Multi-step tool chaining** (multiple tool calls in sequence):
> "I want to return the headphones from order ORD-10001. Can you help?"

The agent will:
1. Call `lookup_order` to get the order details
2. Call `check_return_eligibility` to verify the order is within the return window
3. Tell you the eligibility result and ask for confirmation
4. After you confirm, call `initiate_return` to create the return record

**Order cancellation** (write action with confirmation):
> "I need to cancel order ORD-10005"

The agent will look up the order, confirm it's in "Pending" status, tell you the refund amount, and ask for confirmation before calling `cancel_order`.

**Escalation** (knowing when to hand off):
> "This is unacceptable, I've been waiting 3 weeks and nobody can help me. I want to speak to a manager."

The agent will recognize the customer's frustration and call `create_support_ticket` to escalate to a human agent.

**Combined question** (parallel tool calls):
> "Can you check on order ORD-10004 and also tell me about your loyalty program?"

The agent may call `lookup_order` and `search_knowledge_base` in the same turn, then combine the results into a single response.

### Step 9: Clean up

Delete the application and its resources:

```bash
rad app delete -a customer-agent
```

> [!NOTE]
>
> This deletes the Radius application and its recipe-provisioned Azure resources (OpenAI, Search, etc.). The shared environment resources (PostgreSQL, Blob Storage) deployed via `env.bicep` are **not** deleted — they belong to the environment, not the application.
>
> To fully clean up Azure resources, delete the resource group:
> ```bash
> az group delete --name customer-agent --yes
> ```
