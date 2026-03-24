# Customer Support Agent with Radius

This sample demonstrates how to build and deploy an **agentic AI application** using [Radius](https://radapp.io) — an open-source platform that enables developers and platform engineers to define, deploy, and manage cloud-native applications across any infrastructure.

The application is a customer support agent for the fictional **Contoso Online Store**. Unlike a simple chatbot, this agent autonomously reasons about customer requests, decides which tools to use, takes actions (like cancelling orders or initiating returns), and chains multiple operations together — all powered by [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) function calling.

### Why Radius?

Building AI applications today is hard. A developer who just wants to build an AI agent goes through the toil of understanding infrastructure dependencies like Azure OpenAI deployments, configuring AI Search indexes, setting up managed identities with the right RBAC roles, provisioning storage accounts and more over conforming to the requirements of the enterprise. This creates a high barrier to entry for developers and slows down innovation.

Radius solves this by enabling **platform engineers** codify all that infrastructure into reusable **Recipes** that could be shared across an organization. The developer just declares what they need in a simple `app.bicep` (an AI agent, a database, a frontend) and everything else is handled behind the scenes.

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
                   │ API keys + Managed Identity  │       │ (docs)     │
                   └──────────────────────────────┘       └────────────┘
```

## 📁 Repository structure

```
├── knowledge-base/          # Contoso policy PDFs for RAG
├── scripts/setup-azure.sh   # One-command Azure prerequisite setup
├── src/
│   ├── agent-runtime/       # Agentic backend (FastAPI + OpenAI tool calling)
│   └── web/                 # Chat UI frontend (nginx)
└── radius/
    ├── app.bicep             # Application definition (what the developer writes)
    ├── env.bicep             # Environment + shared resources
    ├── types/                # Custom resource type schemas (AI, Data, Storage)
    ├── extensions/           # Generated Bicep extensions (.tgz)
    └── recipes/              # IaC templates (agent, postgres, blobstorage)
```

## 🎯 Goals

By the end of this walkthrough, you will:

- Understand the Radius concepts of **Resource Types**, **Recipes**, **Environments**, and **Applications**
- See how custom abstractions (like `Radius.AI/agents`) simplify the developer experience building agentic applications
- Deploy a fully functional agentic AI application to Azure with Radius

## ✅ Prerequisites

Before you begin, you need:

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with an active subscription
- [Radius CLI](https://docs.radapp.io/tutorials/install-radius/#install-the-radius-cli)
- [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) for accessing the AKS cluster

---

## 🚀 Walkthrough

### Step 1 : Clone the repository

```bash
git clone https://github.com/Reshrahim/customer-agent.git
cd customer-agent
```

### Step 2 : Set up prerequisites in Azure

Run the setup script to create an Azure resource group, AKS cluster, service principal, and register the resource providers required for this application:

```bash
./scripts/setup-azure.sh --location westus3 --resource-group customer-agent --cluster-name customer-agent-aks
```

> [!NOTE] 
> This takes a few minutes (mostly AKS cluster creation).
>  - `--resource-group <name>` name of the resource group to create. The script will create one if it doesn't exist. Defaults to `customer-agent`.
>  - `--location <region>` region for the resource group and AKS cluster. Defaults to `westus3`.
>  - `--cluster-name <your-cluster-name>` name of the AKS cluster. The script will detect it and skip creation if it already exists. Defaults to `customer-agent-aks`.

The script will output the next steps. Service principal credentials are saved to `.azure-sp.env` for use in the next step.

<details>
<summary>What the script does (click to expand)</summary>

1. Registers required Azure resource providers (Storage, PostgreSQL, ContainerInstance, OperationalInsights, Search, CognitiveServices)
2. Creates a resource group
3. Creates an AKS cluster with 1 node (`Standard_B2as_v2`)
4. Creates a service principal with "Owner" role scoped to the resource group
5. Saves credentials to `.azure-sp.env`

</details>

### Step 3 : Install Radius and register the Azure credential

Verify the current kubectl context is set to the AKS cluster created in the previous step:

```bash
kubectl config current-context
```

Install Radius in your AKS cluster:

```bash
rad install kubernetes
```

Verify the installation:

```bash
kubectl get pods -n radius-system
```

You should see all Radius pods running:

```
NAME                READY   STATUS    RESTARTS   AGE
applications-rp      1/1     Running   0          1m
bicep-de             1/1     Running   0          1m
controller           1/1     Running   0          1m
dashboard            1/1     Running   0          1m
dynamic-rp           1/1     Running   0          1m
ucp                  1/1     Running   0          1m
```

### Step 1: Create the Resource Types required by the application

Resource Types are application abstractions that are infrastructure/cloud provider agnostic. They define the properties that developers can set when they declare resources in their `app.bicep`, and they map to recipes that provision the underlying infrastructure

Register all three types with Radius:

```bash
rad resource-type create -f radius/types/agent.yaml
rad resource-type create -f radius/types/postgreSqlDatabases.yaml
rad resource-type create -f radius/types/blobStorages.yaml
```

Each type definition in `radius/types/` is a YAML file that declares:
- A **namespace and type name** (e.g., `Radius.AI/agents`)
- A **schema** describing the properties developers can set (like `prompt`, `model`, `enableObservability`)
- **Read-only properties** that recipes output back (like `agentEndpoint`)

For example, `radius/types/agent.yaml` defines the `Radius.AI/agents` type. A developer using this type only needs to specify a prompt and model name — they don't need to know that behind the scenes, the recipe provisions 8 Azure resources with role assignments and networking.

You can verify the types were created:

```bash
rad resource-type list
```

```
TYPE                                    NAMESPACE                APIVERSION
Applications.Core/applications          Applications.Core        ["2023-10-01-preview"]
...
Radius.AI/agents                        Radius.AI                ["2025-08-01-preview"]
Radius.Data/postgreSqlDatabases         Radius.Data              ["2025-08-01-preview"]
Radius.Storage/blobStorages             Radius.Storage           ["2025-08-01-preview"]
```

### Step 2: Bicep extensions

Bicep extensions are similar to libraries that you import into your code. An extension is generated from a Resource Type definition and contains the necessary metadata and type information for Bicep to provide type safety and autocompletion when developers write their `app.bicep`.

> [!IMPORTANT]
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

<details>
<summary>Regenerating extensions after modifying a type (click to expand)</summary>

If you modify a type definition, regenerate its extension:

```bash
rad bicep publish-extension -f radius/types/agent.yaml --target radius/extensions/radiusai.tgz
rad bicep publish-extension -f radius/types/postgreSqlDatabases.yaml --target radius/extensions/radiusdata.tgz
rad bicep publish-extension -f radius/types/blobStorages.yaml --target radius/extensions/radiusstorage.tgz
```

</details>

### Step 3: Understand the Recipes

A Recipe defines *how* to provision a Resource Type. Recipes are Infrastructure as Code templates, Bicep in this sample, that Radius executes when you deploy a resource of a given type. They receive context from Radius (the resource name, properties, connections) and output infrastructure.

> [!IMPORTANT]
>
> **No action is needed in this step.** The recipes have already been published to `ghcr.io/reshrahim/recipes/`.

This sample has three recipes:

| Recipe | Type | What it provisions |
|--------|------|--------------------|  
| `recipes/agent.bicep` | `Radius.AI/agents` | Azure OpenAI + model deployment, AI Search (index + indexer + data source), Log Analytics + App Insights (for observability), Managed Identity (for deployment scripts), role assignments, and the `agent-runtime` Kubernetes container |
| `recipes/postgres.bicep` | `Radius.Data/postgreSqlDatabases` | Azure Database for PostgreSQL Flexible Server, database, firewall rule, and seeds 5 sample orders |
| `recipes/blobstorage.bicep` | `Radius.Storage/blobStorages` | Azure Storage Account with a blob container |

The key insight is that the **developer never sees these recipes**. They just declare `resource agent 'Radius.AI/agents' = { ... }` in their `app.bicep`, and Radius automatically finds and executes the matching recipe configured by the platform engineer in the environment.

<details>
<summary>Republishing recipes after making changes (click to expand)</summary>

Bicep templates are published to OCI registries (like container images). If you make changes, republish them to your own registry:

```bash
rad bicep publish \
  --file radius/recipes/agent.bicep \
  --target br:ghcr.io/<org-name>/recipes/agent:1.0

rad bicep publish \
  --file radius/recipes/postgres.bicep \
  --target br:ghcr.io/<org-name>/recipes/postgres:1.0

rad bicep publish \
  --file radius/recipes/blobstorage.bicep \
  --target br:ghcr.io/<org-name>/recipes/blobstorage:1.0
```

</details>

### Step 4: Create the Radius Environment

A Radius Environment is where you configure *which* recipes to use and *where* Azure resources should be provisioned.

Create a Radius resource group

```bash
rad group create azure
```

Register the Azure credential using the service principal from Step 2:

```bash
source .azure-sp.env && rad credential register azure sp \
  --client-id $AZURE_CLIENT_ID \
  --client-secret $AZURE_CLIENT_SECRET \
  --tenant-id $AZURE_TENANT_ID
```

Verify the credential is registered:

```bash
rad credential list
```

```
PROVIDER  REGISTERED
azure     true
```

Now deploy the environment. This will create the environment, register the recipes, and provision the shared PostgreSQL and Blob Storage resources:

```bash
source .azure-sp.env
rad deploy radius/env.bicep --group azure \
  -parameters azureSubscriptionId=$AZURE_SUBSCRIPTION_ID \
  -parameters azureResourceGroup=$AZURE_RESOURCE_GROUP
```

> [!NOTE]
>
> This step takes 10 minutes because it provisions an Azure PostgreSQL Flexible Server and an Azure Storage Account.

Create a workspace so the `rad` CLI knows which environment and group to use by default:

```bash
rad workspace create kubernetes azure \
  --context $(kubectl config current-context) \
  --environment azure \
  --group azure
```

Confirm the environment was created:

```bash
rad environment show -o json
```

Confirm the recipes were added to the environment:

```bash
rad recipe list
```

```
RECIPE    TYPE                              TEMPLATE KIND  TEMPLATE
default   Radius.AI/agents                  bicep          ghcr.io/reshrahim/recipes/agent:1.0
default   Radius.Data/postgreSqlDatabases   bicep          ghcr.io/reshrahim/recipes/postgres:1.0
default   Radius.Storage/blobStorages       bicep          ghcr.io/reshrahim/recipes/blobstorage:1.0
```

<details>
<summary>What env.bicep does (click to expand)</summary>

1. Creates the environment with an Azure provider scope (subscription + resource group)
2. Maps each Resource Type to its Recipe template stored in the OCI registry
3. Creates shared PostgreSQL and Blob Storage resources that multiple applications can reference

</details>

### Step 5: Upload knowledge base documents

The agent uses Azure AI Search for knowledge retrieval (RAG). The search index is populated by an indexer that reads documents from the blob storage container. In this step, you upload the Contoso policy documents, so the agent can answer questions about shipping, returns, and the loyalty program.

The `knowledge-base/` folder contains three PDF documents:
- `contoso-shipping-policy.pdf` 
- `contoso-return-refund-policy.pdf` 
- `contoso-loyalty-program.pdf` 

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

Deploy the application:

```bash
rad deploy radius/app.bicep
```

Take a look at `radius/app.bicep`:
- It references shared resources (`postgresql`, `blobstorage`) using the `existing` keyword
- It creates a `Radius.AI/agents` resource with a system prompt, model name, and **connections** to both shared resources
- It creates a `frontend-ui` container connected to the agent

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

```
What's the status of ORD-10001?
```

The agent will call `lookup_order` and respond with the order details from the database.

**Policy question** (knowledge base search):
```
What's your return policy for electronics?
```

The agent will call `search_knowledge_base` and respond using information from the uploaded PDF documents.

**Multi-step tool chaining** (multiple tool calls in sequence):

```
I want to return the headphones from order ORD-10001. Can you help?
```

The agent will:
1. Call `lookup_order` to get the order details
2. Call `check_return_eligibility` to verify the order is within the return window
3. Tell you the eligibility result and ask for confirmation
4. After you confirm, call `initiate_return` to create the return record

**Escalation** (knowing when to hand off):
```
This is unacceptable, I've been waiting 3 weeks and nobody can help me. I want to speak to a manager.
```

The agent will recognize the customer's frustration and call `create_support_ticket` to escalate to a human agent.

### Step 9: Clean up

Delete the application and its resources:

```bash
rad app delete -a customer-agent
```

This deletes the Radius application and recipe-provisioned Azure resources (OpenAI, Search, etc.). The shared environment resources (PostgreSQL, Blob Storage) deployed via `env.bicep` are **not** deleted — they belong to the environment, not the application.

To fully clean up Azure resources, delete the resource group:

 ```bash
az group delete --name customer-agent --yes
```

Uninstall Radius from the AKS cluster if you no longer need it:

```bash
rad uninstall kubernetes --purge  
```