"""
Contoso Online Store — Customer Support Agent Runtime
"""

import json
import os
import re
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

# ---------------------------------------------------------------------------
# Configuration (injected via Radius connections or direct env vars)
# ---------------------------------------------------------------------------

# Connection names in Recipe: model, search, storage, identity, insights
AZURE_OPENAI_ENDPOINT = os.getenv("CONNECTION_MODEL_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
AZURE_OPENAI_DEPLOYMENT = os.getenv("CONNECTION_MODEL_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"))
AZURE_SEARCH_ENDPOINT = os.getenv("CONNECTION_SEARCH_ENDPOINT", os.getenv("AZURE_SEARCH_ENDPOINT", ""))
AZURE_SEARCH_INDEX = os.getenv("CONNECTION_SEARCH_INDEX", os.getenv("AZURE_SEARCH_INDEX", ""))
AGENT_NAME = os.getenv("AGENT_NAME", "contoso-support")
AGENT_PROMPT = os.getenv("AGENT_PROMPT", "")
APPINSIGHTS_CONN_STR = os.getenv("CONNECTION_INSIGHTS_CONNECTIONSTRING", os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", ""))
AZURE_CLIENT_ID = os.getenv("CONNECTION_IDENTITY_CLIENTID", "")
AZURE_STORAGE_ENDPOINT = os.getenv("CONNECTION_STORAGE_ENDPOINT", os.getenv("AZURE_STORAGE_ENDPOINT", ""))
AZURE_OPENAI_API_KEY = os.getenv("CONNECTION_MODEL_APIKEY", "")
AZURE_STORAGE_KEY = os.getenv("CONNECTION_STORAGE_KEY", "")
AZURE_SEARCH_API_KEY = os.getenv("CONNECTION_SEARCH_APIKEY", "")
POSTGRES_HOST = os.getenv("CONNECTION_POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("CONNECTION_POSTGRES_PORT", "5432")
POSTGRES_DATABASE = os.getenv("CONNECTION_POSTGRES_DATABASE", "")
POSTGRES_USER = os.getenv("CONNECTION_POSTGRES_USER", "pgadmin")
POSTGRES_PASSWORD = os.getenv("CONNECTION_POSTGRES_PASSWORD", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(AGENT_NAME)

# ---------------------------------------------------------------------------
# Credential — use API keys when available, otherwise DefaultAzureCredential
# ---------------------------------------------------------------------------

credential = None
if not AZURE_OPENAI_API_KEY:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    if AZURE_CLIENT_ID:
        credential = DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID)
    else:
        credential = DefaultAzureCredential()

# ---------------------------------------------------------------------------
# Azure OpenAI Client
# ---------------------------------------------------------------------------

openai_client = None
if AZURE_OPENAI_ENDPOINT:
    if AZURE_OPENAI_API_KEY:
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview",
        )
    elif credential:
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )
    if openai_client:
        logger.info("Azure OpenAI client initialized: %s", AZURE_OPENAI_ENDPOINT)
else:
    logger.warning("AZURE_OPENAI_ENDPOINT not set — running in demo mode")

# ---------------------------------------------------------------------------
# Azure Blob Storage (conversation history)
# ---------------------------------------------------------------------------

blob_container_client = None
if AZURE_STORAGE_ENDPOINT:
    try:
        storage_cred = AZURE_STORAGE_KEY if AZURE_STORAGE_KEY else credential
        blob_service = BlobServiceClient(account_url=AZURE_STORAGE_ENDPOINT, credential=storage_cred)
        blob_container_client = blob_service.get_container_client("conversations")
        # Create container if it doesn't exist
        if not blob_container_client.exists():
            blob_container_client.create_container()
        logger.info("Azure Blob Storage initialized: %s", AZURE_STORAGE_ENDPOINT)
    except Exception as e:
        logger.warning("Blob Storage init failed (non-fatal): %s", e)
        blob_container_client = None

# ---------------------------------------------------------------------------
# Optional: Azure AI Search (RAG)
# ---------------------------------------------------------------------------

search_client = None
if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX:
    try:
        from azure.search.documents import SearchClient
        if AZURE_SEARCH_API_KEY:
            from azure.core.credentials import AzureKeyCredential
            search_cred = AzureKeyCredential(AZURE_SEARCH_API_KEY)
        else:
            search_cred = credential

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=search_cred,
        )
        logger.info("Azure AI Search client initialized: %s", AZURE_SEARCH_ENDPOINT)
    except ImportError:
        logger.warning("azure-search-documents not installed — skipping RAG")

# ---------------------------------------------------------------------------
# PostgreSQL (sales/order data)
# ---------------------------------------------------------------------------

pg_pool = None
if POSTGRES_HOST and POSTGRES_DATABASE:
    try:
        import psycopg_pool
        import psycopg

        pg_conninfo = (
            f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DATABASE} "
            f"user={POSTGRES_USER} sslmode=require"
        )
        pg_kwargs = {}
        if POSTGRES_PASSWORD:
            pg_kwargs["password"] = POSTGRES_PASSWORD
        elif credential:
            def _pg_token():
                tok = credential.get_token("https://ossrdbms-aad.database.windows.net/.default")
                return tok.token
            pg_kwargs["password"] = _pg_token

        pg_pool = psycopg_pool.ConnectionPool(
            conninfo=pg_conninfo,
            kwargs=pg_kwargs,
            min_size=1,
            max_size=5,
            open=True,
        )
        logger.info("PostgreSQL pool initialized: %s/%s", POSTGRES_HOST, POSTGRES_DATABASE)
    except Exception as e:
        logger.warning("PostgreSQL init failed (non-fatal): %s", e)
        pg_pool = None


def query_orders(order_number: str) -> dict | None:
    """Look up an order by order number from the sales database."""
    if not pg_pool:
        return None
    try:
        with pg_pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    "SELECT * FROM orders WHERE order_number = %s",
                    (order_number,),
                )
                return cur.fetchone()
    except Exception as e:
        logger.error("Order query failed: %s", e)
        return None


def query_sales_summary() -> list[dict]:
    """Get a summary of recent sales data."""
    if not pg_pool:
        return []
    try:
        with pg_pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    "SELECT * FROM orders ORDER BY order_date DESC LIMIT 20"
                )
                return cur.fetchall()
    except Exception as e:
        logger.error("Sales query failed: %s", e)
        return []

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Contoso Online Store — Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (backed by blob storage when available)
sessions: dict[str, list[dict]] = {}


def _load_session(session_id: str) -> list[dict]:
    """Load session from blob storage if available."""
    if blob_container_client and session_id not in sessions:
        try:
            blob = blob_container_client.get_blob_client(f"{session_id}.json")
            data = blob.download_blob().readall()
            sessions[session_id] = json.loads(data)
        except Exception:
            pass  # blob doesn't exist yet
    return sessions.get(session_id, [])


def _save_session(session_id: str) -> None:
    """Persist session to blob storage if available."""
    if blob_container_client and session_id in sessions:
        try:
            blob = blob_container_client.get_blob_client(f"{session_id}.json")
            blob.upload_blob(json.dumps(sessions[session_id]), overwrite=True)
        except Exception as e:
            logger.warning("Failed to save session %s: %s", session_id, e)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: list[str] = []
    timestamp: str


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = AGENT_PROMPT or """You are the customer support agent for Contoso Online Store, a popular e-commerce retailer that sells electronics, home goods, clothing, and accessories.

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

IMPORTANT: You MUST only reference information that is explicitly present in the order data provided to you. Never fabricate or guess tracking numbers, item statuses, delivery dates, or any other order details. If the order data does not contain the information the customer is asking about, say so honestly and suggest next steps (e.g., contacting the shipping carrier or checking back later).

Always sign off warmly and ask if there's anything else you can help with."""


def retrieve_knowledge(query: str, top_k: int = 3) -> list[str]:
    if not search_client:
        return []
    try:
        results = search_client.search(
            search_text=query, top=top_k, select=["content", "title"],
        )
        return [
            f"[{r['title']}]: {r['content']}"
            for r in results
            if "content" in r
        ]
    except Exception as e:
        logger.error("Knowledge retrieval failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": AGENT_NAME,
        "model": AZURE_OPENAI_DEPLOYMENT,
        "knowledge_enabled": search_client is not None,
        "storage_enabled": blob_container_client is not None,
        "postgres_enabled": pg_pool is not None,
        "observability_enabled": bool(APPINSIGHTS_CONN_STR),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("Chat [session=%s]: %s", request.session_id, request.message[:100])

    sources = retrieve_knowledge(request.message)
    knowledge_context = ""
    if sources:
        knowledge_context = (
            "\n\nRelevant knowledge base articles:\n" + "\n".join(sources)
        )

    # Look up order data if message mentions an order number
    order_context = ""
    order_match = re.search(r'(?:ord(?:er)?)[- ]?(\d+)', request.message, re.IGNORECASE)
    if order_match:
        order_number = f"ORD-{order_match.group(1)}"
        order_data = query_orders(order_number)
        if order_data:
            order_context = f"\n\nOrder data from database:\n{json.dumps(order_data, default=str)}"
        else:
            order_context = f"\n\nNo order found in database for {order_number}."

    history = _load_session(request.session_id)
    if request.session_id not in sessions:
        sessions[request.session_id] = history

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + knowledge_context + order_context}
    ]
    messages.extend(sessions[request.session_id][-20:])
    messages.append({"role": "user", "content": request.message})

    reply_text = ""
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            reply_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Azure OpenAI call failed: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
    else:
        reply_text = (
            f"Hi there! Thanks for reaching out to **Contoso Online Store** support. 😊\n\n"
            f"You asked: *\"{request.message}\"*\n\n"
            "I'm currently running in **demo mode** — the AI model isn't connected yet. "
            "Once it's set up, I'll be able to help you with:\n\n"
            "- 📦 **Order tracking** — just give me your order number\n"
            "- ↩️ **Returns & exchanges** — easy 30-day returns\n"
            "- 🚚 **Shipping updates** — where's your package?\n"
            "- 💳 **Billing questions** — refunds, charges, payments\n\n"
            "Check back soon!"
        )

    sessions[request.session_id].append(
        {"role": "user", "content": request.message}
    )
    sessions[request.session_id].append(
        {"role": "assistant", "content": reply_text}
    )

    _save_session(request.session_id)

    return ChatResponse(
        reply=reply_text,
        session_id=request.session_id,
        sources=[s.split(":")[0] for s in sources],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve conversation history for a session."""
    history = _load_session(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/")
async def root():
    return {
        "service": "Contoso Online Store — Support Agent",
        "endpoints": {"chat": "POST /chat", "health": "GET /health", "session": "GET /sessions/{id}"},
    }
