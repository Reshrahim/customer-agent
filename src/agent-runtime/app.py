"""
Contoso Online Store — Customer Support Agent Runtime
"""

import json
import os
import random
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
# Agentic capabilities — write actions, eligibility checks, escalation
# ---------------------------------------------------------------------------

def _ensure_tables():
    """Create returns and support_tickets tables if they don't exist."""
    if not pg_pool:
        return
    try:
        with pg_pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS returns (
                    id SERIAL PRIMARY KEY,
                    return_number VARCHAR(20) UNIQUE NOT NULL,
                    order_number VARCHAR(20) NOT NULL,
                    items JSONB NOT NULL,
                    reason TEXT NOT NULL,
                    status VARCHAR(30) NOT NULL DEFAULT 'Initiated',
                    refund_amount DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS support_tickets (
                    id SERIAL PRIMARY KEY,
                    ticket_number VARCHAR(20) UNIQUE NOT NULL,
                    subject VARCHAR(200) NOT NULL,
                    description TEXT NOT NULL,
                    priority VARCHAR(20) NOT NULL DEFAULT 'Normal',
                    status VARCHAR(30) NOT NULL DEFAULT 'Open',
                    order_number VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("Ensured returns and support_tickets tables exist")
    except Exception as e:
        logger.warning("Table creation failed (non-fatal): %s", e)


def check_return_eligibility(order_number: str) -> dict:
    """Check if an order is eligible for return based on status, date, and policy."""
    order = query_orders(order_number)
    if not order:
        return {"eligible": False, "reason": f"Order {order_number} not found"}

    status = order["status"]
    if status in ("Cancelled", "Return Initiated", "Returned"):
        return {"eligible": False, "reason": f"Order is already {status}", "order": order}
    if status in ("Pending", "Processing"):
        return {
            "eligible": False,
            "reason": "Order hasn't shipped yet. Consider cancelling instead.",
            "can_cancel": True,
            "order": order,
        }

    # Date-based return window
    order_date = order["order_date"]
    if isinstance(order_date, str):
        order_date = datetime.fromisoformat(order_date)
    days_since = (datetime.utcnow() - order_date.replace(tzinfo=None)).days

    items = order.get("items", [])
    electronics_keywords = ["headphone", "speaker", "watch", "phone", "tablet",
                            "laptop", "monitor", "webcam", "keyboard", "camera"]
    has_electronics = any(
        any(kw in item.get("name", "").lower() for kw in electronics_keywords)
        for item in items
    )
    window = 15 if has_electronics else 30

    if days_since > window:
        return {
            "eligible": False,
            "reason": f"Order is {days_since} days old, beyond the {window}-day return window",
            "order": order,
        }

    return {
        "eligible": True,
        "return_window_days": window,
        "days_remaining": window - days_since,
        "has_electronics": has_electronics,
        "order": order,
    }


def cancel_order_in_db(order_number: str, reason: str) -> dict:
    """Cancel an order. Only works for Pending or Processing orders."""
    if not pg_pool:
        return {"success": False, "error": "Database not available"}

    order = query_orders(order_number)
    if not order:
        return {"success": False, "error": f"Order {order_number} not found"}

    if order["status"] not in ("Pending", "Processing"):
        return {
            "success": False,
            "error": f"Cannot cancel — order status is '{order['status']}'. "
                     "Only Pending or Processing orders can be cancelled.",
        }

    try:
        with pg_pool.connection() as conn:
            conn.execute(
                "UPDATE orders SET status = 'Cancelled' WHERE order_number = %s",
                (order_number,),
            )
            conn.commit()
        return {
            "success": True,
            "order_number": order_number,
            "previous_status": order["status"],
            "new_status": "Cancelled",
            "refund_amount": float(order["total_amount"]),
            "message": f"Order {order_number} cancelled. A refund of "
                       f"${order['total_amount']:.2f} will be processed in 5-10 business days.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def initiate_return_in_db(order_number: str, items: list[str], reason: str) -> dict:
    """Create a return record and update order status."""
    if not pg_pool:
        return {"success": False, "error": "Database not available"}

    order = query_orders(order_number)
    if not order:
        return {"success": False, "error": f"Order {order_number} not found"}

    order_items = order.get("items", [])

    # Match requested items to order items
    returned_items = []
    if items:
        for item_name in items:
            for oi in order_items:
                if item_name.lower() in oi["name"].lower():
                    returned_items.append(oi)
                    break
    if not returned_items:
        returned_items = order_items

    refund = sum(i["price"] * i.get("qty", 1) for i in returned_items)
    return_number = f"RET-{random.randint(10000, 99999)}"

    try:
        with pg_pool.connection() as conn:
            conn.execute(
                """INSERT INTO returns (return_number, order_number, items, reason, refund_amount)
                   VALUES (%s, %s, %s, %s, %s)""",
                (return_number, order_number,
                 json.dumps(returned_items, default=str), reason, refund),
            )
            conn.execute(
                "UPDATE orders SET status = 'Return Initiated' WHERE order_number = %s",
                (order_number,),
            )
            conn.commit()
        return {
            "success": True,
            "return_number": return_number,
            "order_number": order_number,
            "items_returned": [i["name"] for i in returned_items],
            "refund_amount": refund,
            "message": f"Return {return_number} created. Ship items back within 14 days. "
                       f"Refund of ${refund:.2f} will be processed after we receive them (5-10 business days).",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_ticket_in_db(subject: str, description: str, priority: str,
                        order_number: str | None = None) -> dict:
    """Escalate to human support by creating a ticket."""
    if not pg_pool:
        return {"success": False, "error": "Database not available"}

    ticket_number = f"TKT-{random.randint(10000, 99999)}"
    try:
        with pg_pool.connection() as conn:
            conn.execute(
                """INSERT INTO support_tickets
                   (ticket_number, subject, description, priority, order_number)
                   VALUES (%s, %s, %s, %s, %s)""",
                (ticket_number, subject, description, priority, order_number),
            )
            conn.commit()
        eta = "1 hour" if priority == "Urgent" else "4 hours" if priority == "High" else "24 hours"
        return {
            "success": True,
            "ticket_number": ticket_number,
            "subject": subject,
            "priority": priority,
            "message": f"Support ticket {ticket_number} created (priority: {priority}). "
                       f"A human agent will follow up within {eta}.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Contoso Online Store — Support Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    _ensure_tables()

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

You have access to tools that let you look up orders, search policies, and TAKE ACTIONS on behalf of customers. You are an agentic assistant — you reason through problems, plan multi-step actions, and execute them.

## Capabilities
- **Order status**: Look up orders, provide shipping updates, tracking info, delivery estimates.
- **Cancellations**: Cancel orders still in Pending or Processing status.
- **Returns**: Check return eligibility, initiate returns, calculate refunds.
- **Knowledge base**: Search store policies on shipping, returns, loyalty program, etc.
- **Escalation**: Create support tickets to hand off to human agents.

## Store policies
- 30-day return window for most items (15-day window for electronics)
- Free returns on defective items
- Price match guarantee within 14 days of purchase
- Loyalty members earn 2x points on all purchases
- Refunds processed in 5-10 business days

## WORKFLOW RULES — you MUST follow these:
1. **Look before you leap**: ALWAYS call lookup_order before taking any action on an order.
2. **Check before returning**: ALWAYS call check_return_eligibility before initiating a return.
3. **Confirm before acting**: Before executing cancel_order or initiate_return, clearly tell the customer what you plan to do (including the refund amount) and ASK for their confirmation. Only call the action tool after they confirm.
4. **Escalate when appropriate**: Create a support ticket when:
   - The issue is too complex for you to resolve
   - The customer is frustrated or asks to speak with a human
   - Financial disputes involve amounts over $500
   - You cannot find the information needed after searching
5. **Never fabricate data**: Only reference information explicitly returned by your tools. If a tool returns an error or no data, say so honestly.
6. **Multi-step reasoning**: For complex requests, break them into steps. Example for "I want to return my headphones from ORD-10001": lookup_order → check_return_eligibility → explain findings & confirm with customer → initiate_return.

Be friendly, professional, and concise. Always sign off warmly and ask if there's anything else you can help with."""


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
# Tool Definitions (OpenAI function calling)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up a customer order by order number. Use this when a customer asks about an order status, tracking, delivery, or mentions an order number like ORD-10001.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number, e.g. ORD-10001."
                    }
                },
                "required": ["order_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": "Check whether an order is eligible for return based on its status, age, and return policy. ALWAYS call this before initiating a return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number to check eligibility for."
                    }
                },
                "required": ["order_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a customer order. Only works for orders in Pending or Processing status. ONLY call this after the customer has confirmed they want to cancel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number to cancel."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation."
                    }
                },
                "required": ["order_number", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return",
            "description": "Initiate a return for a delivered/shipped order. Creates a return record, generates a return number, and calculates the refund. ONLY call after checking eligibility AND getting customer confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number to return."
                    },
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of specific items to return. Pass empty array [] to return all items in the order."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Customer's reason for the return."
                    }
                },
                "required": ["order_number", "items", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Escalate to a human support agent by creating a support ticket. Use when the issue is too complex, the customer is upset, financial disputes exceed $500, or you cannot resolve the problem with available tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Brief subject line for the ticket."
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the issue and what has been tried so far."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Normal", "High", "Urgent"],
                        "description": "Ticket priority level."
                    },
                    "order_number": {
                        "type": "string",
                        "description": "Related order number, if applicable."
                    }
                },
                "required": ["subject", "description", "priority"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the Contoso knowledge base for store policies, shipping info, return procedures, loyalty program, or FAQs. Use when a customer asks about policies or procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what information to find."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_orders",
            "description": "Get a summary of recent orders for analytics or when a customer asks about order volume or activity.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]


def _execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "lookup_order":
        order_number = arguments.get("order_number", "").upper()
        digits = re.sub(r'[^0-9]', '', order_number)
        if digits:
            order_number = f"ORD-{digits}"
        data = query_orders(order_number)
        if data:
            return json.dumps(data, default=str)
        return json.dumps({"error": f"No order found for {order_number}"})

    elif name == "check_return_eligibility":
        order_number = arguments.get("order_number", "").upper()
        digits = re.sub(r'[^0-9]', '', order_number)
        if digits:
            order_number = f"ORD-{digits}"
        return json.dumps(check_return_eligibility(order_number), default=str)

    elif name == "cancel_order":
        order_number = arguments.get("order_number", "").upper()
        digits = re.sub(r'[^0-9]', '', order_number)
        if digits:
            order_number = f"ORD-{digits}"
        reason = arguments.get("reason", "Customer requested cancellation")
        return json.dumps(cancel_order_in_db(order_number, reason), default=str)

    elif name == "initiate_return":
        order_number = arguments.get("order_number", "").upper()
        digits = re.sub(r'[^0-9]', '', order_number)
        if digits:
            order_number = f"ORD-{digits}"
        items = arguments.get("items", [])
        reason = arguments.get("reason", "Customer requested return")
        return json.dumps(initiate_return_in_db(order_number, items, reason), default=str)

    elif name == "create_support_ticket":
        return json.dumps(create_ticket_in_db(
            subject=arguments.get("subject", ""),
            description=arguments.get("description", ""),
            priority=arguments.get("priority", "Normal"),
            order_number=arguments.get("order_number"),
        ), default=str)

    elif name == "search_knowledge_base":
        query = arguments.get("query", "")
        results = retrieve_knowledge(query, top_k=3)
        if results:
            return "\n\n".join(results)
        return "No relevant knowledge base articles found."

    elif name == "get_recent_orders":
        data = query_sales_summary()
        if data:
            return json.dumps(data, default=str)
        return "No order data available."

    return json.dumps({"error": f"Unknown tool: {name}"})


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

    history = _load_session(request.session_id)
    if request.session_id not in sessions:
        sessions[request.session_id] = history

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(sessions[request.session_id][-20:])
    messages.append({"role": "user", "content": request.message})

    reply_text = ""
    tool_sources = []

    if openai_client:
        try:
            # Agentic loop: let the model call tools until it produces a final response
            max_iterations = 5
            for _ in range(max_iterations):
                response = openai_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    tools=TOOLS,
                    temperature=0.7,
                    max_tokens=800,
                )

                choice = response.choices[0]

                # If the model wants to call tools, execute them and loop
                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    # Append the assistant message with tool calls
                    messages.append(choice.message)

                    for tool_call in choice.message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)
                        logger.info("Tool call: %s(%s)", fn_name, fn_args)

                        result = _execute_tool(fn_name, fn_args)
                        tool_sources.append(fn_name)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                    continue  # Loop back to let the model process tool results

                # Model produced a final text response
                reply_text = choice.message.content or ""
                break
            else:
                reply_text = "I'm sorry, I wasn't able to complete your request. Please try again."

        except Exception as e:
            logger.error("Azure OpenAI call failed: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
    else:
        reply_text = (
            f"Hi there! Thanks for reaching out to **Contoso Online Store** support.\n\n"
            f"You asked: *\"{request.message}\"*\n\n"
            "I'm currently running in **demo mode** — the AI model isn't connected yet. "
            "Once it's set up, I'll be able to help you with:\n\n"
            "- Order tracking — just give me your order number\n"
            "- Returns & exchanges — easy 30-day returns\n"
            "- Shipping updates — where's your package?\n"
            "- Billing questions — refunds, charges, payments\n\n"
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
        sources=list(set(tool_sources)),
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
