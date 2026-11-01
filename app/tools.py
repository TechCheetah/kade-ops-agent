import itertools
import json
import os
from datetime import date

from .config import DATA_DIR
from .retrieval import DOCUMENTS, INDEX

ERP = json.loads((DATA_DIR / "erp.json").read_text())
_ticket_ids = itertools.count(1042)


def today() -> date:
    return date.fromisoformat(os.environ["AGENT_TODAY"]) if os.getenv("AGENT_TODAY") else date.today()


def search_documents(query: str, top_k: int = 4) -> dict:
    hits = INDEX.search(query, top_k=min(max(top_k, 1), 8))
    return {
        "results": [
            {"doc_id": c.doc_id, "title": c.title, "section": c.section, "score": round(s, 2), "text": c.text}
            for c, s in hits
        ]
    }


def read_document(doc_id: str) -> dict:
    if doc_id not in DOCUMENTS:
        raise ValueError(f"Unknown doc_id '{doc_id}'. Known: {', '.join(DOCUMENTS)}")
    return {"doc_id": doc_id, "text": DOCUMENTS[doc_id]}


def query_invoices(supplier: str | None = None, status: str | None = None, overdue_only: bool = False) -> dict:
    ref = today()
    rows = []
    for inv in ERP["invoices"]:
        if supplier and supplier.lower() not in inv["supplier"].lower():
            continue
        if status and inv["status"] != status:
            continue
        overdue = inv["status"] == "open" and date.fromisoformat(inv["due_date"]) < ref
        if overdue_only and not overdue:
            continue
        days = (ref - date.fromisoformat(inv["due_date"])).days
        rows.append({**inv, "overdue": overdue, "days_overdue": max(days, 0)})
    return {
        "as_of": ref.isoformat(),
        "count": len(rows),
        "total_eur": round(sum(r["amount_eur"] for r in rows), 2),
        "invoices": rows,
    }


TICKETS: list[dict] = []


def create_ticket(department: str, title: str, description: str, priority: str) -> dict:
    ticket = {
        "ticket_id": f"OPS-{next(_ticket_ids)}",
        "department": department,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "created",
    }
    TICKETS.append(ticket)
    return ticket


TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Full-text search over the company's internal documents: policies, supplier contracts, runbooks, "
            "procedures and reports. Returns the best-matching sections with their doc_id. Use it before "
            "answering any question about rules, terms or figures, and search again with other wording if the "
            "first results don't contain the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for, in English."},
                "top_k": {"type": "integer", "description": "Number of sections to return (1-8, default 4)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_document",
        "description": "Return the full text of one document by doc_id, when search results are not enough.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_invoices",
        "description": (
            "Query supplier invoices from the ERP. Filters are optional and combine. Returns each invoice with "
            "amount, dates, status (paid, open, disputed), whether it is overdue as of today and the total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier": {"type": "string", "description": "Part of the supplier name."},
                "status": {"type": "string", "enum": ["paid", "open", "disputed"]},
                "overdue_only": {"type": "boolean", "description": "Only open invoices past their due date."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a helpdesk ticket for a department. Only call this when the user explicitly asks to open, "
            "create or file a ticket or request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "enum": ["Finance", "IT", "People & Culture", "Operations", "Customs"]},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            },
            "required": ["department", "title", "description", "priority"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

HANDLERS = {
    "search_documents": search_documents,
    "read_document": read_document,
    "query_invoices": query_invoices,
    "create_ticket": create_ticket,
}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    try:
        return json.dumps(HANDLERS[name](**args), ensure_ascii=False), False
    except Exception as e:
        return f"Error: {e}", True
