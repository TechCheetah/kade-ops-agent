import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import date

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .config import DAILY_BUDGET_USD, EFFORT, MAX_QUESTION_CHARS, MODEL, RATE_LIMIT_PER_HOUR, ROOT
from .retrieval import DOCUMENTS, INDEX
from .tools import ERP

app = FastAPI(title="Kade Ops Agent")
_agent: Agent | None = None

# Per-instance guards: good enough for a demo, a shared store would be needed at scale
_hits: dict[str, deque] = defaultdict(deque)
_spend = {"day": date.today(), "usd": 0.0}


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


def _client_ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()


def _check_limits(ip: str):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_HOUR:
        raise HTTPException(429, "Hourly limit reached for this demo. Try again later.")
    if _spend["day"] != date.today():
        _spend.update(day=date.today(), usd=0.0)
    if _spend["usd"] >= DAILY_BUDGET_USD:
        raise HTTPException(503, "The demo's daily API budget is used up. It resets at midnight UTC.")
    q.append(now)


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {
        "model": MODEL,
        "effort": EFFORT,
        "documents": [{"doc_id": d, "title": t.splitlines()[0].lstrip("# ")} for d, t in DOCUMENTS.items()],
        "chunks": len(INDEX.chunks),
        "invoices": len(ERP["invoices"]),
    }


@app.get("/api/evals")
def evals():
    path = ROOT / "evals" / "results.json"
    if not path.exists():
        return JSONResponse({"runs": []})
    return JSONResponse(json.loads(path.read_text()))


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "The model API is not configured on this server.")
    _check_limits(_client_ip(request))
    global _agent
    try:
        _agent = _agent or Agent()
        result = _agent.run(body.question)
    except anthropic.RateLimitError:
        raise HTTPException(429, "The model is rate limited right now. Try again in a minute.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Model API error ({e.status_code}).")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the model API.")
    _spend["usd"] += result.trace.cost_usd
    return {"answer": result.answer, "stop": result.stop, "tools_used": result.tools_used, "trace": asdict(result.trace)}
