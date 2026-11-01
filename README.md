# Kade Ops Agent

An internal operations agent for a fictional freight forwarder, Kade Logistics B.V. It answers employee questions from company documents and ERP data, writes short reports and opens helpdesk tickets. Every run returns a full trace: model calls, tool calls, tokens, cost and latency.

All company data in `data/` is invented for the demo.

## What it shows

- **Agent loop**: a manual tool-use loop on the Anthropic Python SDK (`app/agent.py`), capped at 8 steps. It handles `refusal`, `max_tokens` and `pause_turn` explicitly and uses server-side model fallback.
- **RAG**: section-level chunking and BM25 retrieval over 8 documents (`app/retrieval.py`), with no vector database because the corpus doesn't need one yet.
- **Integrations**: an ERP invoice query and a helpdesk ticket tool (`app/tools.py`). Schemas are closed, and tool errors go back to the model so it can recover.
- **Cost and observability**: prompt caching, per-call usage priced per model, and a trace for every request.
- **Evals**: 18 graded cases (`evals/cases.json`) checking required facts, invented numbers and tool choice, run across model, effort and prompt configurations (`evals/run.py`).
- **Guards**: per-IP rate limit, daily API budget and input length cap.

## Run

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest httpx
export ANTHROPIC_API_KEY=...
.venv/bin/uvicorn app.main:app --reload       # http://localhost:8000
.venv/bin/python -m pytest -q                  # unit tests, no API calls
.venv/bin/python -m evals.run opus haiku haiku-minimal   # real API calls, costs money
```

Configuration: `AGENT_MODEL` (default `claude-opus-5-5`), `AGENT_EFFORT` (default `medium`), `RATE_LIMIT_PER_HOUR`, `DAILY_BUDGET_USD`, and `AGENT_TODAY` to pin the date for overdue calculations.
