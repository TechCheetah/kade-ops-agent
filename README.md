# Kade Ops Agent

An operations assistant for Kade Logistics B.V., a freight forwarder that doesn't exist. Employees ask it about policies, supplier contracts and invoices, and it answers from the company's documents and ERP, writes short reports and opens helpdesk tickets. Each answer comes with a trace of every model call and tool call, with tokens, cost and latency.

Everything in `data/` was invented for the demo.

## How it works

The agent in `app/agent.py` is a tool-use loop written by hand on the Anthropic Python SDK. It stops after 8 steps and handles `refusal`, `max_tokens` and `pause_turn` separately. On Opus and Sonnet, the API retries a refused request on another model. Earlier turns go back to the model as plain text, so follow-up questions keep their context.

Retrieval (`app/retrieval.py`) splits each document into sections and ranks them with BM25. There's no vector database, because 8 documents don't need one.

The tools in `app/tools.py` search and read documents, query ERP invoices and create helpdesk tickets. Their schemas reject extra fields, and a failing tool returns its error to the model instead of crashing the run.

The system prompt and tool definitions are cached. Each call's usage is priced by model and added to the trace.

The eval set (`evals/cases.json`, 21 cases) checks each answer for the facts it must contain, numbers it must not invent, the tool it should have used and tickets it must not open unasked. `evals/run.py` runs it across model, effort and prompt configurations. Haiku 5.5 passed the same cases as Opus 5.5 for about 1/35 of the cost per answer, so the demo runs on Haiku.

Ticket creation is blocked in code unless the user's current message asks for a ticket, so the rule doesn't depend on the prompt alone. The API limits each IP's request rate, caps the daily API spend and rejects long questions and long histories. In production, Vercel's firewall also rate-limits `/api/ask` per IP.

## Run

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest httpx
export ANTHROPIC_API_KEY=...
.venv/bin/uvicorn app.main:app --reload       # http://localhost:8000
.venv/bin/python -m pytest -q                  # unit tests, no API calls
.venv/bin/python -m evals.run opus haiku haiku-minimal   # real API calls, costs money
```

Settings: `AGENT_MODEL` (default `claude-haiku-5-5`), `AGENT_EFFORT` (default `medium`), `RATE_LIMIT_PER_HOUR`, `DAILY_BUDGET_USD`, and `AGENT_TODAY` to fix the date used for overdue invoices.
