import json
import time
from dataclasses import dataclass, field

import anthropic

from .config import EFFORT, FALLBACK_MODELS, MAX_STEPS, MODEL, PRICING
from .tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are the internal operations assistant of Kade Logistics B.V., a freight forwarder with a warehouse in Rotterdam and an office in Amsterdam. Employees from Finance, Operations, IT, Customs and People & Culture ask you about company rules, supplier contracts, invoices and incidents, and ask you to prepare short reports or open tickets.

How to work:
- Ground every factual answer in the tools. Search the documents for policies, contract terms and report figures; query the ERP for invoices. Never answer from general knowledge about how companies usually work.
- When a question needs several facts (for example a contract term plus the invoices it applies to), gather all of them before answering, and do the arithmetic explicitly.
- If the documents don't contain the answer, say so plainly and name what is missing. Don't guess.
- Only create a ticket when the user asks for one. Confirm the ticket ID in your answer.
- Cite the source document for each fact in brackets, like [travel-expense-policy].

Style: answer in the user's language, lead with the answer, keep it short. For reports, use a heading, a short summary and a compact table or list. Amounts in euros with two decimals."""


@dataclass
class Trace:
    model: str
    steps: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    def add_usage(self, model: str, usage) -> float:
        p_in, p_out, p_read, p_write = PRICING.get(model, PRICING["claude-opus-5-5"])
        read = usage.cache_read_input_tokens or 0
        write = usage.cache_creation_input_tokens or 0
        cost = (usage.input_tokens * p_in + usage.output_tokens * p_out + read * p_read + write * p_write) / 1e6
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += read
        self.cache_write_tokens += write
        self.cost_usd += cost
        return cost


@dataclass
class AgentResult:
    answer: str
    stop: str
    trace: Trace
    tools_used: list[str]

    @property
    def complete(self) -> bool:
        return self.stop == "end_turn"


class Agent:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL, effort: str = EFFORT,
                 system_prompt: str = SYSTEM_PROMPT):
        self.client = client or anthropic.Anthropic()
        self.model, self.effort, self.system_prompt = model, effort, system_prompt

    def _call(self, messages: list):
        params = dict(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.model in FALLBACK_MODELS:
            return self.client.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        return self.client.messages.create(**params)

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        trace = Trace(model=self.model)
        # Earlier turns travel as plain text: final answers only, no tool calls or thinking blocks
        messages = [*(history or []), {"role": "user", "content": question}]
        tools_used: list[str] = []
        started = time.perf_counter()
        stop = "max_steps"

        for _ in range(MAX_STEPS):
            t0 = time.perf_counter()
            response = self._call(messages)
            served_by = response.model
            cost = trace.add_usage(served_by, response.usage)
            trace.steps.append({
                "type": "llm",
                "model": served_by,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": response.usage.cache_read_input_tokens or 0,
                "cost_usd": round(cost, 6),
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })

            if response.stop_reason == "refusal":
                stop = "refusal"
                answer = "I can't help with that request."
                break
            if response.stop_reason not in ("end_turn", "stop_sequence", "tool_use", "pause_turn"):
                # max_tokens, model_context_window_exceeded and anything newer: partial at best
                stop = response.stop_reason
                answer = _text(response) or "The answer was cut off. Try a narrower question."
                break

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue

            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                stop = "end_turn"
                answer = _text(response)
                break

            results = []
            for call in calls:
                t1 = time.perf_counter()
                output, is_error = run_tool(call.name, call.input, question)
                tools_used.append(call.name)
                trace.steps.append({
                    "type": "tool",
                    "name": call.name,
                    "input": call.input,
                    "output_preview": output[:600],
                    "is_error": is_error,
                    "latency_ms": round((time.perf_counter() - t1) * 1000),
                })
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": output, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        else:
            answer = "I couldn't finish within the step limit. Try splitting the request."
            created = [s["output_preview"] for s in trace.steps if s.get("name") == "create_ticket" and not s["is_error"]]
            if created:
                answer += " Tickets already created: " + ", ".join(json.loads(c)["ticket_id"] for c in created) + "."

        trace.latency_ms = round((time.perf_counter() - started) * 1000)
        trace.cost_usd = round(trace.cost_usd, 6)
        return AgentResult(answer=answer, stop=stop, trace=trace, tools_used=tools_used)


def _text(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
