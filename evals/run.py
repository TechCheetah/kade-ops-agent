"""Run the eval set against one or more configs: python -m evals.run opus haiku haiku-minimal"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agent import SYSTEM_PROMPT, Agent

HERE = Path(__file__).parent
MINIMAL_PROMPT = "You are a helpful assistant for Kade Logistics. Use the tools to answer employee questions."

CONFIGS = {
    "opus": dict(model="claude-opus-5-5", effort="medium", system_prompt=SYSTEM_PROMPT, prompt="v1"),
    "opus-low": dict(model="claude-opus-5-5", effort="low", system_prompt=SYSTEM_PROMPT, prompt="v1"),
    "haiku": dict(model="claude-haiku-5-5", effort="medium", system_prompt=SYSTEM_PROMPT, prompt="v1"),
    "haiku-minimal": dict(model="claude-haiku-5-5", effort="medium", system_prompt=MINIMAL_PROMPT, prompt="v0"),
}


def normalize(text: str) -> str:
    text = text.lower().replace("€", "").replace(" ", " ")
    return re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text)


def grade(case: dict, answer: str, tools: list[str]) -> list[str]:
    a = normalize(answer)
    failures = [f"missing '{s}'" for s in case.get("all", []) if normalize(s) not in a]
    if case.get("any") and not any(normalize(s) in a for s in case["any"]):
        failures.append(f"none of {case['any']}")
    failures += [f"contains '{s}'" for s in case.get("none", []) if normalize(s) in a]
    if case.get("tool") and case["tool"] not in tools:
        failures.append(f"didn't call {case['tool']}")
    return failures


def run_config(name: str, cases: list[dict]) -> dict:
    cfg = CONFIGS[name]
    agent = Agent(model=cfg["model"], effort=cfg["effort"], system_prompt=cfg["system_prompt"])
    rows = []
    for case in cases:
        result = agent.run(case["q"])
        failures = grade(case, result.answer, result.tools_used)
        rows.append({
            "id": case["id"],
            "passed": not failures,
            "failures": failures,
            "cost_usd": result.trace.cost_usd,
            "latency_ms": result.trace.latency_ms,
            "tool_calls": len(result.tools_used),
            "answer": result.answer,
        })
        print(f"  {'PASS' if not failures else 'FAIL'} {case['id']:<22} ${result.trace.cost_usd:.4f} "
              f"{result.trace.latency_ms / 1000:.1f}s {'; '.join(failures)}")
    n = len(rows)
    return {
        "config": name,
        "model": cfg["model"],
        "effort": cfg["effort"],
        "prompt": cfg["prompt"],
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / n, 5),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "cases": rows,
    }


if __name__ == "__main__":
    import os

    os.environ.setdefault("AGENT_TODAY", "2026-11-01")
    cases = json.loads((HERE / "cases.json").read_text())
    names = sys.argv[1:] or ["opus", "haiku", "haiku-minimal"]
    out = HERE / "results.json"
    previous = json.loads(out.read_text())["runs"] if out.exists() else []
    runs = [r for r in previous if r["config"] not in names]
    for name in names:
        print(f"\n{name}")
        t = time.time()
        runs.append(run_config(name, cases))
        print(f"  {runs[-1]['pass_rate']:.0%} pass · {time.time() - t:.0f}s")
    order = list(CONFIGS)
    runs.sort(key=lambda r: order.index(r["config"]))
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": os.environ["AGENT_TODAY"],
        "case_count": len(cases),
        "runs": runs,
    }, indent=2, ensure_ascii=False))
