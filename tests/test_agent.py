from types import SimpleNamespace as NS

from app.agent import Agent


def usage(i=100, o=20):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def message(content, stop, model="claude-haiku-5-5"):
    return NS(content=content, stop_reason=stop, usage=usage(), model=model)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


def test_runs_tool_then_answers():
    client = FakeClient([
        message([NS(type="tool_use", id="t1", name="search_documents", input={"query": "hotel limit"})], "tool_use"),
        message([NS(type="text", text="€180 in Amsterdam [travel-expense-policy].")], "end_turn"),
    ])
    result = Agent(client=client, model="claude-haiku-5-5").run("Hotel limit in Amsterdam?")
    assert result.answer.startswith("€180")
    assert result.tools_used == ["search_documents"]
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and "travel-expense-policy" in tool_result["content"]
    assert [s["type"] for s in result.trace.steps] == ["llm", "tool", "llm"]
    assert result.trace.cost_usd > 0


def test_refusal_is_not_read_as_content():
    client = FakeClient([message([], "refusal")])
    result = Agent(client=client, model="claude-haiku-5-5").run("x")
    assert result.stop == "refusal"


def test_opus_goes_through_fallback_beta():
    client = FakeClient([message([NS(type="text", text="ok")], "end_turn", model="claude-opus-5-5")])
    Agent(client=client, model="claude-opus-5-5").run("hi")
    assert client.calls[0]["fallbacks"] == "default"


def test_step_limit_stops_runaway_loops():
    loop = message([NS(type="tool_use", id="t", name="query_invoices", input={})], "tool_use")
    client = FakeClient([loop] * 8)
    result = Agent(client=client, model="claude-haiku-5-5").run("loop")
    assert result.stop == "max_steps"
