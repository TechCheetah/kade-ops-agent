import json

from app.tools import TOOLS, create_ticket, query_invoices, run_tool


def test_overdue_invoices_as_of_fixed_date(monkeypatch):
    monkeypatch.setenv("AGENT_TODAY", "2026-11-01")
    result = query_invoices(overdue_only=True)
    assert {i["id"] for i in result["invoices"]} == {"INV-24077", "INV-24090", "INV-24095"}
    assert result["total_eur"] == 81190.00


def test_disputed_invoice_is_never_overdue(monkeypatch):
    monkeypatch.setenv("AGENT_TODAY", "2026-12-31")
    assert "INV-24063" not in {i["id"] for i in query_invoices(overdue_only=True)["invoices"]}


def test_supplier_filter_is_case_insensitive():
    assert query_invoices(supplier="transnordic")["count"] == 3


def test_ticket_ids_increment():
    a = create_ticket("IT", "a", "b", "low")
    b = create_ticket("IT", "c", "d", "low")
    assert int(b["ticket_id"][4:]) == int(a["ticket_id"][4:]) + 1


def test_tool_errors_are_returned_not_raised():
    output, is_error = run_tool("read_document", {"doc_id": "nope"})
    assert is_error and "Unknown doc_id" in output


def test_every_tool_has_a_handler_and_closed_schema():
    for tool in TOOLS:
        assert tool["input_schema"]["additionalProperties"] is False
        output, _ = run_tool(tool["name"], {}) if tool["name"] == "query_invoices" else ("{}", False)
        json.loads(output)
