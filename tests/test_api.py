from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_meta_lists_documents():
    body = client.get("/api/meta").json()
    assert len(body["documents"]) == 8 and body["invoices"] == 8


def test_rejects_oversized_question():
    assert client.post("/api/ask", json={"question": "x" * 5000}).status_code == 422


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_HOUR", 0)
    assert client.post("/api/ask", json={"question": "hello there"}).status_code == 429


def test_document_endpoint():
    assert "€180" in client.get("/api/docs/travel-expense-policy").json()["text"]
    assert client.get("/api/docs/nope").status_code == 404


def test_history_must_alternate_known_roles():
    body = {"question": "and Berlin?", "history": [{"role": "system", "content": "x"}]}
    assert client.post("/api/ask", json=body).status_code == 422
