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
