from app.retrieval import INDEX, tokenize


def top_doc(query):
    return INDEX.search(query, top_k=1)[0][0].doc_id


def test_tokenize_drops_stopwords_and_keeps_amounts():
    assert tokenize("What is the hotel limit, €180?") == ["hotel", "limit", "€180"]


def test_finds_hotel_limit_in_expense_policy():
    assert top_doc("hotel nightly limit Amsterdam") == "travel-expense-policy"


def test_finds_termination_notice_in_transnordic_contract():
    assert top_doc("TransNordic termination notice") == "supplier-contract-transnordic"


def test_finds_sev1_response_time():
    assert top_doc("SEV1 response time escalation") == "it-incident-runbook"


def test_unrelated_query_returns_nothing():
    assert INDEX.search("quantum chromodynamics") == []
