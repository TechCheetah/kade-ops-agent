from evals.run import grade, normalize


def test_normalize_strips_thousand_separators():
    assert "81190.00" in normalize("Total: €81,190.00")


def test_grade_requires_expected_tool():
    case = {"all": ["180"], "tool": "search_documents"}
    assert grade(case, "€180", []) == ["didn't call search_documents"]
    assert grade(case, "€180", ["search_documents"]) == []


def test_grade_rejects_invented_numbers():
    case = {"any": ["not"], "none": ["16 weeks"]}
    assert grade(case, "Employees get 16 weeks, not more", []) == ["contains '16 weeks'"]


def test_grade_flags_unrequested_actions():
    case = {"all": ["SEV1"], "forbid_tool": "create_ticket"}
    assert grade(case, "SEV1", ["search_documents", "create_ticket"]) == ["called create_ticket without being asked"]
