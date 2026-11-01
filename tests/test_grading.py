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


def test_numbers_match_whole_values_only():
    case = {"all": ["180"]}
    assert grade(case, "The limit is €1,800.", []) == ["missing '180'"]
    assert grade(case, "The limit is €180.00 per night.", []) == []


def test_short_words_need_word_boundaries():
    case = {"any": ["no"]}
    assert grade(case, "I do not know.", []) == ["none of ['no']"]
    assert grade(case, "No, economy only.", []) == []


def test_percentages_do_not_match_larger_ones():
    assert grade({"all": ["5%"]}, "a 15% credit", []) == ["missing '5%'"]
