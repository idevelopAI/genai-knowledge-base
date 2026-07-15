from pathlib import Path

from evals.run_eval import score_case as score_live_case
from evals.run_retrieval_eval import (
    load_cases,
    load_chunks,
    rank_chunks,
    score_case as score_retrieval_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_live_eval_checks_answers_sources_and_excerpts():
    case = {
        "min_sources": 1,
        "expected_source_contains": "guide.pdf",
        "expected_answer_contains_any": ["quarzsand", "sand"],
        "expected_excerpt_contains_any": ["aerifizieren"],
    }
    payload = {
        "answer": "Danach wird Quarzsand verteilt.",
        "sources": [
            {
                "source": "garden-guide.pdf",
                "excerpt": "Nach dem Aerifizieren wird der Rasen besandet.",
            }
        ],
    }

    passed, failures = score_live_case(case, payload)

    assert passed is True
    assert failures == []


def test_live_eval_reports_missing_quality_expectations():
    case = {
        "min_sources": 1,
        "expected_source_contains": "guide.pdf",
        "expected_answer_contains_any": ["quarzsand"],
        "expected_excerpt_contains_any": ["aerifizieren"],
    }

    passed, failures = score_live_case(
        case,
        {"answer": "Keine Details.", "sources": []},
    )

    assert passed is False
    assert len(failures) == 4


def test_committed_retrieval_cases_pass_without_external_services():
    cases = load_cases(PROJECT_ROOT / "evals" / "retrieval_cases.jsonl")
    chunks = load_chunks(PROJECT_ROOT / "data", 1000, 200)

    failures = {
        case["id"]: score_retrieval_case(
            case,
            rank_chunks(case["question"], chunks),
            top_k=3,
        )
        for case in cases
    }

    assert failures == {case["id"]: [] for case in cases}
