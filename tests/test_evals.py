from pathlib import Path

from evals.calibrate_relevance import (
    ScoredCase,
    evaluate_thresholds,
    parse_thresholds,
    recommend_threshold,
)
from evals.run_eval import ask_question, score_case as score_live_case
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
        "answer": "Danach wird Quarzsand verteilt [Source 1].",
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


def test_live_eval_requires_inline_citations_when_configured():
    passed, failures = score_live_case(
        {"require_citations": True},
        {"answer": "Antwort ohne Quellenverweis.", "sources": []},
    )

    assert passed is False
    assert failures == ["expected an inline [Source N] citation"]


def test_live_eval_rejects_citation_to_unavailable_source():
    passed, failures = score_live_case(
        {"require_citations": True},
        {
            "answer": "Antwort [Source 2].",
            "sources": [{"source": "guide.pdf", "excerpt": "Context"}],
        },
    )

    assert passed is False
    assert failures == ["inline citation references an unavailable source"]


def test_live_eval_sends_optional_bearer_token(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "ok", "sources": []}

    def fake_post(url, json, headers, timeout):
        captured.update(
            url=url,
            json=json,
            headers=headers,
            timeout=timeout,
        )
        return Response()

    monkeypatch.setattr("evals.run_eval.requests.post", fake_post)

    payload = ask_question("http://api/ask", "Question", 10, "test-token")

    assert payload == {"answer": "ok", "sources": []}
    assert captured["headers"] == {"Authorization": "Bearer test-token"}


def test_relevance_calibration_recommends_best_balanced_threshold():
    cases = [
        ScoredCase("answer-1", should_refuse=False, best_score=0.8),
        ScoredCase("answer-2", should_refuse=False, best_score=0.6),
        ScoredCase("unknown", should_refuse=True, best_score=0.4),
    ]

    results = evaluate_thresholds(cases, [0.35, 0.45, 0.65])
    recommendation = recommend_threshold(results)

    assert recommendation.threshold == 0.45
    assert recommendation.correct == 3
    assert recommendation.false_accepts == 0
    assert recommendation.false_refusals == 0


def test_relevance_calibration_parses_sorted_unique_thresholds():
    assert parse_thresholds("0.6, 0.4, 0.6") == [0.4, 0.6]


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
