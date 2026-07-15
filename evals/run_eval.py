import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


UNKNOWN_MARKERS = [
    "ich weiss es nicht",
    "ich weis es nicht",
    "nicht im kontext",
    "nicht in den bereitgestellten",
    "i do not know",
    "i don't know",
    "not provided",
]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return cases


def ask_question(api_url: str, question: str, timeout: int) -> dict[str, Any]:
    response = requests.post(api_url, json={"question": question}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def score_case(case: dict[str, Any], payload: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])

    if not answer.strip():
        failures.append("empty answer")

    min_sources = case.get("min_sources")
    if min_sources is not None and len(sources) < min_sources:
        failures.append(f"expected at least {min_sources} sources, got {len(sources)}")

    expected_source = case.get("expected_source_contains")
    if expected_source:
        source_names = " ".join(source.get("source", "") for source in sources)
        if expected_source.lower() not in source_names.lower():
            failures.append(f"missing expected source containing {expected_source!r}")

    normalized_answer = answer.lower()
    expected_answer_terms = case.get("expected_answer_contains_any", [])
    if expected_answer_terms and not any(
        term.lower() in normalized_answer for term in expected_answer_terms
    ):
        failures.append(
            f"answer missing all expected terms {expected_answer_terms!r}"
        )

    expected_excerpt_terms = case.get("expected_excerpt_contains_any", [])
    source_excerpts = " ".join(
        source.get("excerpt", "") for source in sources
    ).lower()
    if expected_excerpt_terms and not any(
        term.lower() in source_excerpts for term in expected_excerpt_terms
    ):
        failures.append(
            f"source excerpts missing all expected terms {expected_excerpt_terms!r}"
        )

    if case.get("should_refuse"):
        if not any(marker in normalized_answer for marker in UNKNOWN_MARKERS):
            failures.append("expected refusal/unknown answer")

    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simple RAG quality checks.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--cases", default="evals/questions.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    passed = 0

    for case in cases:
        case_id = case.get("id", case["question"])
        try:
            payload = ask_question(args.api_url, case["question"], args.timeout)
            ok, failures = score_case(case, payload)
        except requests.exceptions.RequestException as exc:
            ok = False
            failures = [f"request failed: {exc}"]

        if ok:
            passed += 1
            print(f"PASS {case_id}")
        else:
            print(f"FAIL {case_id}: {'; '.join(failures)}")

    total = len(cases)
    print(f"\nSummary: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
