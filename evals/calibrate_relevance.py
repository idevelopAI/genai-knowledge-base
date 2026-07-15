import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.run_eval import load_cases
from rag_index import get_active_collection_name


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    should_refuse: bool
    best_score: float | None


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    correct: int
    total: int
    false_accepts: int
    false_refusals: int


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        try:
            threshold = float(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid threshold {item!r}.") from exc
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError("Thresholds must be between 0 and 1.")
        thresholds.append(threshold)
    if not thresholds:
        raise argparse.ArgumentTypeError("Provide at least one threshold.")
    return sorted(set(thresholds))


def evaluate_thresholds(
    cases: list[ScoredCase],
    thresholds: list[float],
) -> list[ThresholdResult]:
    results = []
    for threshold in thresholds:
        false_accepts = 0
        false_refusals = 0
        for case in cases:
            accepted = case.best_score is not None and case.best_score >= threshold
            if case.should_refuse and accepted:
                false_accepts += 1
            elif not case.should_refuse and not accepted:
                false_refusals += 1

        results.append(
            ThresholdResult(
                threshold=threshold,
                correct=len(cases) - false_accepts - false_refusals,
                total=len(cases),
                false_accepts=false_accepts,
                false_refusals=false_refusals,
            )
        )
    return results


def recommend_threshold(results: list[ThresholdResult]) -> ThresholdResult:
    if not results:
        raise ValueError("No threshold results available.")
    return max(
        results,
        key=lambda result: (
            result.correct,
            -result.false_accepts,
            -result.false_refusals,
            -result.threshold,
        ),
    )


def score_cases(
    cases_path: Path,
    vectorstore: Chroma,
    top_k: int,
) -> list[ScoredCase]:
    scored_cases = []
    for case in load_cases(cases_path):
        candidates = vectorstore.similarity_search_with_relevance_scores(
            case["question"],
            k=top_k,
        )
        valid_scores = [score for _, score in candidates if 0 <= score <= 1]
        scored_cases.append(
            ScoredCase(
                case_id=case.get("id", case["question"]),
                should_refuse=bool(case.get("should_refuse")),
                best_score=max(valid_scores, default=None),
            )
        )
    return scored_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the relevance cutoff against labeled live eval cases."
    )
    parser.add_argument("--cases", default="evals/questions.jsonl")
    parser.add_argument(
        "--vector-store",
        default=os.environ.get("CHROMA_DB_DIR", "./chroma_db"),
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("CHROMA_COLLECTION_NAME", "langchain"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "GEMINI_EMBEDDING_MODEL",
            "models/gemini-embedding-001",
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", "3")),
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GOOGLE_API_KEY must be configured in the process environment.")

    collection_name = get_active_collection_name(
        args.vector_store,
        args.collection,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=args.embedding_model,
        google_api_key=api_key,
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=args.vector_store,
        embedding_function=embeddings,
    )
    scored_cases = score_cases(Path(args.cases), vectorstore, args.top_k)
    results = evaluate_thresholds(scored_cases, args.thresholds)
    recommendation = recommend_threshold(results)

    print("Case scores:")
    for case in scored_cases:
        expected = "refuse" if case.should_refuse else "answer"
        score = "none" if case.best_score is None else f"{case.best_score:.4f}"
        print(f"  {case.case_id:<24} expected={expected:<6} best_score={score}")

    print("\nThreshold comparison:")
    for result in results:
        print(
            f"  {result.threshold:.2f}  correct={result.correct}/{result.total} "
            f"false_accepts={result.false_accepts} "
            f"false_refusals={result.false_refusals}"
        )

    print(f"\nRecommended threshold: {recommendation.threshold:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
