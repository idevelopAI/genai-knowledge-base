import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
STOPWORDS = {
    "aber",
    "auch",
    "das",
    "dem",
    "den",
    "der",
    "die",
    "ein",
    "eine",
    "einer",
    "fuer",
    "ist",
    "mit",
    "nach",
    "sind",
    "und",
    "von",
    "was",
    "wie",
    "wird",
    "werden",
    "welche",
    "welcher",
    "welches",
    "wozu",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(normalize_text(value))
        if token not in STOPWORDS
    ]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return cases


def load_chunks(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_number},
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def rank_chunks(question: str, chunks: list[Document]) -> list[Document]:
    chunk_tokens = [Counter(tokenize(chunk.page_content)) for chunk in chunks]
    document_frequency = Counter()
    for tokens in chunk_tokens:
        document_frequency.update(tokens.keys())

    query_tokens = set(tokenize(question))
    chunk_count = len(chunks)

    def score(index: int) -> float:
        frequencies = chunk_tokens[index]
        return sum(
            (math.log((chunk_count + 1) / (document_frequency[token] + 1)) + 1)
            * (1 + math.log(frequencies[token]))
            for token in query_tokens
            if frequencies[token]
        )

    ranked_indexes = sorted(range(chunk_count), key=score, reverse=True)
    return [chunks[index] for index in ranked_indexes]


def score_case(
    case: dict[str, Any],
    ranked_chunks: list[Document],
    top_k: int,
) -> list[str]:
    failures = []
    selected = ranked_chunks[:top_k]
    expected_source = case["expected_source"]
    matching_chunks = [
        chunk for chunk in selected if chunk.metadata.get("source") == expected_source
    ]

    if not matching_chunks:
        returned_sources = sorted(
            {str(chunk.metadata.get("source")) for chunk in selected}
        )
        failures.append(
            f"expected source {expected_source!r}, got {returned_sources!r}"
        )
        return failures

    matching_context = normalize_text(
        " ".join(chunk.page_content for chunk in matching_chunks)
    )
    for expected_term in case.get("expected_context_terms", []):
        if normalize_text(expected_term) not in matching_context:
            failures.append(f"missing context term {expected_term!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic PDF extraction and retrieval smoke checks."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cases", default="evals/retrieval_cases.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    chunks = load_chunks(Path(args.data_dir), args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("FAIL no document chunks were extracted")
        return 1

    passed = 0
    for case in cases:
        ranked_chunks = rank_chunks(case["question"], chunks)
        failures = score_case(case, ranked_chunks, args.top_k)
        if failures:
            print(f"FAIL {case['id']}: {'; '.join(failures)}")
        else:
            passed += 1
            print(f"PASS {case['id']}")

    print(f"\nSummary: {passed}/{len(cases)} retrieval checks passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
