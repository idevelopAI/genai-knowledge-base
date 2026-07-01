import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class IndexStats:
    pdf_count: int
    pages_loaded: int
    chunks_created: int
    data_dir: str
    vector_store: str


def list_pdf_files(data_dir: str) -> list[Path]:
    return sorted(Path(data_dir).glob("*.pdf"))


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        return "document.pdf"
    if not name.lower().endswith(".pdf"):
        return f"{name}.pdf"
    return name


def next_available_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_pdf_bytes(data: bytes, filename: str, data_dir: str) -> Path:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = next_available_path(directory, sanitize_filename(filename))
    destination.write_bytes(data)
    return destination


def build_vector_db(
    *,
    data_dir: str,
    persist_directory: str,
    embedding_model: str,
    google_api_key: str,
    collection_name: str = "langchain",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset_collection: bool = True,
) -> IndexStats:
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No document pages could be loaded from {data_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )

    if reset_collection:
        existing_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        existing_store.delete_collection()

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return IndexStats(
        pdf_count=len(pdf_files),
        pages_loaded=len(docs),
        chunks_created=len(splits),
        data_dir=data_dir,
        vector_store=persist_directory,
    )
