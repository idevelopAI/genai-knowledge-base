import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


ACTIVE_COLLECTION_FILENAME = ".active_collection"


@dataclass(frozen=True)
class IndexStats:
    pdf_count: int
    pages_loaded: int
    chunks_created: int
    data_dir: str
    vector_store: str
    collection_name: str


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


def validate_pdf_bytes(data: bytes, max_pages: int) -> int:
    if not data.startswith(b"%PDF-"):
        raise ValueError("Uploaded file does not have a valid PDF signature.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError, ValueError, EOFError) as exc:
        raise ValueError("Uploaded file is not a readable PDF.") from exc

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    try:
        page_count = len(reader.pages)
    except (PdfReadError, OSError, ValueError, EOFError) as exc:
        raise ValueError("Uploaded file is not a readable PDF.") from exc

    if page_count == 0:
        raise ValueError("Uploaded PDF does not contain any pages.")
    if page_count > max_pages:
        raise ValueError(f"Uploaded PDF exceeds the {max_pages}-page limit.")
    return page_count


def load_pdf_documents(pdf_files: list[Path]) -> list[Document]:
    documents = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
        except (PdfReadError, OSError, ValueError, EOFError) as exc:
            raise ValueError(f"Unable to read PDF {pdf_path.name!r}.") from exc

        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDF {pdf_path.name!r} is not supported.")

        try:
            for page_number, page in enumerate(reader.pages):
                documents.append(
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={
                            "source": str(pdf_path),
                            "page": page_number,
                        },
                    )
                )
        except (PdfReadError, OSError, ValueError, EOFError) as exc:
            raise ValueError(
                f"Unable to extract text from PDF {pdf_path.name!r}."
            ) from exc
    return documents


def get_active_collection_name(
    persist_directory: str,
    default_collection_name: str,
) -> str:
    pointer = Path(persist_directory) / ACTIVE_COLLECTION_FILENAME
    try:
        collection_name = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default_collection_name
    return collection_name or default_collection_name


def set_active_collection_name(persist_directory: str, collection_name: str) -> None:
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / ACTIVE_COLLECTION_FILENAME
    temporary_pointer = directory / (
        f"{ACTIVE_COLLECTION_FILENAME}.{uuid.uuid4().hex}.tmp"
    )
    temporary_pointer.write_text(collection_name, encoding="utf-8")
    temporary_pointer.replace(pointer)


def delete_collection(persist_directory: str, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        return


def build_vector_db(
    *,
    data_dir: str,
    persist_directory: str,
    embedding_model: str,
    google_api_key: str,
    collection_name: str = "langchain",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IndexStats:
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    docs = load_pdf_documents(pdf_files)
    if not docs:
        raise ValueError(f"No document pages could be loaded from {data_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    if not splits:
        raise ValueError(f"No extractable text found in PDFs from {data_dir}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )

    versioned_collection_name = f"{collection_name}-{uuid.uuid4().hex}"
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=versioned_collection_name,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    set_active_collection_name(persist_directory, versioned_collection_name)

    return IndexStats(
        pdf_count=len(pdf_files),
        pages_loaded=len(docs),
        chunks_created=len(splits),
        data_dir=data_dir,
        vector_store=persist_directory,
        collection_name=versioned_collection_name,
    )
