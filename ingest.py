import argparse
import os

from dotenv import load_dotenv

from rag_index import (
    build_vector_db,
    delete_collection,
    get_active_collection_name,
)


load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "langchain")
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
GEMINI_EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)


def build_vector_database(full_rebuild: bool = False):
    if not GOOGLE_API_KEY:
        raise ValueError("Sicherheitsfehler: GOOGLE_API_KEY wurde nicht gefunden!")

    print(f"Lade PDFs aus '{DATA_DIR}'...")
    previous_collection_name = get_active_collection_name(
        CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
    )
    stats = build_vector_db(
        data_dir=DATA_DIR,
        persist_directory=CHROMA_DB_DIR,
        embedding_model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        collection_name=CHROMA_COLLECTION_NAME,
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        reuse_embeddings=not full_rebuild,
    )
    if previous_collection_name != stats.collection_name:
        delete_collection(CHROMA_DB_DIR, previous_collection_name)
    print(
        "Datenbank erfolgreich erstellt: "
        f"{stats.pdf_count} PDFs, {stats.pages_loaded} Seiten, "
        f"{stats.chunks_created} Chunks ({stats.chunks_reused} wiederverwendet)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the PDF vector index.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-embed every PDF instead of reusing unchanged embeddings.",
    )
    build_vector_database(full_rebuild=parser.parse_args().full)
