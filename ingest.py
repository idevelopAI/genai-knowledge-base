import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 1. Lädt heimlich deinen Key aus der .env Datei
load_dotenv()  

# 2. Holt den Key sicher ins Skript
MY_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not MY_API_KEY:
    raise ValueError("Sicherheitsfehler: GOOGLE_API_KEY wurde nicht gefunden!")

# 3. Wir setzen ihn für LangChain wieder in die Umgebung, 
# aber diesmal sicher aus der versteckten Datei!
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

def build_vector_db():
    print("Lade Dokument...")
    loader = PyPDFLoader("sample_document.pdf")
    docs = loader.load()

    print("Teile Text in Chunks (Abschnitte)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Generiere Embeddings und speichere in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Dies erstellt die lokale Vektordatenbank im Ordner ./chroma_db
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Datenbank erfolgreich erstellt!")

if __name__ == "__main__":
    build_vector_db()