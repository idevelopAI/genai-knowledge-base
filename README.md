# 🤖 End-to-End GenAI Knowledge Base (RAG)

Ein produktionsreifes Retrieval-Augmented Generation (RAG) System, das es ermöglicht, mit internen Dokumenten zu chatten. Die Architektur ist vollständig containerisiert, verfügt über automatisierte CI/CD-Pipelines und bereitet Cloud-Infrastruktur via Terraform vor.

## 🏗️ Architektur & Tech Stack
* **Backend:** Python, FastAPI, LangChain, Google Gemini API
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit (Modern Chat UI)
* **DevOps:** Docker (Containerization), GitHub Actions (CI/CD Pipeline)
* **Infrastructure as Code (IaC):** Terraform (Azure Setup)

## 🚀 Features
* Wandelt PDFs in Vektor-Embeddings um (`ingest.py`).
* Isoliertes und sicheres Docker-Backend mit REST-API.
* Automatisierter Build- und Test-Prozess bei jedem Push via GitHub Actions.
* Interaktives Web-Frontend im ChatGPT-Style.

## 💻 Lokaler Start
Dank Docker kann das komplette System (Backend + Frontend) mit nur einem Befehl gestartet werden:
```bash
zuerst virtuelle Python-Umgebung im Terminal aktivieren (wenn nicht aktiv ist) damit er Streamlit findet
source .venv/bin/activate
und danach
docker run --rm -d --env-file .env -p 8000:8000 rag-backend:latest && streamlit run frontend.py
