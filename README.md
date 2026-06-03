# End-to-End GenAI Knowledge Base (RAG)

Ein produktionsreifes Retrieval-Augmented Generation (RAG) System, das es ermöglicht, mit internen Dokumenten zu chatten. Die Architektur ist vollständig containerisiert, verfügt über automatisierte CI/CD-Pipelines und bereitet Cloud-Infrastruktur via Terraform vor.

## Architektur & Tech Stack
* **Backend:** Python, FastAPI, LangChain, Google Gemini API
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit (Modern Chat UI)
* **DevOps:** Docker (Containerization), GitHub Actions (CI/CD Pipeline)
* **Infrastructure as Code (IaC):** Terraform (Azure Setup)

## Features
* Wandelt PDFs in Vektor-Embeddings um (`ingest.py`).
* Isoliertes und sicheres Docker-Backend mit REST-API.
* Automatisierter Build- und Test-Prozess bei jedem Push via GitHub Actions.
* Interaktives Web-Frontend im ChatGPT-Style.

## Lokaler Start

Das System besteht aus einem containerisierten Backend und einem lokalen Streamlit-Frontend. Es kann in wenigen Schritten gestartet werden:

1. **Virtuelle Umgebung aktivieren** 
   Aktiviere zunächst die Python-Umgebung in deinem Terminal, damit alle Abhängigkeiten für Streamlit gefunden werden:
```bash
   source .venv/bin/activate
`````
3. **Backend & Frontend starten**
Starte den Docker-Container für das Backend im Hintergrund und führe direkt im Anschluss das Streamlit-Frontend aus:
```bash
   docker run --rm -d --env-file .env -p 8000:8000 rag-backend:latest && streamlit run frontend.py
`````
