# 1. Python-Image
FROM python:3.11-slim

# 2.Arbeitsordner im Container setzen
WORKDIR /app

# 3.  ERST die requirements.txt kopieren
COPY requirements.txt .

# 4. Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# 5. Den kompletten restlichen Code (inkl. chroma_db) in den Container kopieren
COPY . .

# 6. dem Container sagen, dass er Port 8000 nach außen öffnen soll (brower, frontend)
EXPOSE 8000

# 7. Der Startbefehl
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
