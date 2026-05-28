# 1. Wir nutzen ein schlankes, offizielles Python-Image
FROM python:3.11-slim

# 2. Wir setzen den Arbeitsordner im Container
WORKDIR /app

# 3. Wir kopieren ERST die requirements.txt
COPY requirements.txt .

# 4. Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# 5. Den kompletten restlichen Code (inkl. chroma_db) in den Container kopieren
COPY . .

# 6. Wir sagen dem Container, dass er Port 8000 nach außen öffnen soll
EXPOSE 8000

# 7. Der Startbefehl
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]