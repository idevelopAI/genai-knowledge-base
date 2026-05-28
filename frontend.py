import streamlit as st
import requests

# 1. Seite konfigurieren
st.set_page_config(page_title="GenAI Wissensdatenbank", page_icon="🤖", layout="centered")

# 2. Titel und Beschreibung
st.title("🤖 Interaktive Wissensdatenbank")
st.markdown("Stelle Fragen an deine internen Dokumente. Die KI antwortet basierend auf dem bereitgestellten Kontext.")

# 3. API-Endpunkt deines Docker-Backends
API_URL = "http://127.0.0.1:8000/ask"

# 4. Session State initialisieren (für den Chat-Verlauf)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Eingabefeld für den Nutzer
if prompt := st.chat_input("Deine Frage an das Dokument..."):
    # Nutzer-Nachricht dem Verlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 7. KI-Antwort generieren
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(KI überlegt...)*")
        
        try:
            # Anfrage an dein Docker-Backend senden
            response = requests.post(API_URL, json={"question": prompt})
            response.raise_for_status() 
            
            # Antwort auslesen
            answer = response.json()["answer"]
            message_placeholder.markdown(answer)
            
            # KI-Nachricht dem Verlauf hinzufügen
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except requests.exceptions.RequestException as e:
            error_msg = f"⚠️ Fehler: Das Backend ist nicht erreichbar. Läuft der Docker-Container? ({str(e)})"
            message_placeholder.error(error_msg)