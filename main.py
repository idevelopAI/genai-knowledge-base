import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Holt sich den Key unsichtbar aus dem System (oder der .env Datei)
MY_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not MY_API_KEY:
    raise ValueError("Sicherheitsfehler: GOOGLE_API_KEY wurde nicht gefunden!")

app = FastAPI(title="GenAI Knowledge Base API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# 2. Wir zwingen LangChain hier explizit, den Key zu nutzen (google_api_key=MY_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=MY_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Auch beim Sprachmodell zwingen wir den Key rein
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, google_api_key=MY_API_KEY)

system_prompt = (
    "Du bist ein hilfsbereiter KI-Assistent für eine interne Wissensdatenbank. "
    "Nutze ausschließlich den folgenden bereitgestellten Kontext, um die Frage zu beantworten. "
    "Wenn du die Antwort im Kontext nicht findest, sage, dass du es nicht weißt.\n\n"
    "Kontext: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        response = rag_chain.invoke(request.question)
        return QueryResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))