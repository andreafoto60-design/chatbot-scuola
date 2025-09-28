import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Chatbot scolastico", page_icon="📚")

st.title("📚 Chatbot scolastico con PDF")

# Carica PDF
loader = PyPDFLoader("documenti/PTOF-2425.pdf")
docs = loader.load()

# Dividi il testo in pezzi
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Embedding + Database vettoriale
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Input utente
query = st.text_input("✍️ Fai una domanda sul documento:")

if query:
    retriever = db.as_retriever()
    docs_retrieved = retriever.get_relevant_documents(query)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    risposta = llm.predict(
        f"Rispondi alla domanda basandoti solo su questo documento:\n\n{docs_retrieved}\n\nDomanda: {query}"
    )
    st.write(risposta)


