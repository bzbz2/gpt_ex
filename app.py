"""Breeze GPT - chat with your PDF / DOCX using OpenAI + Pinecone (based on rag_chatbot.ipynb)."""

import hashlib
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

APP_NAME = "Breeze GPT"
INDEX_NAME = "breeze"  # always the same index
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1024
CHAT_MODEL = "gpt-4.1-mini"
MAX_HISTORY = 20  # messages kept on the frontend


class PineconeDB:
    def __init__(self, api_key: str | None = None):
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))

    def ensure_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        # recreate the index if it exists with a different dimension
        if self.pc.has_index(index_name):
            if self.pc.describe_index(index_name).dimension == dimension:
                return
            self.pc.delete_index(index_name)
        self.pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self.pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    def clear(self, index_name: str):
        index = self.pc.Index(index_name)
        if index.describe_index_stats().get("total_vector_count", 0):
            index.delete(delete_all=True)
            # wait until the delete is visible so old chunks never leak into new answers
            while index.describe_index_stats().get("total_vector_count", 0):
                time.sleep(1)

    def _store(self, index_name: str, embedding) -> PineconeVectorStore:
        return PineconeVectorStore(index=self.pc.Index(index_name), embedding=embedding)

    def insert(self, index_name: str, documents, embedding) -> list[str]:
        ids = [hashlib.md5(f"{d.metadata.get('source')}|{d.metadata.get('page')}|{d.page_content}".encode()).hexdigest() for d in documents]
        return self._store(index_name, embedding).add_documents(documents, ids=ids)

    def retrieve(self, index_name: str, query: str, embedding, k: int = 4):
        return self._store(index_name, embedding).similarity_search(query, k=k)


@st.cache_resource
def get_clients():
    embedding = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    db = PineconeDB()
    db.ensure_index(INDEX_NAME, EMBED_DIM)
    return embedding, llm, db


def load_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        loader = PyMuPDFLoader(path) if suffix == ".pdf" else Docx2txtLoader(path)
        docs = loader.load()
    finally:
        os.remove(path)
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    return docs


def index_file(uploaded_file, embedding, db: PineconeDB) -> int:
    docs = load_file(uploaded_file)
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if not splits:
        raise ValueError("No text could be extracted from this file.")
    db.clear(INDEX_NAME)  # remove all previous data
    db.insert(INDEX_NAME, splits, embedding)
    return len(splits)


def to_lc_messages(history):
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]


def rewrite_question(llm, history, question: str) -> str:
    """Turn a follow-up question into a standalone one so retrieval works in a conversation."""
    if not history:
        return question
    prompt = [
        SystemMessage("Given the chat history and the latest user question, rewrite the question so it can be "
                      "understood without the history. Do NOT answer it, only return the rewritten question."),
        *to_lc_messages(history),
        HumanMessage(question),
    ]
    return llm.invoke(prompt).content


def answer_stream(llm, history, question: str, context_docs):
    context = "\n\n".join(f"[page {d.metadata.get('page', '-')}] {d.page_content}" for d in context_docs)
    prompt = [
        SystemMessage(f"You are {APP_NAME}, a helpful assistant that answers questions about the user's uploaded "
                      "document. Use only the context below. If the answer is not in the context, say you don't "
                      f"know.\n\nContext:\n{context}"),
        *to_lc_messages(history),
        HumanMessage(question),
    ]
    for chunk in llm.stream(prompt):
        yield chunk.content


st.set_page_config(page_title=APP_NAME, page_icon="🌬️")
st.title(f"🌬️ {APP_NAME}")
st.caption("Upload a PDF or DOCX and chat with it.")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Set OPENAI_API_KEY and PINECONE_API_KEY in your .env file.")
    st.stop()

embedding, llm, db = get_clients()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("file_id", None)
st.session_state.setdefault("file_name", None)

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    if uploaded is not None and uploaded.file_id != st.session_state.file_id:
        with st.spinner(f"Indexing {uploaded.name}..."):
            try:
                n = index_file(uploaded, embedding, db)
            except Exception as e:
                st.error(f"Failed to index file: {e}")
            else:
                st.session_state.file_id = uploaded.file_id
                st.session_state.file_name = uploaded.name
                st.session_state.messages = []  # fresh chat for the new file
                st.success(f"Indexed {n} chunks.")
    if st.session_state.file_name:
        st.info(f"Chatting with: **{st.session_state.file_name}**")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about your document...", disabled=not st.session_state.file_name)
if question:
    history = st.session_state.messages[-MAX_HISTORY:]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        standalone = rewrite_question(llm, history, question)
        docs = db.retrieve(INDEX_NAME, standalone, embedding, k=4)
        reply = st.write_stream(answer_stream(llm, history, question, docs))
    st.session_state.messages += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
