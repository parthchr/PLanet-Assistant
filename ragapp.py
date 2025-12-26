import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------- ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing in .env")
    st.stop()

# ---------------- STREAMLIT ----------------
st.set_page_config(page_title="Research Paper Q&A (Groq + RAG)", layout="wide")
st.title("📄 Research Paper Q&A (Groq + RAG)")

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ---------------- PROMPT (✅ FIXED) ----------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Answer the question strictly using the provided context.
    If the answer is not present, say "I don't know".

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def create_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    loader = PyPDFDirectoryLoader("research_papers")
    documents = loader.load()

    if not documents:
        raise ValueError("No PDFs found in research_papers folder")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)
    return FAISS.from_documents(docs, embeddings)

# ---------------- UI ----------------
if st.button("📥 Create Document Embeddings"):
    with st.spinner("Creating vector database..."):
        st.session_state.vectors = create_vector_store()
    st.success("Vector database created!")

user_query = st.text_input("🔍 Ask a question from the research papers")

# ---------------- QUERY ----------------
if user_query:
    if "vectors" not in st.session_state:
        st.error("Please create document embeddings first.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.time()
    response = retrieval_chain.invoke(
        {"input": user_query}  # ✅ CORRECT KEY
    )
    elapsed = time.time() - start

    st.subheader("🧠 Answer")
    st.write(response["answer"])
    st.caption(f"⏱️ Response time: {elapsed:.2f}s")

    with st.expander("📚 Retrieved Context"):
        for i, doc in enumerate(response["context"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
            st.write("---")
