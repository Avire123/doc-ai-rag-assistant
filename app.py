import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load local .env file if available
load_dotenv()

# Automatically fetch API Key from Streamlit Secrets or Environment
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(
    page_title="Document AI & RAG Query Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document AI & RAG Query Assistant")
st.caption("Upload PDFs, CSVs, or TXT files, index them into local FAISS vector storage, and query your data with Gemini AI.")

# Sidebar for File Upload Only
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, CSV, TXT)", 
        type=["pdf", "csv", "txt"], 
        accept_multiple_files=True
    )

# Extract text from uploaded files
def extract_text(files):
    combined_text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                combined_text += page.extract_text() or ""
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            combined_text += df.to_string()
        elif file.name.endswith(".txt"):
            combined_text += file.read().decode("utf-8")
    return combined_text

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main Workflow
if uploaded_files:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing Gemini API Key! Please configure GOOGLE_API_KEY in Streamlit Secrets or .env file.")
    else:
        with st.spinner("Extracting and chunking document text..."):
            raw_text = extract_text(uploaded_files)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(raw_text)

        st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} text chunks.")

        # Vector Store Creation
        if "vector_store" not in st.session_state or st.button("Re-index Documents"):
            with st.spinner("Building local FAISS vector storage..."):
                embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.vector_store = vector_store
                st.success("FAISS vector storage successfully updated!")

        # Query Interface
        st.divider()
        st.header("💬 Chat with your Data")

        user_query = st.text_input("Ask a question about your uploaded documents:")

        if user_query and "vector_store" in st.session_state:
            with st.spinner("Retrieving context and generating answer..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

                source_docs = retriever.invoke(user_query)

                prompt = ChatPromptTemplate.from_template(
                    "You are an assistant for question-answering tasks.\n"
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Keep the answer concise.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}"
                )

                rag_chain = (
                    {"context": lambda x: format_docs(source_docs), "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(user_query)

                st.subheader("Answer:")
                st.write(answer)

                # Source Citations
                with st.expander("📚 View Retrieved Sources / Context Chunks"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source Chunk {i+1}:**")
                        st.info(doc.page_content)
else:
    st.info("Upload one or more files in the sidebar to begin.")