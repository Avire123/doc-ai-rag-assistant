import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Document AI & RAG Query Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document AI & RAG Query Assistant")
st.caption("Upload PDFs or CSVs, index them with local FAISS vector storage, and query your data with Gemini AI.")

# Sidebar for API Key & File Upload
with st.sidebar:
    st.header("Configuration & Upload")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, CSV, TXT)", 
        type=["pdf", "csv", "txt"], 
        accept_multiple_files=True
    )

# Function to extract text from files
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

# Main workflow
if uploaded_files:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("Please provide a Gemini API Key in the sidebar to proceed.")
    else:
        with st.spinner("Extracting and chunking document text..."):
            raw_text = extract_text(uploaded_files)
            
            # Chunking text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(raw_text)

        st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} text chunks.")

        # Create Vector Store
        if "vector_store" not in st.session_state or st.button("Re-index Documents"):
            with st.spinner("Building local FAISS vector storage..."):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.vector_store = vector_store
                st.success("FAISS vector storage successfully updated!")

        # Query Interface
        st.divider()
        st.header("💬 Chat with your Data")

        user_query = st.text_input("Ask a question about your uploaded documents:")

        if user_query and "vector_store" in st.session_state:
            with st.spinner("Retrieving relevant context and generating answer..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Retrieval QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                response = qa_chain.invoke({"query": user_query})

                st.subheader("Answer:")
                st.write(response["result"])

                # Source Citations
                with st.expander("📚 View Retrieved Sources / Context Chunks"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source Chunk {i+1}:**")
                        st.info(doc.page_content)
else:
    st.info("Upload one or more files in the sidebar to begin.")