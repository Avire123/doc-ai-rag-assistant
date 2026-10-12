# 📄 Document AI & RAG Query Assistant 🤖✨

An intelligent **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, and **Google Gemini AI**. Upload your documents (PDFs, CSVs, TXT) and have interactive, context-grounded conversations with your data in real time!

---

## 🌟 Key Features

- 📑 **Multi-Format Document Support**: Upload multiple PDFs, CSVs, and plain text files seamlessly.
- 🧩 **Smart Text Chunking**: Splits large documents into manageable, overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`.
- ⚡ **Google GenAI Embeddings**: Fast, dense vector embeddings powered by Google Generative AI (`gemini-embedding-001`).
- 🔍 **FAISS Vector Storage**: Fast in-memory similarity search to instantly retrieve the most relevant context chunks.
- 🧠 **Gemini-Powered RAG Pipeline**: Answers user questions using the cutting-edge `gemini-3.8-flash` model with customized prompts.
- 📚 **Source Chunk Citations**: Inspect retrieved context passages and sources transparently within an expandable drawer.
- ☁️ **Cloud & Local Ready**: Configured for local development (`.env`) and one-click deployment on **Streamlit Community Cloud** (via Streamlit Secrets).

---

## 🛠️ Architecture & Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) 🎈
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) 🦜️🔗
- **Language Model**: Google Gemini (`gemini-3.8-flash`) ♊
- **Embeddings**: Google Generative AI Embeddings (`gemini-embedding-001`) 🔢
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) ⚡
- **Document Parsers**: `pypdf`, `pandas` 📊

---

## 🚀 Getting Started

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/Avire123/doc-ai-rag-assistant.git
cd doc-ai-rag-assistant
```

### 2. 🐍 Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔑 Configure Gemini API Key

Get your Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

---

## 🖥️ Running the Application

Launch the Streamlit app locally:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the repo.
3. In app settings under **Secrets**, add your API key:

```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
```

4. Click **Save** and deploy! 🎉

---

## 📂 Project Structure

```plaintext
doc-ai-rag-assistant/
│
├── app.py              # Main Streamlit web application & RAG chain logic
├── requirements.txt    # Project dependencies and libraries
├── .env                # Local environment variables (ignored by git)
├── .gitignore          # Files and folders ignored by git
└── README.md           # Project documentation
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Avire123/doc-ai-rag-assistant/issues).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
