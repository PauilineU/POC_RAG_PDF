# RAG PDF Chat Application

A Streamlit app that performs Retrieval-Augmented Generation (RAG) over user-uploaded PDFs using Anthropic's Claude API.

## Features

- **PDF Upload**: Upload any PDF and automatically extract and process the text
- **Smart Chunking**: Text split into 500-token chunks with 50-token overlap for better context
- **Semantic Search**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to embed and retrieve relevant content
- **FAISS Indexing**: Fast vector search using FAISS with cosine similarity (inner product on normalized embeddings)
- **LLM Integration**: Answers questions using Claude 3.5 Sonnet with citations
- **Configurable Retrieval**: Adjust the number of retrieved chunks (k) via sidebar slider
- **Conversation History**: Maintains context across multiple questions

## Setup

1. **Clone the repository** and navigate to the project directory:
```bash
cd poc1_rag_pdf
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
```

3. **Activate the virtual environment**:
   - **Windows**: `.\.venv\Scripts\Activate.ps1` or `.venv\Scripts\activate.bat`
   - **macOS/Linux**: `source .venv/bin/activate`

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Set your Anthropic API key**:
   - Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
   - Or export it as an environment variable:
     - **Windows**: `$env:ANTHROPIC_API_KEY = "your_api_key_here"`
     - **macOS/Linux**: `export ANTHROPIC_API_KEY=your_api_key_here`

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload a PDF**: Use the sidebar file uploader to select your PDF
2. **Adjust retrieval settings**: Use the slider to set how many document chunks (k) should be used for context
3. **Ask questions**: Type your question in the chat input
4. **Get answers**: Claude will answer based on the provided context with citations [1], [2], etc.

## How It Works

1. **Text Extraction**: pypdf extracts text from all pages of the uploaded PDF
2. **Chunking**: Text is split into 500-token chunks with 50-token overlap to preserve context
3. **Embedding**: Each chunk is embedded using the sentence-transformers `all-MiniLM-L6-v2` model
4. **Indexing**: Embeddings are stored in a FAISS index with normalized inner product distance (cosine similarity)
5. **Retrieval**: User questions are embedded and the top-k most similar chunks are retrieved
6. **Augmented Prompting**: Retrieved chunks are provided to Claude as context with instructions to answer only from the provided information
7. **Response**: Claude answers with citations to the specific chunks used

## Architecture

- **Frontend**: Streamlit for interactive web interface
- **Text Processing**: pypdf for PDF parsing
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (CPU version)
- **LLM**: Anthropic Claude 3.5 Sonnet API
- **State Management**: Streamlit session state for PDFs, embeddings, and conversation history

## Requirements

- Python 3.8+
- Streamlit 1.40+
- pypdf
- sentence-transformers
- faiss-cpu
- anthropic

See `requirements.txt` for specific versions.

## Notes

- The app uses in-memory FAISS indexing, so indexes are cleared when the app restarts
- PDFs are processed on upload (can take a few seconds for large documents)
- The token overlap ensures better context preservation between chunks
- Citations in the response refer to the order of retrieved chunks [1], [2], etc.
