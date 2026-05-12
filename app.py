import streamlit as st
import os
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY environment variable not set")
    st.stop()
client = Anthropic()

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 1 token ≈ 4 characters)"""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks with overlap based on token count."""
    chunks = []
    words = text.split()
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = estimate_tokens(word)
        
        if current_tokens + word_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap: keep last ~overlap tokens worth of words
            overlap_words = []
            overlap_tokens = 0
            for w in reversed(current_chunk):
                w_tokens = estimate_tokens(w)
                if overlap_tokens + w_tokens <= overlap:
                    overlap_words.insert(0, w)
                    overlap_tokens += w_tokens
                else:
                    break
            
            current_chunk = overlap_words
            current_tokens = overlap_tokens
        
        current_chunk.append(word)
        current_tokens += word_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(reader.pages):
        try:
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        except Exception as e:
            st.warning(f"Could not extract text from page {page_num + 1}: {e}")
    return text


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS index with inner product (cosine similarity on normalized vectors)."""
    # Normalize embeddings for cosine similarity via inner product
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index


def retrieve_chunks(query: str, k: int = 4) -> list[tuple[str, float]]:
    """Retrieve top-k most relevant chunks for a query."""
    if st.session_state.index is None or not st.session_state.chunks:
        return []
    
    # Embed query
    query_embedding = st.session_state.model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = st.session_state.index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(st.session_state.chunks):
            results.append((st.session_state.chunks[idx], float(distances[0][i])))
    
    return results


def build_augmented_prompt(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """Build augmented prompt with context and retrieval citations."""
    context = "\n\n".join(
        [f"[{i+1}] {chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)]
    )
    
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context from the document:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context above.
- If the answer is not found in the context, say "This information is not available in the provided document."
- Cite the specific sections you used by referencing them as [1], [2], etc.
- Be concise and clear in your answer."""
    
    return prompt


def process_query(user_query: str, k: int) -> str:
    """Process user query and get LLM response."""
    if not st.session_state.chunks:
        return "Please upload a PDF first."
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(user_query, k)
    
    if not retrieved_chunks:
        return "Could not retrieve any relevant information from the document."
    
    # Build augmented prompt
    augmented_prompt = build_augmented_prompt(user_query, retrieved_chunks)
    
    # Add to conversation history
    st.session_state.conversation.append({
        "role": "user",
        "content": user_query
    })
    
    # Call Claude API
    messages = st.session_state.conversation.copy()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions based on provided documents.",
        messages=[
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )
    
    assistant_message = response.content[0].text
    
    # Add assistant response to conversation
    st.session_state.conversation.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message


# Sidebar: PDF Upload and Settings
st.sidebar.title("📄 RAG Configuration")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Chunk text
        st.session_state.chunks = chunk_text(pdf_text, chunk_size=500, overlap=50)
        
        # Embed chunks
        embeddings = st.session_state.model.encode(
            st.session_state.chunks,
            convert_to_numpy=True
        )
        st.session_state.embeddings = embeddings
        
        # Build FAISS index
        st.session_state.index = build_faiss_index(embeddings)
        st.session_state.pdf_loaded = True
        
        st.sidebar.success(f"✅ PDF loaded! {len(st.session_state.chunks)} chunks created.")
        st.sidebar.info(f"Document contains ~{len(pdf_text)} characters")

# Sidebar: Retrieval settings
k = st.sidebar.slider(
    "Number of chunks to retrieve (k)",
    min_value=1,
    max_value=20,
    value=4,
    help="Higher k retrieves more context but may include irrelevant information"
)

if st.sidebar.button("🔄 Clear Conversation"):
    st.session_state.conversation = []
    st.rerun()

# Main interface
st.title("🔍 RAG PDF Chat")

if not st.session_state.pdf_loaded:
    st.info("👈 Please upload a PDF file using the sidebar to get started.")
else:
    # Display conversation
    for message in st.session_state.conversation:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Input area
    user_input = st.chat_input("Ask a question about the PDF...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(user_input, k)
                st.write(response)
