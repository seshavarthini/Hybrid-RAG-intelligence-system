
import hashlib
from pathlib import Path

import streamlit as st

from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_search import BM25Search
from src.hybrid_search import HybridSearch
from src.reranker import Reranker
from src.generator import AnswerGenerator


st.set_page_config(
    page_title="Hybrid RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hybrid RAG PDF Assistant")
st.write("Upload a PDF once, save it, and chat with it anytime.")

# Store uploaded PDFs permanently in this folder
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_saved_pdfs():
    return sorted(
        UPLOAD_DIR.glob("*.pdf"),
        key=lambda file: file.name.lower()
    )


def display_name(pdf_path):
    # Remove the hash prefix from the displayed filename
    parts = pdf_path.name.split("_", 1)
    return parts[1] if len(parts) == 2 else parts[0]


st.subheader("📂 PDF Library")

uploaded_pdf = st.file_uploader(
    "Upload a new PDF",
    type=["pdf"],
    key="pdf_uploader"
)

# Save a newly uploaded PDF
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing_file = next(
        (
            file for file in get_saved_pdfs()
            if hashlib.sha256(file.read_bytes()).hexdigest()
            == pdf_hash
        ),
        None
    )

    if existing_file:
        st.info("This PDF is already saved in your library.")
    else:
        original_name = Path(uploaded_pdf.name).name
        safe_name = "".join(
            char if char.isalnum() or char in "._- " else "_"
            for char in original_name
        )
        target = UPLOAD_DIR / f"{pdf_hash[:12]}_{safe_name}"
        target.write_bytes(pdf_bytes)

        st.success(f"Saved {original_name} successfully!")
        st.rerun()


pdf_files = get_saved_pdfs()

if not pdf_files:
    st.info("Upload a PDF to start your library.")
    st.stop()

selected_pdf = st.selectbox(
    "Choose a saved PDF",
    pdf_files,
    format_func=display_name
)

# Cache the search system for each PDF content hash.
# Reusing the same PDF avoids rebuilding indexes on normal reruns.
@st.cache_resource(show_spinner=False)
def load_rag(pdf_path, pdf_hash):
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    if not chunks:
        raise ValueError(
            "No readable text found. This may be a scanned PDF."
        )

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(embedding_model)
    vector_store.build(chunks)

    bm25 = BM25Search()
    bm25.build(chunks)

    hybrid = HybridSearch(bm25, vector_store)
    reranker = Reranker()
    generator = AnswerGenerator()

    return hybrid, reranker, generator


try:
    selected_hash = hashlib.sha256(
        selected_pdf.read_bytes()
    ).hexdigest()

    with st.spinner("Preparing your PDF search system..."):
        hybrid, reranker, generator = load_rag(
            str(selected_pdf.resolve()),
            selected_hash
        )

    st.success(f"Ready to chat with: {display_name(selected_pdf)}")

    # Start a separate conversation when another PDF is selected
    if st.session_state.get("active_pdf") != selected_hash:
        st.session_state["active_pdf"] = selected_hash
        st.session_state["messages"] = []

    # Show conversation history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**{source['filename']} — "
                            f"Page {source['page_number']}**"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "Ask a question about your selected PDF..."
    )

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                candidates = hybrid.search(
                    question,
                    top_k=10
                )

                results = reranker.rerank(
                    question,
                    candidates,
                    top_k=3
                )

                answer = generator.generate(
                    question,
                    results
                )

                st.markdown(answer)

                if results:
                    with st.expander("Sources"):
                        for result in results:
                            st.write(
                                f"**{result['filename']} — "
                                f"Page {result['page_number']}**"
                            )
                            st.write(result["text"])

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": results
        })

except Exception as error:
    st.error(f"Unable to process the PDF: {error}")
