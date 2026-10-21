
import hashlib
import secrets
from pathlib import Path
from urllib.parse import urlencode
import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_search import BM25Search
from src.hybrid_search import HybridSearch
from src.reranker import Reranker
from src.generator import AnswerGenerator


# --------------------------------------------------
# 1. Page setup and environment
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase configuration missing in .env")
    st.stop()


# --------------------------------------------------
# 2. Separate Supabase clients per OAuth flow
# --------------------------------------------------

@st.cache_resource
def get_client_registry():
    # Server-side registry. Each random key maps to its own client.
    return {}


def get_supabase_client(session_key):
    registry = get_client_registry()

    if session_key not in registry:
        registry[session_key] = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
        )

    return registry[session_key]


def get_saved_pdfs():
    return sorted(
        UPLOAD_DIR.glob("*.pdf"),
        key=lambda file: file.name.lower(),
    )


def display_name(pdf_path):
    parts = pdf_path.name.split("_", 1)
    return parts[1] if len(parts) == 2 else parts[0]


# --------------------------------------------------
# 3. OAuth callback
# --------------------------------------------------

auth_session_key = st.query_params.get("auth_session")
auth_code = st.query_params.get("code")
auth_error = st.query_params.get("error_description")

if auth_error:
    st.error(f"Google login failed: {auth_error}")
    st.stop()

if auth_code:
    if not auth_session_key:
        st.error(
            "OAuth session key missing. Please start a new Google login."
        )
        st.stop()

    registry = get_client_registry()

    if auth_session_key not in registry:
        st.error(
            "OAuth session expired or unavailable. "
            "Please restart the app and sign in again."
        )
        st.stop()

    supabase = registry[auth_session_key]

    try:
        auth_response = supabase.auth.exchange_code_for_session(
            {"auth_code": str(auth_code)}
        )

        if not auth_response.user or not auth_response.session:
            st.error("Supabase did not return a valid user session.")
            st.stop()

        st.session_state["supabase_user"] = auth_response.user
        st.session_state["supabase_session"] = auth_response.session
        st.session_state["auth_session_key"] = auth_session_key

        # Remove the one-time authorization code only.
        st.query_params.pop("code", None)
        st.query_params.pop("error", None)
        st.query_params.pop("error_description", None)

        st.rerun()

    except Exception as error:
        st.error(f"Supabase login failed: {error}")
        st.info("Start a fresh Google login and try again.")
        st.stop()


# --------------------------------------------------
# 4. Google login
# --------------------------------------------------

if "supabase_user" not in st.session_state:
    st.title("Hybrid RAG Intelligence")
    st.write("Sign in to access your PDF assistant.")

    if st.button("Login with Google", type="primary"):
        try:
            session_key = secrets.token_urlsafe(32)
            supabase = get_supabase_client(session_key)

            redirect_url = (
                "http://localhost:8501/?"
                + urlencode({"auth_session": session_key})
            )

            response = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": redirect_url,
                },
            })

            if not response.url:
                st.error("Could not create the Google login URL.")
                st.stop()

            st.session_state["oauth_url"] = response.url
            st.session_state["oauth_session_key"] = session_key

        except Exception as error:
            st.error(f"Could not start Google login: {error}")

    oauth_url = st.session_state.get("oauth_url")

    if oauth_url:
        st.link_button(
            "Continue with Google",
            oauth_url,
            type="primary",
        )

    st.stop()


# --------------------------------------------------
# 5. Restore the authenticated user's client
# --------------------------------------------------

session_key = st.session_state.get("auth_session_key")

if not session_key:
    st.error("Authentication session is missing. Please log in again.")
    st.stop()

registry = get_client_registry()

if session_key not in registry:
    st.error(
        "Authentication client is unavailable. "
        "Please restart the app and log in again."
    )
    st.stop()

supabase = registry[session_key]

user = st.session_state["supabase_user"]
st.sidebar.success(f"Logged in as {user.email}")


# --------------------------------------------------
# 6. Logout
# --------------------------------------------------

if st.sidebar.button("Logout"):
    try:
        supabase.auth.sign_out()
    except Exception as error:
        st.sidebar.warning(f"Sign-out warning: {error}")
    finally:
        registry.pop(session_key, None)

        for key in (
            "supabase_user",
            "supabase_session",
            "auth_session_key",
            "oauth_url",
            "oauth_session_key",
            "messages",
            "active_pdf",
        ):
            st.session_state.pop(key, None)

        st.query_params.clear()
        st.rerun()


# --------------------------------------------------
# 7. PDF library
# --------------------------------------------------

st.title("📚 Hybrid RAG PDF Assistant")
st.write("Upload a PDF once, save it, and chat with it anytime.")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.subheader("📂 PDF Library")

uploaded_pdf = st.file_uploader(
    "Upload a new PDF",
    type=["pdf"],
    key="pdf_uploader",
)


# --------------------------------------------------
# 8. Save PDF and prevent duplicate uploads
# --------------------------------------------------

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing_file = next(
        (
            file
            for file in get_saved_pdfs()
            if hashlib.sha256(file.read_bytes()).hexdigest() == pdf_hash
        ),
        None,
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


# --------------------------------------------------
# 9. Select a saved PDF
# --------------------------------------------------

pdf_files = get_saved_pdfs()

if not pdf_files:
    st.info("Upload a PDF to start your library.")
    st.stop()

selected_pdf = st.selectbox(
    "Choose a saved PDF",
    pdf_files,
    format_func=display_name,
)


# --------------------------------------------------
# 10. Build and cache the RAG system
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag(pdf_path, pdf_hash):
    pages = extract_pages(pdf_path)

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50,
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


# --------------------------------------------------
# 11. Chat with the selected PDF
# --------------------------------------------------

try:
    selected_hash = hashlib.sha256(
        selected_pdf.read_bytes()
    ).hexdigest()

    with st.spinner("Preparing your PDF search system..."):
        hybrid, reranker, generator = load_rag(
            str(selected_pdf.resolve()),
            selected_hash,
        )

    st.success(f"Ready to chat with: {display_name(selected_pdf)}")

    if st.session_state.get("active_pdf") != selected_hash:
        st.session_state["active_pdf"] = selected_hash
        st.session_state["messages"] = []

    st.session_state.setdefault("messages", [])

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
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                candidates = hybrid.search(
                    question,
                    top_k=10,
                )

                results = reranker.rerank(
                    question,
                    candidates,
                    top_k=3,
                )

                answer = generator.generate(question, results)

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
            "sources": results,
        })

except Exception as error:
    st.error(f"Unable to process the PDF: {error}")