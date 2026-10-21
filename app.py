import hashlib
from datetime import datetime, timezone
import secrets
import os
from pathlib import Path
from urllib.parse import urlencode

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

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Supabase clients used during OAuth
# --------------------------------------------------

@st.cache_resource
def get_client_registry():
    # Keeps the OAuth client (and its PKCE verifier) between Streamlit reruns.
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
# 3. Handle Supabase OAuth callback
# --------------------------------------------------

auth_session_key = st.query_params.get("auth_session")
auth_code = st.query_params.get("code")
auth_error = st.query_params.get("error_description")

if auth_error:
    st.error(f"Google login failed: {auth_error}")
    st.stop()

if auth_code:
    if not auth_session_key:
        st.error("OAuth session key missing. Please start login again.")
        st.stop()

    supabase = get_client_registry().get(auth_session_key)

    if supabase is None:
        st.error(
            "The OAuth session expired. Restart the app and log in again."
        )
        st.stop()

    try:
        # supabase-py v2 expects the authorization code as a string.
        auth_response = supabase.auth.exchange_code_for_session({
            "auth_code": str(auth_code)
        })

        if not auth_response.user or not auth_response.session:
            st.error("Supabase did not return a valid user session.")
            st.stop()

        st.session_state["supabase_user"] = auth_response.user
        st.session_state["supabase_session"] = auth_response.session
        st.session_state["auth_session_key"] = auth_session_key

        st.query_params.clear()
        st.rerun()

    except Exception as error:
        st.error(f"Supabase login failed: {error}")
        st.info("Click Login with Google to start a new login.")
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
# 5. Restore the authenticated Supabase client
# --------------------------------------------------

session_key = st.session_state.get("auth_session_key")

if not session_key:
    st.error("Authentication session is missing. Please log in again.")
    st.stop()

registry = get_client_registry()
supabase = registry.get(session_key)

if supabase is None:
    st.error(
        "Authentication client is unavailable. "
        "Restart the app and log in again."
    )
    st.stop()

user = st.session_state["supabase_user"]
st.sidebar.success(f"Logged in as {user.email}")


# --------------------------------------------------
# 6. Chat history: one database row per question/answer
# --------------------------------------------------

def load_chat_history():
    response = (
        supabase.table("chat_history")
        .select("id, question, answer")
        .eq("user_id", user.id)
        .order("id")
        .execute()
    )

    messages = []

    for row in response.data or []:
        if row.get("question"):
            messages.append({
                "role": "user",
                "content": row["question"],
            })

        if row.get("answer"):
            messages.append({
                "role": "assistant",
                "content": row["answer"],
            })

    return messages


def save_chat_exchange(question, answer):
    supabase.table("chat_history").insert({
        "user_id": user.id,
        "question": question,
        "answer": answer,
    }).execute()


def register_pdf_document(pdf_path, pdf_hash):
    """Save PDF metadata using the actual pdf_documents table columns."""
    file_name = display_name(pdf_path)
    owner_email = user.email

    if not owner_email:
        raise RuntimeError("The signed-in Supabase user has no email address.")

    # This table contains id, owner_email, file_name, and uploaded_at.
    # Check first so reruns do not create repeated rows for the same user/file.
    existing = (
        supabase.table("pdf_documents")
        .select("id")
        .eq("owner_email", owner_email)
        .eq("file_name", file_name)
        .limit(1)
        .execute()
    )

    if existing.data:
        return existing.data[0]

    result = (
        supabase.table("pdf_documents")
        .insert({
            "owner_email": owner_email,
            "file_name": file_name,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Supabase returned no inserted row. Check pdf_documents table permissions (RLS)."
        )

    return result.data[0]


# --------------------------------------------------
# 7. Logout
# --------------------------------------------------

if st.sidebar.button("Logout"):
    try:
        supabase.auth.sign_out()
    except Exception as error:
        st.sidebar.warning(f"Sign-out warning: {error}")

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
# 8. PDF library
# --------------------------------------------------

st.title("📚 Hybrid RAG PDF Assistant")
st.write("Upload a PDF once, save it, and chat with it anytime.")

st.subheader("📂 PDF Library")

uploaded_pdf = st.file_uploader(
    "Upload a new PDF",
    type=["pdf"],
    key="pdf_uploader",
)


# --------------------------------------------------
# 9. Save PDF and prevent duplicate uploads
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
        try:
            register_pdf_document(existing_file, pdf_hash)
            st.success("PDF details are synced with Supabase.")
        except Exception as error:
            st.error(f"PDF saved locally, but Supabase update failed: {error}")
    else:
        original_name = Path(uploaded_pdf.name).name

        safe_name = "".join(
            char if char.isalnum() or char in "._- " else "_"
            for char in original_name
        )

        target = UPLOAD_DIR / f"{pdf_hash[:12]}_{safe_name}"

        try:
            target.write_bytes(pdf_bytes)
            register_pdf_document(target, pdf_hash)
            st.success(f"Saved {original_name} and updated Supabase.")
            st.rerun()
        except Exception as error:
            st.error(f"PDF saved locally, but Supabase update failed: {error}")


# --------------------------------------------------
# 10. Select a saved PDF
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
# 11. Build and cache the RAG system
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
# 12. Chat with the selected PDF
# --------------------------------------------------

try:
    selected_hash = hashlib.sha256(
        selected_pdf.read_bytes()
    ).hexdigest()

    hybrid, reranker, generator = load_rag(
        str(selected_pdf.resolve()),
        selected_hash,
    )

    st.success("PDF ready for chatting!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "active_pdf" not in st.session_state:
        st.session_state["active_pdf"] = None

    # Load the signed-in user's saved history when PDF selection changes.
    # Note: the current table has no PDF identifier, so history is per user,
    # not separated by PDF.
    if st.session_state["active_pdf"] != selected_hash:
        st.session_state["active_pdf"] = selected_hash

        try:
            st.session_state["messages"] = load_chat_history()
        except Exception as history_error:
            st.error(f"Unable to load chat history: {history_error}")
            st.session_state["messages"] = []

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

    question = st.chat_input("Ask a question about your PDF")

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    candidates = hybrid.search(question, top_k=10)
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

            # Save question and answer together in one database row.
            try:
                save_chat_exchange(question, answer)
            except Exception as save_error:
                st.error(f"Unable to save chat history: {save_error}")

        except Exception as chat_error:
            st.error(f"Unable to generate the answer: {chat_error}")

except Exception as error:
    st.error(f"Unable to process the PDF: {error}")
