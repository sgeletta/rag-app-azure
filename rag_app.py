import os
import pickle
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

# --- Constants ---
LOG_DIR = "logs"
INDEX_DIR = "faiss_indexes"
DOCS_DIR = "docs"

# --- Authentication Setup ---
def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.subheader("🔐 Login Required")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username in st.secrets["users"] and st.secrets["users"][username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        return False
    return True

# --- Utilities ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model from HuggingFace and caches it."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

@st.cache_resource
def get_llm():
    """Loads the Ollama LLM and caches it."""
    return Ollama(model="mistral")

def load_document(path):
    """Loads a document from a file path based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PDFPlumberLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(path).load()
    elif ext == ".csv":
        return CSVLoader(path).load()
    else:
        st.warning(f"Unsupported file type: {ext}. Skipping.")
        return []

def log_to_db(project_name, query, answer, source_documents, user_tag):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, username TEXT, user_tag TEXT, query TEXT, answer TEXT, documents TEXT)''')
    
    # Extract unique source filenames from the document objects
    doc_names = sorted(list(set([os.path.basename(doc.metadata.get("source", "unknown")) for doc in source_documents])))

    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(),
               st.session_state.get("username", "unknown"),
               user_tag,
               query,
               answer,
               ", ".join(doc_names)))
    conn.commit()
    conn.close()


def get_user_tags(project_name):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT DISTINCT user_tag FROM logs", conn)
    conn.close()
    return df["user_tag"].tolist()


def get_session_summary(project_name):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby("user_tag").agg(
        num_queries=("query", "count"),
        last_query_time=("timestamp", "max"),
        documents_used=("documents", lambda x: ', '.join(set(', '.join(x).split(', '))))
    ).reset_index()
    return summary


def plot_session_timeline(project_name):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT timestamp, user_tag FROM logs", conn)
    conn.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    grouped = df.groupby(["date", "user_tag"]).size().reset_index(name="queries")
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=grouped, x="date", y="queries", hue="user_tag", marker="o")
    plt.title("Queries Over Time by User Tag")
    plt.xlabel("Date")
    plt.ylabel("Query Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# --- UI Components ---

def render_document_management(project_name, project_dir, db, faiss_path, db_session_key):
    """Renders the document management UI for uploading, deleting, and previewing documents."""
    st.header("📚 Document Management")
    embedding = get_embedding_model()

    # --- Document Upload and Indexing ---
    with st.expander("➕ Upload & Index New Documents", expanded=not db):
        uploaded_files = st.file_uploader(
            "Upload documents", type=["pdf", "txt", "docx", "md", "csv"], accept_multiple_files=True
        )
        if st.button("Process and Index Uploaded Files") and uploaded_files:
            doc_paths = []
            existing_docs_list = os.listdir(project_dir)
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_docs_list:
                    file_path = os.path.join(project_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    doc_paths.append(file_path)

            if doc_paths:
                with st.spinner("Processing documents..."):
                    all_docs = []
                    for path in doc_paths:
                        all_docs.extend(load_document(path))
                    
                    if all_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(all_docs)

                        if db:
                            db.add_documents(chunks)
                        else:
                            db = FAISS.from_documents(chunks, embedding)
                        
                        db.save_local(faiss_path)
                        st.session_state[db_session_key] = db
                        st.success(f"Indexed {len(doc_paths)} new document(s).")
                        st.rerun()
                    else:
                        st.warning("Could not load any content from the uploaded files.")
            else:
                st.info("All uploaded files are already present in the project directory.")

    existing_docs = os.listdir(project_dir)

    # --- Document Management and Preview ---
    if existing_docs:
        st.subheader("🗑️ Manage Existing Documents")
        docs_to_remove = st.multiselect("Select documents to remove:", existing_docs)
        if st.button("Remove Selected Documents") and docs_to_remove:
            try:
                if db:
                    full_paths_to_remove = {os.path.join(project_dir, doc) for doc in docs_to_remove}
                    # This part is fragile as it uses a private attribute `_dict`.
                    # It's a workaround for efficient deletion without rebuilding the index.
                    ids_to_remove = [
                        doc_id for doc_id, doc in db.docstore._dict.items()
                        if doc.metadata.get("source") in full_paths_to_remove
                    ]
                    if ids_to_remove:
                        db.delete(ids_to_remove)
                        db.save_local(faiss_path)
                        st.session_state[db_session_key] = db
            except Exception as e:
                st.error(f"Error updating index: {e}. Please consider re-indexing from scratch.")

            for doc_name in docs_to_remove:
                os.remove(os.path.join(project_dir, doc_name))

            if not os.listdir(project_dir) and os.path.exists(faiss_path):
                os.remove(faiss_path)
                st.session_state[db_session_key] = None

            st.success("Selected documents removed and index updated.")
            st.rerun()
        
        st.divider()

        with st.expander("📄 Document Previews", expanded=False):
            for f in existing_docs:
                try:
                    parsed_docs = load_document(os.path.join(project_dir, f))
                    if parsed_docs:
                        content_preview = parsed_docs[0].page_content[:200].strip() + "..."
                        st.markdown(f"**{f}**\n```\n{content_preview}\n```")
                except Exception as e:
                    st.warning(f"Could not preview {f}: {e}")
    else:
        st.info("Upload documents to get started.")

def render_qa_interface(project_name, project_dir, db, selected_tag):
    st.header("💬 Document Q&A Interface")

    # --- Q&A Section ---
    if db:
        st.subheader("❓ Ask Questions")
        qa_session_key = f"qa_pairs_{project_name}"
        if qa_session_key not in st.session_state:
            st.session_state[qa_session_key] = []

        with st.form("qa_form", clear_on_submit=True):
            user_query = st.text_input("Enter your question:", key="user_query")
            submitted = st.form_submit_button("Submit")
            if submitted and user_query:
                llm = get_llm()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever(),
                    return_source_documents=True
                )
                with st.spinner("Thinking..."):
                    # Pass an empty list of callbacks to work around a potential
                    # version incompatibility issue between langchain and pydantic.
                    response = qa_chain.invoke({"query": user_query}, config={"callbacks": []})
                    result = response.get("result", "Sorry, I could not find an answer.")
                    source_documents = response.get("source_documents", [])
                    st.session_state[qa_session_key].append((user_query, result, source_documents))
                    log_to_db(project_name, user_query, result, source_documents, selected_tag)

        # Display conversation history
        if st.session_state[qa_session_key]:
            st.subheader("Conversation History")
            for q, a, sources in reversed(st.session_state[qa_session_key]):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                if sources:
                    with st.expander("View Sources"):
                        for i, doc in enumerate(sources):
                            source_name = os.path.basename(doc.metadata.get("source", "N/A"))
                            st.markdown(f"**Source {i+1}: {source_name}**")
                            st.markdown(f"> {doc.page_content.strip()}")
                st.divider()
    else:
        st.info("Please go to the 'Document Management' tab to upload and index documents to begin the Q&A session.")


def render_log_reports(project_name, selected_tag):
    st.header("🧾 Query Logs")
    db_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM logs", conn)
        conn.close()

        if not df.empty:
            if selected_tag != "all_tags":
                df = df[df["user_tag"] == selected_tag]
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Logs as CSV",
                data=csv_data,
                file_name=f"log_export_{project_name}_{selected_tag}.csv",
                mime="text/csv",
            )
        else:
            st.info("No logs found for this project yet.")
    else:
        st.info("No log file found for this project.")

def render_analytics_dashboard(project_name):
    st.header("📊 Analytics Dashboard")
    db_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if os.path.exists(db_file):
        st.subheader("Session Summary")
        summary_df = get_session_summary(project_name)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No session data to summarize.")

        st.subheader("Query Timeline")
        timeline_plot = plot_session_timeline(project_name)
        if timeline_plot:
            st.pyplot(timeline_plot)
        else:
            st.info("No timeline data available.")
    else:
        st.info("No analytics available yet. Ask some questions first!")

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Document Q&A with RAG")
    st.sidebar.title("⚙️ Setup & Navigation")

    # --- Project and Session Setup ---
    project_name = st.sidebar.text_input("Project Name", "default_project")
    project_dir = os.path.join(DOCS_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Clear session state if project changes to avoid using stale data
    if "current_project" not in st.session_state or st.session_state.current_project != project_name:
        st.session_state.current_project = project_name
        for key in list(st.session_state.keys()):
            if key.startswith("db_") or key.startswith("qa_pairs_"):
                del st.session_state[key]
        st.rerun()

    existing_tags = get_user_tags(project_name)
    tag_options = ["all_tags"] + existing_tags + ["<new>"]
    selected_tag = st.sidebar.selectbox("Filter by or create a User/Session Tag", tag_options)
    if selected_tag == "<new>":
        selected_tag = st.sidebar.text_input("Enter new tag", f"session_{datetime.now():%H%M%S}")

    # --- Vector DB Loading ---
    embedding = get_embedding_model()
    faiss_path = os.path.join(INDEX_DIR, f"{project_name}_faiss")
    db_session_key = f"db_{project_name}"

    if db_session_key not in st.session_state:
        st.session_state[db_session_key] = None

    if st.session_state[db_session_key] is None and os.path.exists(faiss_path):
        with st.spinner("Loading vector store..."):
            try:
                st.session_state[db_session_key] = FAISS.load_local(
                    faiss_path, embeddings=embedding, allow_dangerous_deserialization=True
                )
            except (IOError, EOFError, pickle.UnpicklingError) as e:
                st.error(f"Failed to load FAISS index: {e}. A re-index is required.")
                if os.path.exists(faiss_path):
                    os.remove(faiss_path)

    db = st.session_state[db_session_key]

    # --- Main Content Area ---
    tabs = ["Q&A Interface", "Document Management", "Log Reports", "Analytics Dashboard"]
    
    # Determine the default tab based on whether a vector store (db) exists.
    # If db exists, default to Q&A. Otherwise, default to Document Management.
    default_tab_index = 0 if db else 1
    selected_tab = st.radio("Navigation", tabs, horizontal=True, index=default_tab_index)

    if selected_tab == "Q&A Interface":
        render_qa_interface(project_name, project_dir, db, selected_tag)
    elif selected_tab == "Document Management":
        render_document_management(project_name, project_dir, db, faiss_path, db_session_key)
    elif selected_tab == "Log Reports":
        render_log_reports(project_name, selected_tag)
    elif selected_tab == "Analytics Dashboard":
        render_analytics_dashboard(project_name)

# --- Entry Point ---
if __name__ == "__main__":
    if authenticate_user():
        st.sidebar.success(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
        main()
