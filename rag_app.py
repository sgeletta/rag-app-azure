import os
import pickle
import sqlite3
from datetime import datetime
from contextlib import contextmanager
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    CSVLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
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
        # Use columns to create a centered, narrower login form for a better UI look.
        _col1, col2, _col3 = st.columns([1, 1.5, 1])
        with col2:
            with st.form("login_form"):
                st.subheader("🔐 Login Required")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    # Retrieve credentials from environment variables
                    configured_username = os.environ.get("RAG_APP_USERNAME")
                    configured_password = os.environ.get("RAG_APP_PASSWORD")

                    if not configured_username or not configured_password:
                        st.error(
                            "Authentication credentials (RAG_APP_USERNAME, RAG_APP_PASSWORD) "
                            "are not configured in the environment. Please contact your administrator."
                        )
                    elif username == configured_username and password == configured_password:
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
    # The OLLAMA_BASE_URL environment variable is the single source of truth.
    # - In Azure (via Bicep), it's set to 'http://<prefix>-ollama-app'.
    # - In local Docker (via docker-compose), it's set to 'http://ollama:11434'.
    # The code should not modify it. The default is for running Streamlit outside of Docker.
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") # This default works for local.

    return ChatOllama(model="mistral", base_url=base_url)

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

def _migrate_log_db_schema(conn):
    """Ensures the 'logs' table has the 'feedback' column."""
    c = conn.cursor()
    try:
        # Check if 'feedback' column exists
        c.execute("PRAGMA table_info(logs)")
        columns = [info[1] for info in c.fetchall()]
        if 'feedback' not in columns:
            # Add the feedback column with a default value for existing rows
            c.execute("ALTER TABLE logs ADD COLUMN feedback INTEGER DEFAULT 0")
            conn.commit()
    except sqlite3.OperationalError:
        # This can happen if the 'logs' table doesn't exist at all yet.
        # The calling function's CREATE TABLE IF NOT EXISTS will handle it.
        pass

@contextmanager
def get_db_connection(project_name):
    """Context manager for handling SQLite database connections."""
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    conn = sqlite3.connect(db_path)
    try:
        _migrate_log_db_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_to_db(project_name, query, answer, source_documents, user_tag):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    conn = sqlite3.connect(db_path)
    _migrate_log_db_schema(conn)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, username TEXT, user_tag TEXT, query TEXT, answer TEXT, documents TEXT, feedback INTEGER DEFAULT 0)''')
    
    # Extract unique source filenames from the document objects
    doc_names = sorted(list(set([os.path.basename(doc.metadata.get("source", "unknown")) for doc in source_documents])))

    c.execute("INSERT INTO logs (timestamp, username, user_tag, query, answer, documents) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(),
               st.session_state.get("username", "unknown"),
               user_tag,
               query,
               answer, ", ".join(doc_names)))
    log_id = c.lastrowid
    conn.close()
    return log_id

def update_feedback(project_name, log_id, feedback_value):
    """Updates the feedback for a specific log entry."""
    with get_db_connection(project_name) as conn:
        c = conn.cursor()
        # Using rowid to update is standard and efficient in SQLite
        c.execute("UPDATE logs SET feedback = ? WHERE rowid = ?", (feedback_value, log_id))

def _update_feedback_and_session(project_name, log_id, feedback_value, qa_session_key):
    """Callback to update DB and session state for feedback."""
    # 1. Update the persistent database
    update_feedback(project_name, log_id, feedback_value)

    # 2. Update the session state for immediate UI feedback
    # Find the matching log entry in the session and update its feedback value
    for i, item in enumerate(st.session_state[qa_session_key]):
        # item is a tuple: (query, answer, sources, log_id, feedback_value)
        if item[3] == log_id:
            st.session_state[qa_session_key][i] = (item[0], item[1], item[2], item[3], feedback_value)
            break


def get_user_tags(project_name):
    """Gets a list of user tags, ordered by most recently used."""
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return []
    try:
        with get_db_connection(project_name) as conn:
        # Order tags by the most recent timestamp for that tag
            query = "SELECT user_tag FROM logs GROUP BY user_tag ORDER BY MAX(timestamp) DESC"
            df = pd.read_sql_query(query, conn)
            tags = df["user_tag"].tolist()
    except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
        # Fallback for empty db or table not yet created
        tags = []
    return tags


def get_session_summary(project_name):
    log_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(log_file):
        return pd.DataFrame()
    with get_db_connection(project_name) as conn:
        df = pd.read_sql_query("SELECT * FROM logs", conn)

    if df.empty:
        return pd.DataFrame()
    
    # Ensure feedback column exists and is numeric, fill NaNs
    if 'feedback' not in df.columns:
        df['feedback'] = 0
    df['feedback'] = pd.to_numeric(df['feedback'], errors='coerce').fillna(0)

    summary = df.groupby("user_tag").agg(
        num_queries=("query", "count"),
        last_query_time=("timestamp", "max"),
        documents_used=("documents", lambda x: ', '.join(sorted(list(set(i.strip() for i in ', '.join(x).split(',') if i.strip()))))),
        positive_feedback=("feedback", lambda x: (x > 0).sum()),
        negative_feedback=("feedback", lambda x: (x < 0).sum())
    ).reset_index()
    return summary


def plot_session_timeline(project_name):
    log_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(log_file):
        return None
    with get_db_connection(project_name) as conn:
        df = pd.read_sql_query("SELECT timestamp, user_tag FROM logs", conn)

    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    grouped = df.groupby(["date", "user_tag"]).size().reset_index(name="queries")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=grouped, x="date", y="queries", hue="user_tag", marker="o", ax=ax)
    ax.set_title("Queries Over Time by User Tag")
    ax.set_xlabel("Date")
    ax.set_ylabel("Query Count")
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig

def plot_feedback_distribution(project_name):
    """Generates a bar chart showing feedback distribution by user tag."""
    log_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(log_file):
        return None
    with get_db_connection(project_name) as conn:
        # Select only logs that have received feedback
        df = pd.read_sql_query("SELECT user_tag, feedback FROM logs WHERE feedback != 0", conn)

    if df.empty:
        return None

    df['feedback_type'] = df['feedback'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='user_tag', hue='feedback_type', palette={'Positive': 'mediumseagreen', 'Negative': 'lightcoral'}, ax=ax)
    ax.set_title("Feedback Distribution by User Tag")
    ax.set_xlabel("User Tag")
    ax.set_ylabel("Feedback Count")
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig

def plot_daily_query_counts(project_name):
    """Generates a faceted bar chart of daily query counts per user tag."""
    log_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(log_file):
        return None
    with get_db_connection(project_name) as conn:
        df = pd.read_sql_query("SELECT timestamp, user_tag FROM logs", conn)

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby(["date", "user_tag"]).size().reset_index(name="query_count")

    g = sns.catplot(
        data=daily_counts, x="date", y="query_count", col="user_tag",
        kind="bar", height=4, aspect=1.2, col_wrap=4
    )
    g.fig.suptitle("Daily Query Counts per Session", y=1.03)
    g.set_axis_labels("Date", "Query Count")
    g.set_xticklabels(rotation=45, ha='right')
    g.tight_layout(rect=[0, 0, 1, 0.97])
    return g.fig
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
            # 1. Remove the physical files
            for doc_name in docs_to_remove:
                os.remove(os.path.join(project_dir, doc_name))

            # 2. Get list of remaining files
            remaining_docs_list = os.listdir(project_dir)

            # 3. If there are remaining files, re-index them. Otherwise, clear the index.
            if remaining_docs_list:
                with st.spinner(f"Removed {len(docs_to_remove)} document(s). Re-building index..."):
                    remaining_doc_paths = [os.path.join(project_dir, doc) for doc in remaining_docs_list]

                    all_docs = []
                    for path in remaining_doc_paths:
                        all_docs.extend(load_document(path))

                    if all_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(all_docs)

                        # Create new index from remaining docs
                        new_db = FAISS.from_documents(chunks, embedding)
                        new_db.save_local(faiss_path) # This will overwrite the old one
                        st.session_state[db_session_key] = new_db
                        st.success("Index successfully rebuilt with remaining documents.")
                    else:
                        # No content could be extracted from remaining files
                        st.session_state[db_session_key] = None
                        if os.path.exists(faiss_path):
                            shutil.rmtree(faiss_path)
                        st.warning("Could not extract content from remaining files. Index is now empty.")
            else:
                # No documents left, so remove the index completely
                st.session_state[db_session_key] = None
                if os.path.exists(faiss_path):
                    shutil.rmtree(faiss_path)
                st.success("All documents removed. Index is now empty.")

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

            # 1. Retrieve relevant documents
            with st.spinner("Searching for relevant documents..."):
                # Use similarity search with a score threshold to filter irrelevant docs.
                # FAISS returns a distance score (lower is better). We can filter out results with a high score.
                # The threshold is empirical; you may need to adjust it based on your data and embedding model.
                SIMILARITY_THRESHOLD = 1.2
                retrieved_docs_with_scores = db.similarity_search_with_score(user_query, k=5)
                source_documents = [doc for doc, score in retrieved_docs_with_scores if score < SIMILARITY_THRESHOLD]

            # If no relevant documents are found after filtering, inform the user and stop.
            if not source_documents:
                st.warning("Could not find any relevant documents to answer your question. Please try rephrasing your query or check the uploaded documents.")
                # We still log this attempt for analytics purposes
                log_to_db(project_name, user_query, "No relevant documents found.", [], selected_tag)
                return

            # 2. Prepare the context and prompt for the LLM
            context = "\n\n".join([doc.page_content for doc in source_documents])
            template = """
            You are an assistant for question-answering tasks.
            Use ONLY the following pieces of retrieved context to answer the question.
            If you don't know the answer from the context, just say that you don't know. Do not make up an answer.
            Your answer must be based solely on the provided context.

            Context: {context}

            Question: {question}

            Answer:
            """
            prompt = PromptTemplate.from_template(template)
            # Using LangChain Expression Language (LCEL) for a more modern and composable approach.
            # This replaces the deprecated LLMChain. The StrOutputParser ensures the output is a string.
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser

            # 3. Stream the response to the UI and capture the full text
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                # st.write_stream directly handles the generator from the LCEL chain.
                # The LCEL chain with a StrOutputParser streams string chunks directly.
                full_response = st.write_stream(chain.stream({"context": context, "question": user_query}))

            # 4. Log the interaction and update session state
            # Now full_response is a string and can be safely logged to the database.
            log_id = log_to_db(project_name, user_query, full_response, source_documents, selected_tag)
            # Append a tuple with an initial feedback state (0 for 'no feedback')
            st.session_state[qa_session_key].append((user_query, full_response, source_documents, log_id, 0))
            st.rerun() # Rerun to display the new message in the history below

        # Display conversation history
        if st.session_state[qa_session_key]:
            st.subheader("Conversation History")
            # Unpack the new feedback value from the session state tuple
            for q, a, sources, log_id, feedback in reversed(st.session_state[qa_session_key]):
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

                    # Determine button state based on feedback value in session state
                    is_upvoted = feedback == 1
                    is_downvoted = feedback == -1

                    col1, col2, _ = st.columns([1, 1, 10])
                    with col1:
                        st.button("👍", key=f"up_{log_id}",
                                  on_click=_update_feedback_and_session,
                                  args=(project_name, log_id, 1, qa_session_key),
                                  disabled=is_upvoted)
                    with col2:
                        st.button("👎", key=f"down_{log_id}",
                                  on_click=_update_feedback_and_session,
                                  args=(project_name, log_id, -1, qa_session_key),
                                  disabled=is_downvoted)

                if sources:
                    with st.expander("View Sources"):
                        for i, doc in enumerate(sources):
                            source_name = os.path.basename(doc.metadata.get("source", "N/A"))
                            st.markdown(f"**Source {i+1}: {source_name}**")
                            st.markdown(f"> {doc.page_content.strip()}")
                st.divider()
    else:
        st.info("Please go to the 'Document Management' tab to upload and index documents to begin the Q&A session.")


def render_log_reports(project_name, log_filter_tag):
    st.header("🧾 Query Logs")
    db_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if os.path.exists(db_file):
        with get_db_connection(project_name) as conn:
            df = pd.read_sql_query("SELECT * FROM logs", conn)

        if not df.empty:            
            df_to_display = df.copy()
            if log_filter_tag != "Show All":
                df_to_display = df[df["user_tag"] == log_filter_tag]
            st.dataframe(df_to_display, use_container_width=True)
            
            csv_data = df_to_display.to_csv(index=False).encode('utf-8')
            # Make filename safe and descriptive
            filter_name = "all" if log_filter_tag == "Show All" else log_filter_tag

            st.download_button(
                "Download Logs as CSV",
                data=csv_data,
                file_name=f"log_export_{project_name}_{filter_name}.csv",
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

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Query Timeline")
            timeline_plot = plot_session_timeline(project_name)
            if timeline_plot:
                st.pyplot(timeline_plot)
            else:
                st.info("No timeline data available.")
        
        with col2:
            st.subheader("Feedback Analysis")
            feedback_plot = plot_feedback_distribution(project_name)
            if feedback_plot:
                st.pyplot(feedback_plot)
            else:
                st.info("No feedback has been provided yet.")

        st.divider()
        st.subheader("Daily Query Counts per Session")
        daily_counts_plot = plot_daily_query_counts(project_name)
        if daily_counts_plot:
            st.pyplot(daily_counts_plot)
        else:
            st.info("No query data available for this visualization.")
    else:
        st.info("No analytics available yet. Ask some questions first!")

# --- Main Application ---
def main():
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

    # --- UI for selecting session tag and log filter ---
    existing_tags = get_user_tags(project_name)
    st.sidebar.subheader("🏷️ Session & Filtering")

    # 1. Select tag for the current Q&A session
    session_tag_options = existing_tags + ["<new>"]
    # Set a sensible default: the most recent tag, or "<new>" if none exist.
    default_session_tag_index = 0 if existing_tags else session_tag_options.index("<new>")
    
    session_tag_selection = st.sidebar.selectbox(
        "Tag for this Q&A session:",
        session_tag_options,
        index=default_session_tag_index
    )
    
    if session_tag_selection == "<new>":
        session_tag = st.sidebar.text_input("Enter new tag name:", f"session_{datetime.now():%H%M%S}")
    else:
        session_tag = session_tag_selection

    # 2. Select filter for the log reports tab
    log_filter_options = ["Show All"] + existing_tags
    log_filter_tag = st.sidebar.selectbox("Filter logs by tag:", log_filter_options)

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
            except (IOError, EOFError, pickle.UnpicklingError, KeyError):
                # If loading the index fails (e.g., due to a version mismatch after an update),
                # silently delete the old index. The user will be prompted to re-index.
                if os.path.exists(faiss_path) and os.path.isdir(faiss_path):
                    shutil.rmtree(faiss_path)

    db = st.session_state[db_session_key]

    # --- Main Content Area ---
    tabs = ["Q&A Interface", "Document Management", "Log Reports", "Analytics Dashboard"]
    
    # Determine the default tab based on whether a vector store (db) exists.
    # If db exists, default to Q&A. Otherwise, default to Document Management.
    default_tab_index = 0 if db else 1
    selected_tab = st.radio("Navigation", tabs, horizontal=True, index=default_tab_index)

    if selected_tab == "Q&A Interface":
        render_qa_interface(project_name, project_dir, db, session_tag)
    elif selected_tab == "Document Management":
        render_document_management(project_name, project_dir, db, faiss_path, db_session_key)
    elif selected_tab == "Log Reports":
        render_log_reports(project_name, log_filter_tag)
    elif selected_tab == "Analytics Dashboard":
        render_analytics_dashboard(project_name)

# --- Entry Point ---
if __name__ == "__main__":
    # Set page config as the very first Streamlit command.
    st.set_page_config(layout="wide", page_title="Document Q&A with RAG")

    if authenticate_user():
        st.sidebar.success(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
        main()
