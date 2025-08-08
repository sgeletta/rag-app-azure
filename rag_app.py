import os
import getpass
from datetime import datetime
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (TextLoader, PDFPlumberLoader,
                                                   UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader,
                                                   CSVLoader)

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
LOG_DIR = "logs"
INDEX_DIR = "faiss_indexes"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def load_document(path):
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
        return []


def log_to_db(project_name, query, answer, documents, user_tag):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, username TEXT, user_tag TEXT, query TEXT, answer TEXT, documents TEXT)''')
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(),
               st.session_state.get("username", "unknown"),
               user_tag,
               query,
               answer,
               ", ".join(documents)))
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


# --- Main Application ---
# --- Phase 1 - Step 4: Per-document Indexing and Retrieval ---

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("📁 Project & Session Setup")

    tabs = ["Q&A interface", "Log reports", "Analytics dashboard"]
    selected_tab = st.sidebar.radio("Navigation", tabs)

    project_name = st.sidebar.text_input("Project name", "default_project")
    project_dir = os.path.join("docs", project_name)
    os.makedirs(project_dir, exist_ok=True)

    existing_tags = get_user_tags(project_name)
    selected_tag = st.sidebar.selectbox("User/session tag", existing_tags + ["<new>"])
    if selected_tag == "<new>":
        selected_tag = st.sidebar.text_input("Enter new tag", "session_1")

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    faiss_path = os.path.join(INDEX_DIR, f"{project_name}_faiss")
    db = None
    if os.path.exists(faiss_path):
        try:
            db = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
        except (IOError, EOFError, pickle.UnpicklingError) as e:
            st.error(f"Failed to load the FAISS index. It might be corrupted. Please consider removing and re-indexing documents. Error: {e}")
            db = None

    if selected_tab == "Q&A interface":
        st.header("💬 Document Q&A Interface")

        doc_paths = []
        uploaded_files = []
        new_files = []
        choice = "Upload additional documents"
        existing_docs = os.listdir(project_dir)
        if existing_docs:
            choice = st.radio("Available actions:", ["Query existing documents", "Upload additional documents"])

        if choice == "Upload additional documents":
            uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "docx", "md", "csv"],
                                              accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(project_dir, uploaded_file.name)
                    if uploaded_file.name not in existing_docs:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        doc_paths.append(file_path)
                        new_files.append(uploaded_file.name)

                if doc_paths:
                    all_docs = []
                    for path in doc_paths:
                        all_docs.extend(load_document(path))
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(all_docs)

                    if db:
                        db.add_documents(chunks)
                    else:
                        db = FAISS.from_documents(chunks, embedding)
                    db.save_local(faiss_path)
                    st.success("New documents uploaded and indexed.")
                else:
                    st.info("All uploaded files are already indexed.")

        # Document Removal Feature
        if existing_docs:
            st.subheader("🗑️ Manage Documents")
            docs_to_remove = st.multiselect("Select documents to remove:", existing_docs)
            if st.button("Remove selected documents") and docs_to_remove:
                # Efficiently remove documents from the index without rebuilding
                if db:
                    full_paths_to_remove = {os.path.join(project_dir, doc) for doc in docs_to_remove}
                    ids_to_remove = [
                        doc_id for doc_id, doc in db.docstore._dict.items()
                        if doc.metadata.get("source") in full_paths_to_remove
                    ]
                    if ids_to_remove:
                        db.delete(ids_to_remove)
                        db.save_local(faiss_path)

                # Physically remove the files
                for doc_name in docs_to_remove:
                    os.remove(os.path.join(project_dir, doc_name))

                # If no documents are left, remove the index file
                if not os.listdir(project_dir) and os.path.exists(faiss_path):
                    os.remove(faiss_path)

                st.success("Selected documents removed and index updated.")
                st.rerun()

        existing_docs = os.listdir(project_dir)

        if db:
            st.subheader("Ask questions about your documents")
            if "qa_pairs" not in st.session_state:
                st.session_state.qa_pairs = []
            user_query = st.text_input("Enter your question")
            if st.button("Submit question") and user_query:
                llm = Ollama(model="mistral")
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
                response = qa_chain.invoke({"query": user_query})
                result = response.get("result", "Sorry, I could not find an answer.")
                st.session_state.qa_pairs.append((user_query, result))
                log_to_db(project_name, user_query, result, existing_docs, selected_tag)
                st.rerun()

            for i, (q, a) in enumerate(st.session_state.qa_pairs, start=1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")

            # Document Preview Section
            with st.expander("📄 Document Preview", expanded=False):
                for f in existing_docs:
                    parsed_docs = load_document(os.path.join(project_dir, f))
                    if parsed_docs:
                        top_paragraph = parsed_docs[0].page_content.strip().split("\n")[0]
                        st.markdown(f"**{f}**  \n{top_paragraph}")

    elif selected_tab == "Log reports":
        st.header("🧾 Query Logs")
        db_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
        if os.path.exists(db_file):
            df = pd.read_sql_query("SELECT * FROM logs", sqlite3.connect(db_file))
            if selected_tag:
                df = df[df["user_tag"] == selected_tag]
            st.dataframe(df, use_container_width=True)
            csv_file = os.path.join(LOG_DIR, f"log_export_{project_name}.csv")
            df.to_csv(csv_file, index=False)
            with open(csv_file, "rb") as f:
                st.download_button("Download Logs as CSV", f, file_name=os.path.basename(csv_file))
        else:
            st.info("No log file found for this project.")

    elif selected_tab == "Analytics dashboard":
        st.header("📊 Analytics Dashboard")
        db_file = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
        if os.path.exists(db_file):
            df = pd.read_sql_query("SELECT * FROM logs", sqlite3.connect(db_file))
            if selected_tag:
                filtered_df = df[df["user_tag"] == selected_tag]
            else:
                filtered_df = df

            st.subheader("Session Summary")
            summary_df = get_session_summary(project_name)
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("Query Timeline")
            timeline_plot = plot_session_timeline(project_name)
            if timeline_plot:
                st.pyplot(timeline_plot)
            else:
                st.info("No timeline data available.")
        else:
            st.info("No analytics available yet.")


# --- Entry Point ---
if __name__ == "__main__":
    if authenticate_user():
        st.sidebar.success(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
        main()
