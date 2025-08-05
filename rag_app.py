import os
import getpass
from datetime import datetime
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader, CSVLoader

# Ensure cryptography support for encrypted PDFs
try:
    import cryptography
except ImportError:
    raise ImportError("The 'cryptography' package is required for AES-encrypted PDFs. Please install it using: pip install cryptography")

LOG_DIR = "logs"
INDEX_DIR = "faiss_indexes"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Utility: Load documents by file type
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    elif ext == ".docx":
        return Docx2txtLoader(path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(path).load()
    elif ext == ".csv":
        return CSVLoader(path).load()
    else:
        return []

# Utility: Log query and response to SQLite
def log_to_db(project_name, query, response, documents, user_tag):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            user_tag TEXT,
            user_id TEXT,
            query TEXT,
            response TEXT,
            documents TEXT
        )''')
    user_id = st.session_state.get("user_id", "anonymous")
    cur.execute('''
        INSERT INTO logs (timestamp, user_tag, user_id, query, response, documents)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (datetime.now().isoformat(), user_tag, user_id, query, response, ', '.join(documents)))
    conn.commit()
    conn.close()

# Utility: Get distinct user tags for sidebar dropdown
def get_user_tags(project_name):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_tag FROM logs")
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows if row[0]]

# Utility: Get session-specific document list
def get_session_documents(project_name, user_tag):
    db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM logs WHERE user_tag = ?", conn, params=(user_tag,))
    conn.close()
    if df.empty:
        return []
    all_docs = ', '.join(df['documents'].dropna().tolist()).split(', ')
    return list(set([doc for doc in all_docs if doc.strip()]))

# Utility: Get session summary
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

# Utility: Timeline plot
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

# ---- Search Log Utility ----
def filter_query_log(df, query_text, doc_filter, start_date, end_date):
    if query_text:
        df = df[df["query"].str.contains(query_text, case=False, na=False)]
    if doc_filter:
        df = df[df["documents"].str.contains(doc_filter, case=False, na=False)]
    if start_date:
        df = df[df["timestamp"] >= start_date]
    if end_date:
        df = df[df["timestamp"] <= end_date]
    return df

# ---- Main App ----
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("RAG App Navigation")
    tabs = ["Q&A interface", "Log reports", "Analytics dashboard"]
    selected_tab = st.sidebar.radio("Select view:", tabs)

    st.sidebar.markdown("---")
    default_user_id = getpass.getuser()
    st.session_state["user_id"] = st.sidebar.text_input("User ID", value=st.session_state.get("user_id", default_user_id))

    st.sidebar.markdown("---")
    project_name = st.sidebar.text_input("Enter project name", "default_project")
    project_dir = os.path.join("docs", project_name)
    os.makedirs(project_dir, exist_ok=True)

    existing_tags = get_user_tags(project_name)
    selected_tag = st.sidebar.selectbox("Select or enter user/session tag", existing_tags + ["<new>"])
    if selected_tag == "<new>":
        selected_tag = st.sidebar.text_input("Enter new tag", "session_1")

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    faiss_path = os.path.join(INDEX_DIR, f"{project_name}_faiss")
    db = None
    if os.path.exists(faiss_path):
        try:
            db = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
        except:
            db = None

    doc_paths = [os.path.join(project_dir, f) for f in os.listdir(project_dir)]

    if "loaded_docs" not in st.session_state and doc_paths:
        loaded_docs = []
        for path in doc_paths:
            loaded_docs.extend(load_document(path))
        st.session_state.loaded_docs = loaded_docs

    if selected_tab == "Q&A interface":
        st.header("Q&A Interface")

        existing_docs = os.listdir(project_dir)
        show_upload = not existing_docs

        if show_upload:
            uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "docx", "md", "csv"], accept_multiple_files=True)
        else:
            st.markdown("---")
            action = st.radio("What would you like to do?", ["Query Existing Documents", "Upload More Documents"])
            uploaded_files = st.file_uploader("Upload additional documents", type=["pdf", "txt", "docx", "md", "csv"], accept_multiple_files=True) if action == "Upload More Documents" else None

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(project_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                doc_paths.append(file_path)

            all_docs = []
            for path in doc_paths:
                all_docs.extend(load_document(path))
            st.session_state.loaded_docs = all_docs
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)
            db = FAISS.from_documents(chunks, embedding)
            db.save_local(faiss_path)

        if db:
            st.markdown("---")
            st.subheader("Ask a question about your documents:")
            if "qa_pairs" not in st.session_state:
                st.session_state.qa_pairs = []
            with st.form(key="qa_form"):
                user_query = st.text_input("Enter your question")
                submit_btn = st.form_submit_button("Submit")
                if submit_btn and user_query:
                    llm = Ollama(model="mistral")
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
                    result = qa_chain.run(user_query)
                    st.session_state.qa_pairs.append((user_query, result))
                    log_to_db(project_name, user_query, result, [os.path.basename(p) for p in doc_paths], selected_tag)

            for idx, (q, a) in enumerate(st.session_state.qa_pairs, 1):
                st.markdown(f"**Q{idx}:** {q}")
                st.markdown(f"**A{idx}:** {a}")

            with st.expander("Document Preview", expanded=False):
                loaded_docs = st.session_state.get("loaded_docs", [])
                for doc in loaded_docs:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        fname = os.path.basename(doc.metadata['source'])
                        content = doc.page_content[:1000]
                        st.markdown(f"**{fname}**  \n{content}")

    elif selected_tab == "Log reports":
        st.header("Query Log Viewer")

        db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
        if not os.path.exists(db_path):
            st.info("No logs found for this project.")
        else:
            df = pd.read_sql_query("SELECT * FROM logs", sqlite3.connect(db_path))

            st.markdown("### Filter Logs")
            col1, col2 = st.columns(2)
            with col1:
                filter_user_tag = st.selectbox("Filter by tag", options=["All"] + sorted(df["user_tag"].dropna().unique().tolist()))
                filter_doc = st.text_input("Filter by document name (partial match)")
            with col2:
                filter_query = st.text_input("Filter by query text (partial match)")
                start_date = st.date_input("Start date", value=None)
                end_date = st.date_input("End date", value=None)

            filtered_df = df.copy()
            if filter_user_tag != "All":
                filtered_df = filtered_df[filtered_df["user_tag"] == filter_user_tag]
            filtered_df = filter_query_log(filtered_df, filter_query, filter_doc, str(start_date) if start_date else None, str(end_date) if end_date else None)

            st.markdown("### Results")
            st.dataframe(filtered_df, use_container_width=True)

            if not filtered_df.empty:
                csv_path = os.path.join(LOG_DIR, f"log_export_{project_name}.csv")
                filtered_df.to_csv(csv_path, index=False)
                with open(csv_path, "rb") as f:
                    st.download_button("Download filtered logs as CSV", f, file_name=os.path.basename(csv_path))

    elif selected_tab == "Analytics dashboard":
        st.header("Analytics Dashboard")

        db_path = os.path.join(LOG_DIR, f"query_log_{project_name}.db")
        if not os.path.exists(db_path):
            st.info("No logs available yet.")
        else:
            df = pd.read_sql_query("SELECT * FROM logs", sqlite3.connect(db_path))

            st.subheader("Session Summary")
            summary_df = get_session_summary(project_name)
            if summary_df.empty:
                st.info("No summary available.")
            else:
                st.dataframe(summary_df, use_container_width=True)

            st.subheader("Query Timeline")
            timeline_plot = plot_session_timeline(project_name)
            if timeline_plot:
                st.pyplot(timeline_plot)
            else:
                st.info("Not enough data to plot a timeline.")

if __name__ == "__main__":
    main()
