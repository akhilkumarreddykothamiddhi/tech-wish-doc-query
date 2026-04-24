import streamlit as st
import os, uuid, base64
from pathlib import Path
from groq import Groq
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase import create_client
import snowflake.connector
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────
#  TECHWISH BRANDING & ASSETS
# ─────────────────────────────────────────────────────────────────
LOGO_PATH    = "Techwish-Logo-white (3).png"
COMPANY_NAME = "Techwish AI"

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION (Reading from Secrets)
# ─────────────────────────────────────────────────────────────────
def cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

SNOWFLAKE_ACCOUNT   = cfg("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = cfg("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = cfg("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = cfg("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE  = cfg("SNOWFLAKE_DATABASE", "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = cfg("SNOWFLAKE_SCHEMA", "PUBLIC")

SUPABASE_URL      = cfg("SUPABASE_URL")
SUPABASE_ANON_KEY = cfg("SUPABASE_ANON_KEY")
GROQ_API_KEY      = cfg("GROQ_API_KEY")
# CRITICAL: This must be https://docquey-techwish.streamlit.app in your secrets
APP_URL           = cfg("APP_URL", "https://docquey-techwish.streamlit.app")
ALLOWED_DOMAIN    = cfg("ALLOWED_DOMAIN", "techwish.com")

DOCS_FOLDER   = "docs"
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

NO_CONTEXT_MSG = "I'm sorry, I don't have information about that in the available documents."

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=COMPANY_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  SESSION INIT
# ─────────────────────────────────────────────────────────────────
for k, v in {
    "user_id": None, "user_email": None, "user_name": None,
    "chat_sid": None, "chat_msgs": [],
    "logged_out": False, "redirect_url": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
#  AUTH HANDLER & REDIRECT BREAKOUT (TOP-LEVEL)
# ─────────────────────────────────────────────────────────────────

# 1. Handle the return from Google (Callback)
if "code" in st.query_params and not st.session_state.user_id:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        sess = supabase.auth.exchange_code_for_session({"auth_code": st.query_params["code"]})
        user = sess.user
        
        if ALLOWED_DOMAIN and not user.email.endswith(f"@{ALLOWED_DOMAIN}"):
            st.error(f"Access restricted to @{ALLOWED_DOMAIN} accounts.")
            st.stop()
        
        st.session_state.user_id = user.id
        st.session_state.user_email = user.email
        st.session_state.user_name = user.user_metadata.get("full_name", user.email.split("@")[0])
        st.session_state.logged_out = False
        
        # Clear params and refresh
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.query_params.clear()
        st.error(f"Authentication failed: {e}")

# 2. Handle the breakout to Google (Redirection)
if st.session_state.redirect_url:
    url = st.session_state.redirect_url
    st.session_state.redirect_url = None 
    components.html(f"""<script>window.top.location.href = "{url}";</script>""", height=0)
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS (DB, AI, PROCESSING)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@st.cache_resource
def get_db():
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT, user=SNOWFLAKE_USER, password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE, database=SNOWFLAKE_DATABASE, schema=SNOWFLAKE_SCHEMA
    )

def _sf_exec(sql: str, params: tuple = ()):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
    finally: cur.close()

def _sf_fetch(sql: str, params: tuple = ()):
    cur = get_db().cursor()
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    finally: cur.close()

def db_upsert_user(user_id, email, name):
    _sf_exec("MERGE INTO users USING (SELECT %s AS id, %s AS email, %s AS name) AS s ON users.user_id = s.id WHEN NOT MATCHED THEN INSERT (user_id, email, full_name) VALUES (s.id, s.email, s.name)", (user_id, email, name))

@st.cache_resource(show_spinner=False)
def embedder():
    return SentenceTransformer(EMBED_MODEL)

def pdf_text(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text):
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]

@st.cache_resource(show_spinner="📚 Indexing Documents...")
def build_index():
    folder = Path(DOCS_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs: return None, [], []
    chunks, meta = [], []
    for p in pdfs:
        for c in chunk_text(pdf_text(str(p))):
            chunks.append(c); meta.append(p.name)
    embs = embedder().encode(chunks).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, chunks, meta

def search(query, idx, chunks):
    if not idx: return "", False
    q = embedder().encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, TOP_K)
    relevant = [chunks[i] for i, s in zip(ids[0], scores[0]) if i < len(chunks) and s >= 0.30]
    return "\n\n---\n\n".join(relevant), bool(relevant)

def ask_groq(messages, system):
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL, messages=[{"role":"system","content":system}] + messages[-10:], temperature=0.0
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────────────────────────────
#  STYLING & LOGO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; background: #050810 !important; color: #e8eaf0 !important; }
.stChatInput { max-width: 860px !important; margin: 0 auto !important; }
[data-testid="stSidebar"] { background: #0b0f1e !important; }
</style>
""", unsafe_allow_html=True)

def logo_b64(path):
    if not Path(path).exists(): return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

# ─────────────────────────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────────────────────────

if not st.session_state.user_id:
    st.markdown('<div class="login-bg"></div>', unsafe_allow_html=True)
    b64 = logo_b64(LOGO_PATH)
    if b64: st.markdown(f'<div class="login-top-bar"><img src="{b64}" height="36"></div>', unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-card-html"><div class="login-title">Techwish Knowledge Hub</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="login-btn-wrap">', unsafe_allow_html=True)
        if st.button("🔑 Continue with Google", type="primary", use_container_width=True, key="login_btn"):
            res = supabase_client().auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": APP_URL, "scopes": "email profile", "query_params": {"prompt": "select_account"}}
            })
            if res.url:
                st.session_state.redirect_url = res.url
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- Post-Login UI ---
doc_idx, doc_chunks, doc_meta = build_index()

with st.sidebar:
    st.markdown(f"### {st.session_state.user_name}")
    st.write(st.session_state.user_email)
    if st.button("Logout"):
        st.session_state.user_id = None
        st.rerun()

# Chat logic
for m in st.session_state.chat_msgs:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.chat_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        ctx, found = search(prompt, doc_idx, doc_chunks)
        if not found:
            ans = NO_CONTEXT_MSG
        else:
            ans = ask_groq(st.session_state.chat_msgs, f"Answer using: {ctx}")
        st.markdown(ans)
        st.session_state.chat_msgs.append({"role": "assistant", "content": ans})
