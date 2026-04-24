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

# -----------------------------------------------------------------
#  TECHWISH BRANDING & ASSETS
# -----------------------------------------------------------------
LOGO_PATH    = "Techwish-Logo-white (3).png"
COMPANY_NAME = "Techwish AI"

# -----------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------
def cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

SNOWFLAKE_ACCOUNT   = cfg("SNOWFLAKE_ACCOUNT",   "your-account")
SNOWFLAKE_USER      = cfg("SNOWFLAKE_USER",       "your_username")
SNOWFLAKE_PASSWORD  = cfg("SNOWFLAKE_PASSWORD",   "your_password")
SNOWFLAKE_WAREHOUSE = cfg("SNOWFLAKE_WAREHOUSE",  "COMPUTE_WH")
SNOWFLAKE_DATABASE  = cfg("SNOWFLAKE_DATABASE",   "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = cfg("SNOWFLAKE_SCHEMA",     "PUBLIC")

SUPABASE_URL      = cfg("SUPABASE_URL")
SUPABASE_ANON_KEY = cfg("SUPABASE_ANON_KEY")
GROQ_API_KEY      = cfg("GROQ_API_KEY")
ALLOWED_DOMAIN    = cfg("ALLOWED_DOMAIN", "techwish.com")

# -- Auto-detect the correct redirect URL --------------------------
# On Streamlit Cloud, STREAMLIT_SERVER_ADDRESS is not set but we can
# detect it from the request headers. We fall back to APP_URL secret,
# then localhost for local dev.
def get_app_url() -> str:
    # Explicit override always wins (set this in st.secrets as APP_URL)
    override = cfg("APP_URL", "")
    if override:
        return override.rstrip("/")
    # Try to detect from Streamlit's runtime headers (Cloud deployment)
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            host = headers.get("Host", "")
            if host:
                scheme = "https" if "streamlit.app" in host or "share.streamlit.io" in host else "http"
                return f"{scheme}://{host}"
    except Exception:
        pass
    # Fallback: localhost
    return "http://localhost:8501"

APP_URL = get_app_url()

DOCS_FOLDER   = "docs"
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

NO_CONTEXT_MSG = "I'm sorry, I don't have information about that in the available documents."

# -----------------------------------------------------------------
#  SMALL TALK DETECTION
# -----------------------------------------------------------------
SMALL_TALK_KEYWORDS = [
    "hi","hello","hey","hru","how are you","how r u","good morning",
    "good afternoon","good evening","good night","what's up","whats up",
    "sup","howdy","greetings","thanks","thank you","thank u","ty",
    "bye","goodbye","see you","take care","who are you","what are you",
    "what can you do","what do you do","help me","how can you help",
    "introduce yourself","tell me about yourself","nice to meet you",
    "nice","ok","okay","cool","great","awesome","wow","lol","haha",
    "good","bad","sad","happy","fine","alright","sure","yes","no",
    "yep","nope","please","sorry","excuse me","pardon"
]

def is_small_talk(text: str) -> bool:
    t = text.lower().strip()
    words = t.split()
    if len(words) <= 6:
        for kw in SMALL_TALK_KEYWORDS:
            if kw in t:
                return True
    return False

SMALL_TALK_SYSTEM = """You are the Techwish AI Knowledge Assistant - a friendly, professional AI assistant for Techwish employees.
You are currently handling a casual greeting or small talk message (NOT a document question).
Respond warmly, briefly, and professionally. Introduce yourself as the Techwish AI Knowledge Assistant when relevant.
Mention that you can answer questions about company documents, policies, and more.
Keep your reply concise - 1 to 3 sentences max."""

# -----------------------------------------------------------------
#  PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title=COMPANY_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------
#  LOGO HELPER
# -----------------------------------------------------------------
def logo_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

# -----------------------------------------------------------------
#  GLOBAL CSS
# -- GLOBAL CSS/JS loaded from file to avoid tokenizer issues --
def _load_styles():
    _p = Path(__file__).parent / 'styles.html'
    if _p.exists():
        return _p.read_text(encoding='utf-8')
    return ''
st.markdown(_load_styles(), unsafe_allow_html=True)

# -----------------------------------------------------------------
#  SNOWFLAKE DATABASE
# -----------------------------------------------------------------
@st.cache_resource
def get_db():
    conn = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT, user=SNOWFLAKE_USER, password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE, database=SNOWFLAKE_DATABASE, schema=SNOWFLAKE_SCHEMA,
    )
    _ensure_tables(conn)
    return conn

def _sf_exec(sql, params=()):
    conn = get_db(); cur = conn.cursor()
    try: cur.execute(sql, params); conn.commit()
    finally: cur.close()

def _sf_fetch(sql, params=()):
    cur = get_db().cursor()
    try: cur.execute(sql, params); return cur.fetchall()
    finally: cur.close()

def _ensure_tables(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(256) PRIMARY KEY, email VARCHAR(256) NOT NULL UNIQUE,
        full_name VARCHAR(256) DEFAULT '', department VARCHAR(256) DEFAULT 'General',
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
        id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(256) NOT NULL,
        user_email VARCHAR(256) NOT NULL, user_name VARCHAR(256) DEFAULT '',
        title VARCHAR(200) NOT NULL, created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id VARCHAR(36) PRIMARY KEY, session_id VARCHAR(36) NOT NULL,
        role VARCHAR(20) NOT NULL, content TEXT NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS document_metadata (
        id VARCHAR(36) PRIMARY KEY, filename VARCHAR(500) NOT NULL,
        uploaded_by VARCHAR(256), chunk_count INTEGER DEFAULT 0,
        uploaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.close()

def db_upsert_user(uid, email, name):
    _sf_exec("""MERGE INTO users AS t USING (SELECT %s AS user_id,%s AS email,%s AS full_name) AS s
        ON t.user_id=s.user_id WHEN NOT MATCHED THEN INSERT (user_id,email,full_name) VALUES (s.user_id,s.email,s.full_name)""",
        (uid, email, name))

def db_sessions(uid):
    rows = _sf_fetch("SELECT id,title,created_at FROM chat_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30", (uid,))
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(uid, email, name, title):
    sid = str(uuid.uuid4())
    _sf_exec("INSERT INTO chat_sessions (id,user_id,user_email,user_name,title) VALUES (%s,%s,%s,%s,%s)", (sid,uid,email,name,title))
    return sid

def db_messages(session_id):
    rows = _sf_fetch("SELECT role,content FROM chat_messages WHERE session_id=%s ORDER BY created_at", (session_id,))
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save(session_id, role, content):
    _sf_exec("INSERT INTO chat_messages (id,session_id,role,content) VALUES (%s,%s,%s,%s)", (str(uuid.uuid4()), session_id, role, content))

def db_delete(session_id):
    _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
    _sf_exec("DELETE FROM chat_sessions WHERE id=%s", (session_id,))

def db_delete_all_sessions(uid):
    rows = _sf_fetch("SELECT id FROM chat_sessions WHERE user_id=%s", (uid,))
    for row in rows: _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id=%s", (uid,))

def db_log_document(filename, uploaded_by, chunk_count):
    _sf_exec("INSERT INTO document_metadata (id,filename,uploaded_by,chunk_count) VALUES (%s,%s,%s,%s)",
        (str(uuid.uuid4()), filename, uploaded_by, chunk_count))

# -----------------------------------------------------------------
#  AI & PDF SEARCH
# -----------------------------------------------------------------
@st.cache_resource
def supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@st.cache_resource(show_spinner=False)
def embedder():
    return SentenceTransformer(EMBED_MODEL)

def pdf_text(path):
    doc = fitz.open(path); pages = [p.get_text() for p in doc]; doc.close()
    return "\n".join(pages)

def chunk_text(text):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+CHUNK_SIZE]); i += CHUNK_SIZE - CHUNK_OVERLAP
    return out

@st.cache_resource(show_spinner="📚 Building document index…")
def build_index():
    em = embedder(); folder = Path(DOCS_FOLDER); folder.mkdir(parents=True, exist_ok=True)
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs: return None, [], []
    chunks, meta = [], []
    for p in pdfs:
        for c in chunk_text(pdf_text(str(p))): chunks.append(c); meta.append(p.name)
    embs = em.encode(chunks, batch_size=64, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    return idx, chunks, meta

def search(query, idx, chunks):
    if not idx: return "", False
    q = embedder().encode([query]).astype("float32"); faiss.normalize_L2(q)
    scores, ids = idx.search(q, TOP_K)
    SCORE_THRESHOLD = 0.30
    relevant = [chunks[i] for i, s in zip(ids[0], scores[0]) if i < len(chunks) and s >= SCORE_THRESHOLD]
    if not relevant: return "", False
    return "\n\n---\n\n".join(relevant), True

def ask_groq(messages, system):
    history = messages[-10:]
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}] + history,
        temperature=0.0, max_tokens=1024)
    return resp.choices[0].message.content

def ask_groq_smalltalk(prompt):
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": SMALL_TALK_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=200)
    return resp.choices[0].message.content

# -----------------------------------------------------------------
#  SESSION INIT
# -----------------------------------------------------------------
for k, v in {
    "user_id": None, "user_email": None, "user_name": None,
    "chat_sid": None, "chat_msgs": [], "logged_out": False,
    "viewing_session": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------------------------------------
#  OAUTH CALLBACK
# -----------------------------------------------------------------
if "code" in st.query_params and not st.session_state.user_id and not st.session_state.logged_out:
    try:
        sess = supabase_client().auth.exchange_code_for_session({"auth_code": st.query_params["code"]})
        user = sess.user
        if ALLOWED_DOMAIN and not user.email.endswith(f"@{ALLOWED_DOMAIN}"):
            st.error(f"Access restricted to @{ALLOWED_DOMAIN} accounts.")
            st.stop()
        st.session_state.user_id    = user.id
        st.session_state.user_email = user.email
        st.session_state.user_name  = user.user_metadata.get("full_name", user.email.split("@")[0])
        st.session_state.logged_out = False
        db_upsert_user(user.id, user.email, st.session_state.user_name)
        st.query_params.clear(); st.rerun()
    except Exception as e:
        st.error(f"Auth error: {e}")

# -----------------------------------------------------------------
#  LOGIN PAGE  - button is INSIDE the card
# -----------------------------------------------------------------
if not st.session_state.user_id:
    if "code" in st.query_params and st.session_state.logged_out:
        st.query_params.clear()

    b64 = logo_b64(LOGO_PATH)

    st.markdown("""
    <style>
    html, body { background: #08090f !important; overflow: hidden !important; }
    .login-bg {
        position: fixed; inset: 0; z-index: 0;
        background: radial-gradient(ellipse 80% 60% at 20% 0%, rgba(99,102,241,0.18) 0%, transparent 60%),
                    radial-gradient(ellipse 60% 50% at 80% 100%, rgba(236,72,153,0.14) 0%, transparent 60%),
                    #08090f;
    }
    /* Full-page centered layout */
    .login-page-wrap {
        position: fixed; inset: 0; z-index: 10;
        display: flex; align-items: center; justify-content: center;
        pointer-events: none;
    }
    .login-card {
        pointer-events: all;
        width: 100%; max-width: 440px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 2.8rem 3rem 2.4rem;
        text-align: center;
        box-shadow: 0 40px 80px rgba(0,0,0,0.5);
        backdrop-filter: blur(24px);
        display: flex; flex-direction: column; align-items: center; gap: 0;
    }
    .login-badge {
        display: inline-flex; align-items: center; gap: 7px;
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
        border-radius: 999px; padding: 5px 14px; margin-bottom: 1.5rem;
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #a5b4fc;
    }
    .login-title {
        font-size: 2.2rem; font-weight: 800; line-height: 1.1;
        letter-spacing: -0.03em; margin: 0 0 0.5rem;
        background: linear-gradient(135deg, #fff 0%, rgba(165,180,252,0.9) 50%, rgba(236,72,153,0.8) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .login-sub {
        font-size: 0.82rem; color: rgba(255,255,255,0.38); margin-bottom: 1.6rem;
        font-weight: 300; line-height: 1.6; max-width: 320px;
    }
    .feature-chips {
        display: flex; gap: 10px; margin-bottom: 1.8rem; width: 100%;
    }
    .fchip {
        flex: 1; background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 11px;
        padding: 10px 8px; font-size: 0.65rem; color: rgba(255,255,255,0.38);
        text-align: center;
    }
    .fchip .fi { font-size: 1.1rem; margin-bottom: 4px; display: block; }
    /* Google login button - inside card */
    .login-btn-inside div[data-testid="stButton"] > button {
        background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#ec4899 100%) !important;
        color: white !important; font-family: 'Sora',sans-serif !important;
        font-weight: 700 !important; font-size: 0.93rem !important;
        border-radius: 13px !important; padding: 0.8rem 1.5rem !important;
        border: none !important; cursor: pointer !important; width: 100% !important;
        box-shadow: 0 8px 32px rgba(99,102,241,0.45) !important;
        transition: all 0.2s ease !important;
    }
    .login-btn-inside div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 16px 48px rgba(99,102,241,0.6) !important;
    }
    /* Hide sidebar & reset main on login page */
    section[data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stMain"] { margin-left: 0 !important; padding-top: 0 !important; }
    .main .block-container { padding: 0 !important; }
    /* The Streamlit widget area - place it inside the card visually */
    .login-widget-anchor {
        position: fixed;
        left: 50%;
        transform: translateX(-50%);
        width: 380px;
        z-index: 20;
        /* vertically we need to place it roughly at the bottom of the card:
           center (50vh) + card half-height minus button offset */
        top: calc(50vh + 118px);
    }
    </style>
    """, unsafe_allow_html=True)

    logo_tag = f'<img src="{b64}" height="30" style="display:block;margin-bottom:4px;">' if b64 else ''

    st.markdown(f"""
    <div class="login-bg"></div>
    <div class="login-page-wrap">
        <div class="login-card">
            {logo_tag}
            <div class="login-badge">&#128274;&nbsp;&nbsp;Secure Workspace</div>
            <div class="login-title">Your Intelligent<br>Knowledge Hub</div>
            <div class="login-sub">Ask anything. Get instant, precise answers drawn directly from your organization's documents.</div>
            <div class="feature-chips">
                <div class="fchip"><span class="fi">&#128196;</span>PDF-powered</div>
                <div class="fchip"><span class="fi">&#9889;</span>Instant answers</div>
                <div class="fchip"><span class="fi">&#128274;</span>@techwish.com only</div>
            </div>
            <!-- Login button rendered below by Streamlit, absolutely positioned to sit here -->
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit renders buttons in flow; we absolutely position this to sit inside the card
    st.markdown('<div class="login-widget-anchor">', unsafe_allow_html=True)
    st.markdown('<div class="login-btn-inside">', unsafe_allow_html=True)
    if st.button("🔑  Continue with Google", key="google_login_btn", use_container_width=True):
        try: supabase_client().auth.sign_out()
        except: pass
        st.session_state.logged_out = False
        # Detect the actual URL of the current page at click time
        try:
            # Streamlit >= 1.31 exposes the browser URL
            redirect_url = st.context.url if hasattr(st, "context") and hasattr(st.context, "url") else APP_URL
            # Strip query params to get the clean base URL
            redirect_url = redirect_url.split("?")[0].rstrip("/")
        except Exception:
            redirect_url = APP_URL
        res = supabase_client().auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_url,
                "scopes": "email profile",
                "query_params": {"prompt": "select_account", "access_type": "offline"}
            }
        })
        # Use st.markdown + immediate JS redirect - works on Streamlit Cloud
        st.markdown(
            f"""<script>window.top.location.href = "{res.url}";</script>
            <meta http-equiv="refresh" content="0; url={res.url}">""",
            unsafe_allow_html=True
        )
        st.stop()
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------
#  BUILD INDEX
# -----------------------------------------------------------------
doc_idx, doc_chunks, doc_meta = build_index()

# -----------------------------------------------------------------
#  SIDEBAR
# -----------------------------------------------------------------
name     = st.session_state.user_name
email    = st.session_state.user_email
initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"

with st.sidebar:
    b64 = logo_b64(LOGO_PATH)

    # -- Logo block - exactly 52px tall to match topnav ------------
    if b64:
        st.markdown(f"""
        <div class="sb-logo-block">
            <img src="{b64}" style="max-width:110px;height:auto;display:block;">
            <div class="sb-logo-caption">Knowledge Assistant</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sb-logo-block">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
                    display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0;">🧠</div>
                <div class="sb-logo-text">{COMPANY_NAME}</div>
            </div>
            <div class="sb-logo-caption">Knowledge Assistant</div>
        </div>
        """, unsafe_allow_html=True)

    # -- User pill -------------------------------------------------
    st.markdown(f"""
    <div class="user-pill">
        <div class="user-av">{initials}</div>
        <div class="user-av-name">{name}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # -- New Conversation ------------------------------------------
    st.markdown('<div class="new-chat-btn" style="padding:0 0.85rem 0.5rem;">', unsafe_allow_html=True)
    if st.button("✦  New Conversation", use_container_width=True):
        st.session_state.chat_sid        = None
        st.session_state.chat_msgs       = []
        st.session_state.viewing_session = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Chat History ----------------------------------------------
    st.markdown("### Chat History")

    sessions = db_sessions(st.session_state.user_id)
    if not sessions:
        st.markdown(
            '<p style="font-size:0.72rem;color:rgba(232,234,240,0.22);padding:0.2rem 0.85rem 0.4rem;">No conversations yet.</p>',
            unsafe_allow_html=True)

    for s in sessions:
        col1, col2 = st.columns([5, 1])
        label = s["title"][:22] + ("…" if len(s["title"]) > 22 else "")
        with col1:
            if st.button(f"💬  {label}", key=f"s_{s['id']}", use_container_width=True):
                st.session_state.viewing_session = s["id"]
                st.session_state.chat_sid        = s["id"]
                st.session_state.chat_msgs       = db_messages(s["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"d_{s['id']}"):
                db_delete(s["id"])
                if st.session_state.chat_sid == s["id"]:
                    st.session_state.chat_sid        = None
                    st.session_state.chat_msgs       = []
                    st.session_state.viewing_session = None
                st.rerun()

    # -- Sign Out - pinned at the bottom of the sidebar -----------
    st.markdown("""
    <div style="position:absolute;bottom:0;left:0;right:0;padding:0.75rem 0.85rem 1rem;
                border-top:1px solid rgba(255,255,255,0.06);background:#0e1018;">
    """, unsafe_allow_html=True)
    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("⏻  Sign Out", key="sidebar_signout", use_container_width=True):
        st.session_state["_do_signout"] = True
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------
#  FIXED TOP NAV
# -----------------------------------------------------------------
st.markdown(f"""
<div class="topnav">
    <div class="topnav-brand">
        <span class="topnav-brand-icon">&#9889;</span>
        Powered by&nbsp;<span>Techwish DocQuery</span>
    </div>
    <div class="topnav-right">
        <div class="status-pill">
            <div class="status-dot"></div>
            <span class="topnav-email">{email}</span>
        </div>
        <div class="avatar-wrap">
            <button class="avatar-circle" id="tw-avatar-btn" onclick="toggleAvatarMenu(event)">{initials}</button>
            <div class="avatar-dropdown" id="tw-avatar-menu">
                <div class="dd-header">
                    <div class="dd-user-name">{name}</div>
                    <div class="dd-user-email">{email}</div>
                </div>
                <div class="dd-section-label">Theme</div>
                <div class="theme-swatches">
                    <div class="theme-option" data-theme-name="dark" onclick="setTheme('dark')">
                        <div class="theme-swatch active" data-theme="dark" style="background:#08090f;border:2px solid rgba(255,255,255,0.5);" title="Dark"></div>
                        <div class="theme-swatch-label">Dark</div>
                    </div>
                    <div class="theme-option" data-theme-name="light" onclick="setTheme('light')">
                        <div class="theme-swatch" data-theme="light" style="background:#f0f0ec;border:2px solid rgba(0,0,0,0.2);" title="Light"></div>
                        <div class="theme-swatch-label">Light</div>
                    </div>
                    <div class="theme-option" data-theme-name="midnight" onclick="setTheme('midnight')">
                        <div class="theme-swatch" data-theme="midnight" style="background:linear-gradient(135deg,#1a1040,#0d0a1e);" title="Night"></div>
                        <div class="theme-swatch-label">Night</div>
                    </div>
                </div>
                <div class="dd-sep"></div>
                <div class="dd-item danger" onclick="fetch('').then(()=>{{}}); document.dispatchEvent(new CustomEvent('tw-signout'));">
                    <span class="dd-item-icon">&#10199;</span>
                    Sign out
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Handle sign-out (session_state flag - works on Streamlit Cloud, no query param needed)
if st.session_state.get("_do_signout"):
    try: supabase_client().auth.sign_out()
    except: pass
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.session_state.logged_out = True
    st.session_state.user_id    = None
    st.session_state.user_email = None
    st.session_state.user_name  = None
    st.session_state.chat_sid   = None
    st.session_state.chat_msgs  = []
    st.rerun()

# -----------------------------------------------------------------
#  SYSTEM PROMPT
# -----------------------------------------------------------------
SYSTEM_PROMPT = """You are the Techwish AI Knowledge Assistant.
You ONLY answer using the Context provided below. This is a strict rule with no exceptions.

RULES:
- Answer ONLY from the provided Context. Never use your training data or outside knowledge.
- If the Context does not contain a clear answer, respond ONLY with: "I'm sorry, I don't have information about that in the available documents."
- Do NOT say "According to the context" or "The document says". Answer directly and professionally.
- Do NOT guess, infer, or extrapolate beyond what is explicitly in the Context.
- Be concise, clear, and helpful.

Context:
{context}"""

# -----------------------------------------------------------------
#  MAIN CHAT AREA
# -----------------------------------------------------------------
viewing       = st.session_state.get("viewing_session")
has_chat_msgs = bool(st.session_state.chat_msgs)

if not has_chat_msgs and not viewing:
    # -- WELCOME --------------------------------------------------
    st.markdown("""
    <div class="welcome-outer" id="tw-welcome">
        <div class="welcome-orb">🧠</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-sub">
            Ask me anything about the documents uploaded to this workspace.<br>
            I'll give you precise, grounded answers - no hallucinations.
        </div>
        <div class="starter-grid">
            <div class="starter-card">
                <div class="starter-icon">📋</div>
                <div class="starter-text">Summarize key points from the uploaded documents</div>
            </div>
            <div class="starter-card">
                <div class="starter-icon">🔍</div>
                <div class="starter-text">Find specific information or policies in the docs</div>
            </div>
            <div class="starter-card">
                <div class="starter-icon">❓</div>
                <div class="starter-text">Ask any question and get a document-backed answer</div>
            </div>
            <div class="starter-card">
                <div class="starter-icon">📊</div>
                <div class="starter-text">Compare or contrast topics across multiple documents</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif viewing and not has_chat_msgs:
    # -- HISTORY DETAIL VIEW ---------------------------------------
    hist_msgs = db_messages(viewing)
    s_info    = next((s for s in db_sessions(st.session_state.user_id) if s["id"] == viewing), None)
    title_txt = s_info["title"] if s_info else "Conversation"
    date_txt  = s_info["date"]  if s_info else ""

    st.markdown(f"""
    <div style="max-width:840px;margin:0 auto;padding:1.2rem 1.5rem;
                overflow-y:auto;height:calc(100vh - 52px - 80px);">
        <div class="hist-session-header">
            <div class="hsh-title">{title_txt}</div>
            <div class="hsh-meta">{date_txt} · {len(hist_msgs)} messages</div>
        </div>
    """, unsafe_allow_html=True)

    for m in hist_msgs:
        role_label = "You" if m["role"] == "user" else "Assistant"
        st.markdown(f"""
        <div class="hist-msg-card">
            <div class="hist-role">{role_label}</div>
            <div class="hist-body">{m["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # -- LIVE CHAT VIEW --------------------------------------------
    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------
#  CHAT INPUT
# -----------------------------------------------------------------
if prompt := st.chat_input("Ask anything about your documents…"):
    # Hide welcome immediately via JS
    st.markdown("""
    <script>
    var w = document.getElementById('tw-welcome');
    if (w) w.style.display = 'none';
    </script>
    """, unsafe_allow_html=True)

    if st.session_state.viewing_session and not st.session_state.chat_msgs:
        st.session_state.viewing_session = None
        st.session_state.chat_sid        = None
        st.session_state.chat_msgs       = []

    if not st.session_state.chat_sid:
        st.session_state.chat_sid = db_new_session(
            st.session_state.user_id, email, name, prompt[:50])

    st.session_state.chat_msgs.append({"role": "user", "content": prompt})
    db_save(st.session_state.chat_sid, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if is_small_talk(prompt):
                answer = ask_groq_smalltalk(prompt)
            else:
                context, has_context = search(prompt, doc_idx, doc_chunks)
                if not has_context:
                    answer = NO_CONTEXT_MSG
                else:
                    system = SYSTEM_PROMPT.format(context=context)
                    answer = ask_groq(st.session_state.chat_msgs, system)
        st.markdown(answer)
        st.session_state.chat_msgs.append({"role": "assistant", "content": answer})
        db_save(st.session_state.chat_sid, "assistant", answer)

    st.markdown(
        '<script>document.getElementById("chat-bottom")?.scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True
    )
