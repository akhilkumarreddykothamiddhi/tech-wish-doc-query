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

# ─────────────────────────────────────────────────────────────────
#  TECHWISH BRANDING & ASSETS
# ─────────────────────────────────────────────────────────────────
LOGO_PATH    = "Techwish-Logo-white (3).png"
COMPANY_NAME = "Techwish AI"

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
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

# ── Auto-detect the correct redirect URL ──────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  SMALL TALK DETECTION
# ─────────────────────────────────────────────────────────────────
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

SMALL_TALK_SYSTEM = """You are the Techwish AI Knowledge Assistant — a friendly, professional AI assistant for Techwish employees.
You are currently handling a casual greeting or small talk message (NOT a document question).
Respond warmly, briefly, and professionally. Introduce yourself as the Techwish AI Knowledge Assistant when relevant.
Mention that you can answer questions about company documents, policies, and more.
Keep your reply concise — 1 to 3 sentences max."""

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
#  LOGO HELPER
# ─────────────────────────────────────────────────────────────────
def logo_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

/* ── RESET / BASE ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: #08090f !important;
    color: #e8eaf0 !important;
}
:root {
    --bg: #08090f; --bg2: #0e1018; --bg3: #13161f;
    --border: rgba(255,255,255,0.07); --border2: rgba(255,255,255,0.12);
    --accent: #6366f1; --text: #e8eaf0;
    --text2: rgba(232,234,240,0.55); --text3: rgba(232,234,240,0.28);
    --green: #22c55e; --red: #ef4444;
    --topnav-h: 52px;
}
#MainMenu, footer, .stDeployButton { display: none !important; visibility: hidden !important; }

/* ── Hide Streamlit header completely ────────────────────────── */
header[data-testid="stHeader"] {
    display: none !important; height: 0 !important;
    min-height: 0 !important; overflow: hidden !important;
}

/* ── Hide the sidebar collapse/expand toggle entirely ────────── */
[data-testid="collapsedControl"],
[data-testid="stSidebarToggleButton"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-headerNoPadding"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ── REMOVE ALL SCROLLBARS from main page ── */
html, body {
    height: 100vh !important;
    overflow: hidden !important;
    scrollbar-width: none !important;
}
html::-webkit-scrollbar,
body::-webkit-scrollbar,
[data-testid="stApp"]::-webkit-scrollbar,
[data-testid="stMain"]::-webkit-scrollbar,
.main::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
}
[data-testid="stApp"],
[data-testid="stMain"],
.main {
    overflow: hidden !important;
    scrollbar-width: none !important;
}

/* Main content area — push right of sidebar, below fixed topnav */
section[data-testid="stMain"] {
    margin-left: 248px !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding-top: var(--topnav-h) !important;
}
section[data-testid="stMain"] > div {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: 0 !important;
    height: 100% !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

/* ════════════════════════════════════════════════════════════
   CHAT SCROLL AREA — only the messages scroll
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatMessageContainer"],
[data-testid="stVerticalBlock"] {
    flex: 1 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scroll-behavior: smooth !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(255,255,255,0.1) transparent !important;
}
[data-testid="stVerticalBlock"]::-webkit-scrollbar { width: 4px; }
[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1); border-radius: 2px;
}

/* ════════════════════════════════════════════════════════════
   SIDEBAR — permanently fixed, NO scroll, NO collapse
   ═══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #0e1018 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 248px !important; max-width: 248px !important;
    width: 248px !important;
    overflow: hidden !important;
    height: 100vh !important;
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    z-index: 200 !important;
    flex-shrink: 0 !important;
    scrollbar-width: none !important;
}
section[data-testid="stSidebar"]::-webkit-scrollbar { display: none !important; width: 0 !important; }
section[data-testid="stSidebar"] > div {
    padding: 0 !important; background: #0e1018 !important;
    height: 100vh !important; overflow: hidden !important;
    scrollbar-width: none !important;
}
section[data-testid="stSidebar"] > div::-webkit-scrollbar { display: none !important; }

/* Inner scrollable zone for history items only */
.sb-history-scroll {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    max-height: calc(100vh - 280px);
    padding: 0 0.85rem;
    scrollbar-width: none !important;
}
.sb-history-scroll::-webkit-scrollbar { display: none !important; }

/* Padding so history list doesn't hide under the pinned sign-out button */
section[data-testid="stSidebar"] .stVerticalBlock {
    padding-bottom: 70px !important;
}

/* All sidebar buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: rgba(232,234,240,0.6) !important;
    border-radius: 9px !important; font-size: 0.78rem !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.14s ease !important;
    box-shadow: none !important; width: 100% !important;
    font-family: 'Sora', sans-serif !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.28) !important; color: white !important;
}
.new-chat-btn div[data-testid="stButton"] > button {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    color: #a5b4fc !important; font-weight: 700 !important;
}
.new-chat-btn div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(99,102,241,0.45) !important; color: white !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.58rem !important; text-transform: uppercase !important;
    letter-spacing: 0.13em !important; color: rgba(232,234,240,0.25) !important;
    margin: 0 0 0.4rem !important; font-weight: 700 !important; padding: 0 !important;
}

/* ════════════════════════════════════════════════════════════
   SIDEBAR LOGO BLOCK — aligned with topnav height (52px)
   ═══════════════════════════════════════════════════════════ */
.sb-logo-block {
    height: 52px;
    padding: 0 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: 2px;
}
.sb-logo-block img { max-width: 110px !important; height: auto !important; display: block; }
.sb-logo-caption {
    font-size: 0.58rem;
    font-weight: 600;
    color: rgba(165, 180, 252, 0.65);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding-left: 1px;
    margin-top: 1px;
    line-height: 1;
}
.sb-logo-text { font-size: 0.82rem; font-weight: 700; color: white; }
.sb-logo-sub  { font-size: 0.6rem; color: rgba(232,234,240,0.28); margin-top: 1px; }

/* User pill */
.user-pill {
    background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15);
    border-radius: 11px; padding: 9px 11px;
    margin: 0.8rem 0.85rem 0;
    display: flex; align-items: center; gap: 9px;
}
.user-av {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg,#6366f1,#ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700; color: white;
}
.user-av-name { font-size: 0.76rem; font-weight: 600; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ════════════════════════════════════════════════════════════
   TOP NAV — FIXED, always on top, full width minus sidebar
   ═══════════════════════════════════════════════════════════ */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.4rem;
    height: var(--topnav-h);
    background: rgba(8, 9, 15, 0.98);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: fixed;
    top: 0;
    left: 248px;
    right: 0;
    z-index: 999;
}
.topnav-brand {
    font-size: 0.84rem;
    font-weight: 600;
    color: rgba(232,234,240,0.6);
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.topnav-brand-icon { font-size: 1rem; }
.topnav-brand span {
    background: linear-gradient(90deg, #a5b4fc, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
.topnav-right {
    display: flex;
    align-items: center;
    gap: 14px;
}
.status-pill { display: flex; align-items: center; gap: 6px; }
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; flex-shrink: 0;
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50%      { box-shadow: 0 0 0 4px rgba(34,197,94,0.1); }
}
.topnav-email { font-size: 0.72rem; font-weight: 500; color: rgba(232,234,240,0.6); }

/* ════════════════════════════════════════════════════════════
   AVATAR BUTTON + DROPDOWN — top right corner
   ═══════════════════════════════════════════════════════════ */
.avatar-wrap { position: relative; }
.avatar-circle {
    width: 34px; height: 34px; border-radius: 50%; cursor: pointer;
    background: linear-gradient(135deg,#6366f1,#ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700; color: white;
    border: 2px solid rgba(255,255,255,0.15);
    transition: transform .15s, box-shadow .15s;
    font-family: 'Sora', sans-serif; flex-shrink: 0;
    outline: none;
    user-select: none;
}
.avatar-circle:hover {
    transform: scale(1.08);
    box-shadow: 0 0 0 3px rgba(99,102,241,0.35);
    border-color: rgba(165,180,252,0.5);
}
.avatar-circle:focus { outline: none; }

/* Avatar dropdown — fixed, top-right corner */
.avatar-dropdown {
    display: none;
    position: fixed;
    top: 58px;
    right: 14px;
    background: #0e1018;
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 16px;
    padding: 6px;
    min-width: 210px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.8), 0 0 0 1px rgba(99,102,241,0.1);
    z-index: 10000;
    animation: dropdown-in 0.15s ease;
}
@keyframes dropdown-in {
    from { opacity: 0; transform: translateY(-6px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.avatar-dropdown.open { display: block; }

.dd-header {
    padding: 10px 13px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 4px;
}
.dd-user-name  { font-size: 0.8rem; font-weight: 700; color: #e8eaf0; margin-bottom: 2px; }
.dd-user-email { font-size: 0.66rem; color: rgba(232,234,240,0.35); }
.dd-sep { height: 1px; background: rgba(255,255,255,0.07); margin: 4px 0; }
.dd-section-label {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: rgba(232,234,240,0.28); padding: 6px 13px 4px;
}
.theme-swatches { display: flex; gap: 10px; padding: 4px 13px 10px; align-items: center; }
.theme-swatch {
    width: 26px; height: 26px; border-radius: 50%; cursor: pointer;
    border: 2px solid transparent; transition: border-color .15s, transform .12s;
    flex-shrink: 0; position: relative;
}
.theme-swatch:hover { transform: scale(1.12); }
.theme-swatch.active { border-color: rgba(255,255,255,0.7); }
.theme-swatch-label {
    font-size: 0.6rem; color: rgba(232,234,240,0.35); text-align: center; margin-top: 2px;
}
.theme-option {
    display: flex; flex-direction: column; align-items: center; cursor: pointer; gap: 3px;
}
.theme-option:hover .theme-swatch-label { color: rgba(232,234,240,0.7); }
.dd-item {
    display: flex; align-items: center; gap: 9px;
    padding: 9px 13px; border-radius: 10px;
    font-size: 0.76rem; color: rgba(232,234,240,0.6);
    cursor: pointer; transition: background .12s, color .12s;
    margin: 1px 0;
}
.dd-item:hover { background: rgba(255,255,255,0.05); color: #e8eaf0; }
.dd-item-icon { font-size: 0.9rem; line-height: 1; }
.dd-item.danger { color: rgba(248,113,113,0.8); }
.dd-item.danger:hover { background: rgba(239,68,68,0.1); color: #fca5a5; }

/* ════════════════════════════════════════════════════════════
   WELCOME SCREEN — hidden as soon as any chat message exists
   ═══════════════════════════════════════════════════════════ */
.welcome-outer {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: calc(100vh - var(--topnav-h) - 80px);
    text-align: center; padding: 2rem;
    overflow: hidden;
}
/* Hide welcome screen when any chat message is present */
[data-testid="stChatMessage"] ~ .welcome-outer,
[data-testid="stChatMessage"] + .welcome-outer,
.chat-active .welcome-outer { display: none !important; }

.welcome-orb {
    width: 62px; height: 62px; border-radius: 18px; margin: 0 auto 1.1rem;
    background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#ec4899 100%);
    display: flex; align-items: center; justify-content: center; font-size: 1.7rem;
    box-shadow: 0 12px 36px rgba(99,102,241,0.35);
    animation: orb-glow 3s ease-in-out infinite;
}
@keyframes orb-glow {
    0%,100% { box-shadow: 0 12px 36px rgba(99,102,241,0.35); }
    50%      { box-shadow: 0 20px 56px rgba(139,92,246,0.55); }
}
.welcome-title {
    font-size: 1.75rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.5rem;
    background: linear-gradient(135deg,#fff 0%,#a5b4fc 55%,#ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;
}
.welcome-sub {
    font-size: 0.84rem; color: rgba(232,234,240,0.38); line-height: 1.7;
    margin-bottom: 1.8rem; max-width: 400px;
}
.starter-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    max-width: 520px; width: 100%; margin: 0 auto;
}
.starter-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px; padding: 14px 16px; text-align: left; cursor: pointer;
    transition: all 0.18s;
}
.starter-card:hover {
    background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.22);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.12);
}
.starter-icon { font-size: 1rem; margin-bottom: 6px; }
.starter-text { font-size: 0.73rem; color: rgba(232,234,240,0.45); line-height: 1.5; }

/* ════════════════════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.4rem 2rem !important; max-width: 840px !important; margin: 0 auto !important;
}
[data-testid="stChatMessageContent"] {
    border-radius: 16px !important; font-size: 0.86rem !important; line-height: 1.7 !important;
}
[data-testid="chatAvatarIcon-user"]      { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; color: white !important; }
[data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg,#0ea5e9,#6366f1) !important; color: white !important; }

/* History detail cards */
.hist-session-header {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
}
.hsh-title { font-size: 0.84rem; font-weight: 600; color: #e8eaf0; }
.hsh-meta  { font-size: 0.65rem; color: rgba(232,234,240,0.3); }
.hist-msg-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 11px; padding: 10px 14px; margin-bottom: 8px;
}
.hist-role {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: rgba(232,234,240,0.28); margin-bottom: 5px;
}
.hist-body { font-size: 0.8rem; color: rgba(232,234,240,0.72); line-height: 1.7; }

/* ════════════════════════════════════════════════════════════
   CHAT INPUT — pinned at the bottom
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important; backdrop-filter: blur(10px) !important;
    max-width: 840px !important; margin: 0 auto !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.45) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.09) !important;
}
[data-testid="stChatInput"] textarea {
    color: white !important; font-family: 'Sora', sans-serif !important; font-size: 0.84rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(232,234,240,0.25) !important; }

[data-testid="stBottom"] {
    background: rgba(8,9,15,0.95) !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    padding: 0.75rem 0 1rem !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

.stSpinner { color: #6366f1 !important; }
hr { border-color: rgba(255,255,255,0.05) !important; }
#chat-bottom { height: 1px; }

/* Sign out button in sidebar */
.signout-btn div[data-testid="stButton"] > button {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    color: rgba(252,165,165,0.8) !important;
    border-radius: 9px !important; font-size: 0.78rem !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.14s ease !important;
    width: 100% !important;
    font-family: 'Sora', sans-serif !important;
}
.signout-btn div[data-testid="stButton"] > button:hover {
    background: rgba(239,68,68,0.16) !important;
    border-color: rgba(239,68,68,0.45) !important;
    color: #fca5a5 !important;
}
</style>

<script>
// ══════════════════════════════════════════════════════════════
//  TECHWISH APP JS
// ══════════════════════════════════════════════════════════════

var TW_THEMES = {
    dark:     { bg:'#08090f',  bg2:'#0e1018', text:'#e8eaf0', sidebar:'#0e1018', topnav:'rgba(8,9,15,0.98)' },
    light:    { bg:'#f0f0ec',  bg2:'#ffffff', text:'#111118', sidebar:'#ffffff', topnav:'rgba(240,240,236,0.98)' },
    midnight: { bg:'#0d0a1e',  bg2:'#130f2a', text:'#e0d8ff', sidebar:'#100d22', topnav:'rgba(13,10,30,0.98)' }
};

window.toggleAvatarMenu = function(event) {
    if (event) { event.preventDefault(); event.stopPropagation(); }
    var menu = document.getElementById('tw-avatar-menu');
    if (!menu) return;
    menu.classList.toggle('open');
};

window.setTheme = function(name) {
    var t = TW_THEMES[name];
    if (!t) return;
    var r = document.documentElement;
    r.style.setProperty('--bg',  t.bg);
    r.style.setProperty('--bg2', t.bg2);
    r.style.setProperty('--text', t.text);
    document.querySelectorAll('html,body,[data-testid="stApp"],[data-testid="stMain"]').forEach(function(el) {
        el.style.setProperty('background', t.bg, 'important');
        el.style.setProperty('color', t.text, 'important');
    });
    document.querySelectorAll('section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div').forEach(function(el) {
        el.style.setProperty('background', t.sidebar, 'important');
    });
    document.querySelectorAll('.topnav').forEach(function(el) { el.style.background = t.topnav; });
    document.querySelectorAll('.theme-swatch').forEach(function(d) { d.classList.remove('active'); });
    var active = document.querySelector('.theme-swatch[data-theme="' + name + '"]');
    if (active) active.classList.add('active');
    var menu = document.getElementById('tw-avatar-menu');
    if (menu) menu.classList.add('open');
};

// Close dropdown on outside click
document.addEventListener('mousedown', function(e) {
    var menu = document.getElementById('tw-avatar-menu');
    var btn  = document.getElementById('tw-avatar-btn');
    if (!menu || !btn) return;
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
        menu.classList.remove('open');
    }
}, true);

// Rebind avatar btn after Streamlit rerenders
(function bindAvatarBtn() {
    setInterval(function() {
        var btn = document.getElementById('tw-avatar-btn');
        if (btn && !btn._twBound) {
            btn._twBound = true;
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var menu = document.getElementById('tw-avatar-menu');
                if (menu) menu.classList.toggle('open');
            });
        }
        document.querySelectorAll('.theme-option[data-theme-name]').forEach(function(el) {
            if (!el._twBound) {
                el._twBound = true;
                el.addEventListener('click', function() { window.setTheme(el.dataset.themeName); });
            }
        });

        // Hide welcome screen as soon as any chat message appears
        var hasMsgs = document.querySelector('[data-testid="stChatMessage"]');
        var welcome = document.querySelector('.welcome-outer');
        if (hasMsgs && welcome) {
            welcome.style.display = 'none';
        }
    }, 300);
})();

// Auto-scroll chat to bottom
function twScrollBottom() {
    var el = document.getElementById('chat-bottom');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
    // Also hide welcome if messages exist
    var hasMsgs = document.querySelector('[data-testid="stChatMessage"]');
    var welcome = document.querySelector('.welcome-outer');
    if (hasMsgs && welcome) welcome.style.display = 'none';
}
new MutationObserver(twScrollBottom).observe(document.documentElement, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SNOWFLAKE DATABASE
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  AI & PDF SEARCH
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  SESSION INIT
# ─────────────────────────────────────────────────────────────────
for k, v in {
    "user_id": None, "user_email": None, "user_name": None,
    "chat_sid": None, "chat_msgs": [], "logged_out": False,
    "viewing_session": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
#  OAUTH CALLBACK
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  LOGIN PAGE  — button is INSIDE the card
# ─────────────────────────────────────────────────────────────────
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
    /* Google login button — inside card */
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
    /* The Streamlit widget area — place it inside the card visually */
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
        # Use st.markdown + immediate JS redirect — works on Streamlit Cloud
        st.markdown(
            f"""<script>window.top.location.href = "{res.url}";</script>
            <meta http-equiv="refresh" content="0; url={res.url}">""",
            unsafe_allow_html=True
        )
        st.stop()
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  BUILD INDEX
# ─────────────────────────────────────────────────────────────────
doc_idx, doc_chunks, doc_meta = build_index()

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
name     = st.session_state.user_name
email    = st.session_state.user_email
initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"

with st.sidebar:
    b64 = logo_b64(LOGO_PATH)

    # ── Logo block — exactly 52px tall to match topnav ────────────
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

    # ── User pill ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="user-pill">
        <div class="user-av">{initials}</div>
        <div class="user-av-name">{name}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── New Conversation ──────────────────────────────────────────
    st.markdown('<div class="new-chat-btn" style="padding:0 0.85rem 0.5rem;">', unsafe_allow_html=True)
    if st.button("✦  New Conversation", use_container_width=True):
        st.session_state.chat_sid        = None
        st.session_state.chat_msgs       = []
        st.session_state.viewing_session = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat History ──────────────────────────────────────────────
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

    # ── Sign Out — pinned at the bottom of the sidebar ───────────
    st.markdown("""
    <div style="position:absolute;bottom:0;left:0;right:0;padding:0.75rem 0.85rem 1rem;
                border-top:1px solid rgba(255,255,255,0.06);background:#0e1018;">
    """, unsafe_allow_html=True)
    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("⏻  Sign Out", key="sidebar_signout", use_container_width=True):
        st.session_state["_do_signout"] = True
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  FIXED TOP NAV
# ─────────────────────────────────────────────────────────────────
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

# Handle sign-out (session_state flag — works on Streamlit Cloud, no query param needed)
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

# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
#  MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────
viewing       = st.session_state.get("viewing_session")
has_chat_msgs = bool(st.session_state.chat_msgs)

if not has_chat_msgs and not viewing:
    # ── WELCOME ──────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-outer" id="tw-welcome">
        <div class="welcome-orb">🧠</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-sub">
            Ask me anything about the documents uploaded to this workspace.<br>
            I'll give you precise, grounded answers — no hallucinations.
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
    # ── HISTORY DETAIL VIEW ───────────────────────────────────────
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
    # ── LIVE CHAT VIEW ────────────────────────────────────────────
    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────────────────────────
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
