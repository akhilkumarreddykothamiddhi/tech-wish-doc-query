import streamlit as st
import os, uuid, base64, random, string, smtplib, hashlib, time
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import snowflake.connector
import bcrypt as _bcrypt

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

SNOWFLAKE_ACCOUNT   = cfg("SNOWFLAKE_ACCOUNT",   "TCFIWLF-SJ78956")
SNOWFLAKE_USER      = cfg("SNOWFLAKE_USER",       "AKHILREDDY2")
SNOWFLAKE_PASSWORD  = cfg("SNOWFLAKE_PASSWORD",   "Reddy@1614421a")
SNOWFLAKE_WAREHOUSE = cfg("SNOWFLAKE_WAREHOUSE",  "COMPUTE_WH")
SNOWFLAKE_DATABASE  = cfg("SNOWFLAKE_DATABASE",   "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = cfg("SNOWFLAKE_SCHEMA",     "PUBLIC")
SNOWFLAKE_ROLE      = cfg("SNOWFLAKE_ROLE",       "ACCOUNTADMIN")

GROQ_API_KEY  = cfg("GROQ_API_KEY",  "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")
ALLOWED_DOMAIN = cfg("ALLOWED_DOMAIN", "techwish.com")

# ── Gmail SMTP (fill these in st.secrets or env) ──────────────────
SMTP_SENDER_EMAIL    = cfg("SMTP_SENDER_EMAIL",    "")   # your Gmail address
SMTP_APP_PASSWORD    = cfg("SMTP_APP_PASSWORD",    "")   # Gmail App Password

# ─────────────────────────────────────────────────────────────────
#  DOCS FOLDER & MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────
DOCS_FOLDER   = "docs"
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

NO_CONTEXT_MSG = "I'm sorry, I don't have information about that in the available documents."

OTP_EXPIRY_SECONDS = 600   # 10 minutes
OTP_LENGTH         = 8

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
Respond warmly, briefly, and professionally. Introduce yourself when relevant.
Mention you can answer questions about company documents, policies, and more.
Keep replies concise — 1 to 3 sentences max."""

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
#  GLOBAL CSS  (dark space theme, Outfit font)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: #050810 !important;
    color: #e8eaf0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

html, body { height: 100vh !important; }
.main .block-container {
    padding: 0 !important; max-width: 100% !important;
    height: 100vh !important; overflow-y: auto !important;
}
section[data-testid="stMain"] { height: 100vh !important; overflow-y: auto !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0b0f1e !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 270px !important; max-width: 290px !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    transition: all 0.3s ease !important;
    display: flex !important; flex-direction: column !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important; background: #0b0f1e !important;
    display: flex !important; flex-direction: column !important; height: 100% !important;
}
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important; top: 14px !important; right: 10px !important;
    z-index: 9999 !important; background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.30) !important; border-radius: 8px !important;
    color: #a5b4fc !important; padding: 4px 8px !important; font-size: 0.75rem !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
}
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    position: fixed !important; left: 0 !important; top: 50% !important;
    transform: translateY(-50%) !important; z-index: 9999 !important;
    background: rgba(99,102,241,0.18) !important; border: 1px solid rgba(99,102,241,0.4) !important;
    border-left: none !important; border-radius: 0 10px 10px 0 !important;
    padding: 14px 8px !important; cursor: pointer !important; transition: all 0.2s ease !important;
}

.sb-header {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: relative;
}
.sb-logo-wrap { display: flex; align-items: center; gap: 8px; margin-bottom: 0.35rem; }
.sb-logo-wrap img { max-width: 130px !important; height: auto !important; display: block !important; }
.sb-tagline {
    font-size: 0.72rem; color: rgba(255,255,255,0.35);
    font-weight: 400; letter-spacing: 0.01em; margin-top: 2px;
}
.sb-section { padding: 1rem 1.2rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sb-section-label {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: rgba(255,255,255,0.22); margin-bottom: 0.6rem;
}
.sb-bottom {
    padding: 0.8rem 1.2rem; margin-top: auto;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex; flex-direction: column; gap: 6px;
}
.user-pill {
    background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px; padding: 10px 12px; margin: 0 1.2rem 0.8rem;
    display: flex; align-items: center; gap: 10px;
}
.user-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: white; flex-shrink: 0;
}
.user-info { overflow: hidden; }
.user-name  { font-size: 0.8rem; font-weight: 600; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.user-email { font-size: 0.66rem; color: rgba(255,255,255,0.32); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.70) !important;
    border-radius: 10px !important; font-size: 0.82rem !important;
    padding: 0.52rem 0.85rem !important; text-align: left !important;
    transition: all 0.15s ease !important; box-shadow: none !important; width: 100% !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.32) !important; color: white !important;
}
.new-chat-btn div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.18)) !important;
    border: 1px solid rgba(99,102,241,0.32) !important;
    color: #c4b5fd !important; font-weight: 700 !important;
}
.clear-btn div[data-testid="stButton"] > button {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.20) !important;
    color: rgba(248,113,113,0.85) !important; font-weight: 600 !important;
}
.signout-btn div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.40) !important;
}
.signout-btn div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.09) !important;
    border-color: rgba(99,102,241,0.28) !important; color: #a5b4fc !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.6rem !important; text-transform: uppercase !important;
    letter-spacing: 0.13em !important; color: rgba(255,255,255,0.22) !important;
    margin: 0 0 0.5rem !important; font-weight: 700 !important; padding: 0 !important;
}

/* ── LOGIN PAGE ───────────────────────────────────────────────── */
.login-bg {
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(236,72,153,0.14) 0%, transparent 60%),
        #050810;
}
.orb { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; animation: float 8s ease-in-out infinite; }
.orb-1 { width:500px; height:500px; top:-100px; left:-100px; background:rgba(99,102,241,0.12); animation-delay:0s; }
.orb-2 { width:400px; height:400px; bottom:-80px; right:-80px; background:rgba(236,72,153,0.10); animation-delay:-3s; }
.orb-3 { width:300px; height:300px; top:40%; left:60%; background:rgba(14,165,233,0.08); animation-delay:-6s; }
@keyframes float {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(30px,-20px) scale(1.05); }
    66%      { transform: translate(-20px,30px) scale(0.95); }
}
.login-top-bar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 200;
    padding: 1rem 2rem; background: rgba(5,8,16,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(12px);
}
.login-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px; padding: 2.5rem 2.8rem 2rem;
    box-shadow: 0 40px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.1), inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(24px); text-align: center;
}
.login-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px; padding: 6px 16px; margin-bottom: 1.4rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #a5b4fc;
}
.login-title {
    font-size: 2.2rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.03em;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #fff 0%, rgba(165,180,252,0.9) 50%, rgba(236,72,153,0.8) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.login-sub {
    font-size: 0.88rem; color: rgba(255,255,255,0.38); margin-bottom: 1.6rem;
    font-weight: 300; line-height: 1.6;
}
.step-indicator {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    margin-bottom: 1.4rem;
}
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(255,255,255,0.15); transition: all 0.3s;
}
.step-dot.active { background: #6366f1; box-shadow: 0 0 8px rgba(99,102,241,0.6); width: 20px; border-radius: 4px; }
.step-dot.done   { background: #22c55e; }
.otp-display {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; text-align: left;
}
.otp-label { font-size: 0.68rem; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.otp-hint  { font-size: 0.78rem; color: rgba(255,255,255,0.5); line-height: 1.5; }
.otp-timer { font-size: 0.72rem; color: rgba(236,72,153,0.8); margin-top: 4px; }
.feature-row { display: flex; gap: 10px; margin-top: 1.2rem; }
.feature-chip {
    flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 10px 6px; font-size: 0.68rem;
    color: rgba(255,255,255,0.35); text-align: center;
}
.feature-chip .ficon { font-size: 1.1rem; margin-bottom: 3px; }

/* Streamlit input styling inside login */
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 12px !important; color: #111118 !important;
    font-family: 'Outfit', sans-serif !important; font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    background: #ffffff !important;
}
div[data-testid="stTextInput"] input::placeholder { color: rgba(17,17,24,0.35) !important; }
div[data-testid="stTextInput"] label { color: rgba(255,255,255,0.55) !important; font-size: 0.78rem !important; }

/* Primary button (login/submit) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: white !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    border-radius: 14px !important; padding: 0.8rem 1.5rem !important;
    border: none !important; width: 100% !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.4) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 48px rgba(99,102,241,0.55) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: rgba(255,255,255,0.6) !important;
    border-radius: 14px !important; font-size: 0.85rem !important;
    padding: 0.65rem 1.2rem !important; width: 100% !important;
    font-family: 'Outfit', sans-serif !important; transition: all 0.15s !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.3) !important; color: white !important;
}

/* ── TOPBAR ──────────────────────────────────────────────────── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 2rem; background: rgba(5,8,16,0.92);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(12px); flex-shrink: 0; z-index: 100;
    position: sticky; top: 0;
}
.topbar-left { display: flex; align-items: center; gap: 14px; }
.topbar-title {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.topbar-sub { font-size: 0.72rem; color: rgba(255,255,255,0.3); margin-top: 1px; }
.topbar-right { display: flex; align-items: center; gap: 8px; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.2), 0 0 8px rgba(34,197,94,0.4);
    animation: pulse-dot 2s ease-in-out infinite; flex-shrink: 0;
}
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 0 2px rgba(34,197,94,0.2), 0 0 8px rgba(34,197,94,0.4); }
    50%      { box-shadow: 0 0 0 4px rgba(34,197,94,0.1), 0 0 16px rgba(34,197,94,0.5); }
}
.topbar-user-email { font-size: 0.82rem; font-weight: 500; color: rgba(255,255,255,0.75); }

/* ── WELCOME SCREEN ──────────────────────────────────────────── */
.welcome-outer {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; min-height: calc(100vh - 80px); padding: 2rem;
}
.welcome-inner { max-width: 620px; width: 100%; }
.welcome-orb {
    width: 64px; height: 64px; border-radius: 20px; margin: 0 auto 1.2rem;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    display: flex; align-items: center; justify-content: center; font-size: 1.8rem;
    box-shadow: 0 16px 48px rgba(99,102,241,0.4), 0 0 0 1px rgba(99,102,241,0.3);
    animation: glow-orb 3s ease-in-out infinite;
}
@keyframes glow-orb {
    0%,100% { box-shadow: 0 16px 48px rgba(99,102,241,0.4), 0 0 0 1px rgba(99,102,241,0.3); }
    50%      { box-shadow: 0 20px 64px rgba(139,92,246,0.6), 0 0 0 1px rgba(139,92,246,0.4); }
}
.welcome-title {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 60%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;
}
.welcome-sub {
    font-size: 0.88rem; color: rgba(255,255,255,0.4); line-height: 1.6;
    margin-bottom: 1.8rem; font-weight: 300;
}
.starter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; width: 100%; }
.starter-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 12px 14px; text-align: left; cursor: pointer; transition: all 0.2s ease;
}
.starter-card:hover { background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.25); transform: translateY(-1px); }
.starter-icon { font-size: 1rem; margin-bottom: 4px; }
.starter-text { font-size: 0.74rem; color: rgba(255,255,255,0.5); line-height: 1.4; }

/* ── CHAT ────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.4rem 2rem !important; max-width: 860px !important; margin: 0 auto !important;
}
[data-testid="stChatMessageContent"] { border-radius: 18px !important; font-size: 0.9rem !important; line-height: 1.65 !important; }
[data-testid="chatAvatarIcon-user"]      { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; }
[data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; color: white !important; }

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 18px !important; backdrop-filter: blur(10px) !important;
    max-width: 860px !important; margin: 0 auto !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
}
[data-testid="stChatInput"] textarea { color: white !important; font-family: 'Outfit', sans-serif !important; font-size: 0.88rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color: rgba(255,255,255,0.25) !important; }
.stSpinner { color: #6366f1 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
#chat-bottom { height: 1px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SNOWFLAKE DATABASE
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = snowflake.connector.connect(
        account   = SNOWFLAKE_ACCOUNT,
        user      = SNOWFLAKE_USER,
        password  = SNOWFLAKE_PASSWORD,
        warehouse = SNOWFLAKE_WAREHOUSE,
        database  = SNOWFLAKE_DATABASE,
        schema    = SNOWFLAKE_SCHEMA,
        role      = SNOWFLAKE_ROLE,
    )
    _ensure_tables(conn)
    return conn

def _sf_exec(sql: str, params: tuple = ()):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
    finally:
        cur.close()

def _sf_fetch(sql: str, params: tuple = ()):
    cur = get_db().cursor()
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        cur.close()

def _ensure_tables(conn):
    cur = conn.cursor()
    # App users (email/password auth)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_users (
            user_id      VARCHAR(36)  PRIMARY KEY,
            email        VARCHAR(256) NOT NULL UNIQUE,
            full_name    VARCHAR(256) DEFAULT '',
            password_hash VARCHAR(512) NOT NULL,
            is_verified  BOOLEAN DEFAULT FALSE,
            created_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    # OTP store
    cur.execute("""
        CREATE TABLE IF NOT EXISTS otp_tokens (
            id          VARCHAR(36) PRIMARY KEY,
            email       VARCHAR(256) NOT NULL,
            otp_code    VARCHAR(32) NOT NULL,
            purpose     VARCHAR(32) NOT NULL,
            expires_at  NUMBER NOT NULL,
            used        BOOLEAN DEFAULT FALSE,
            created_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    # Chat sessions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         VARCHAR(36)  PRIMARY KEY,
            user_id    VARCHAR(36)  NOT NULL,
            user_email VARCHAR(256) NOT NULL,
            user_name  VARCHAR(256) DEFAULT '',
            title      VARCHAR(200) NOT NULL,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    # Chat messages
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            role       VARCHAR(20) NOT NULL,
            content    TEXT        NOT NULL,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.close()

# ── User helpers ──────────────────────────────────────────────────
def db_get_user(email: str):
    rows = _sf_fetch("SELECT user_id, email, full_name, password_hash, is_verified FROM app_users WHERE email = %s", (email,))
    if rows:
        r = rows[0]
        return {"user_id": r[0], "email": r[1], "full_name": r[2], "password_hash": r[3], "is_verified": r[4]}
    return None

def db_create_user(email: str, full_name: str, password_hash: str) -> str:
    uid = str(uuid.uuid4())
    _sf_exec(
        "INSERT INTO app_users (user_id, email, full_name, password_hash, is_verified) VALUES (%s,%s,%s,%s,%s)",
        (uid, email, full_name, password_hash, False)
    )
    return uid

def db_verify_user(email: str):
    _sf_exec("UPDATE app_users SET is_verified = TRUE WHERE email = %s", (email,))

# ── OTP helpers ───────────────────────────────────────────────────
def generate_otp() -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=OTP_LENGTH))

def db_save_otp(email: str, otp: str, purpose: str):
    # Invalidate old OTPs for same email+purpose
    _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE email=%s AND purpose=%s AND used=FALSE", (email, purpose))
    expires = int(time.time()) + OTP_EXPIRY_SECONDS
    _sf_exec(
        "INSERT INTO otp_tokens (id, email, otp_code, purpose, expires_at) VALUES (%s,%s,%s,%s,%s)",
        (str(uuid.uuid4()), email, otp, purpose, expires)
    )

def db_verify_otp(email: str, otp: str, purpose: str) -> bool:
    now = int(time.time())
    rows = _sf_fetch("""
        SELECT id FROM otp_tokens
        WHERE email=%s AND otp_code=%s AND purpose=%s AND used=FALSE AND expires_at > %s
        ORDER BY created_at DESC LIMIT 1
    """, (email, otp.strip().upper(), purpose, now))
    if rows:
        _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE id=%s", (rows[0][0],))
        return True
    return False

# ── Chat helpers ──────────────────────────────────────────────────
def db_sessions(user_id: str):
    rows = _sf_fetch("SELECT id, title, created_at FROM chat_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30", (user_id,))
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(user_id: str, email: str, name: str, title: str) -> str:
    sid = str(uuid.uuid4())
    _sf_exec("INSERT INTO chat_sessions (id, user_id, user_email, user_name, title) VALUES (%s,%s,%s,%s,%s)",
             (sid, user_id, email, name, title))
    return sid

def db_messages(session_id: str):
    rows = _sf_fetch("SELECT role, content FROM chat_messages WHERE session_id=%s ORDER BY created_at", (session_id,))
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save_msg(session_id: str, role: str, content: str):
    _sf_exec("INSERT INTO chat_messages (id, session_id, role, content) VALUES (%s,%s,%s,%s)",
             (str(uuid.uuid4()), session_id, role, content))

def db_delete_session(session_id: str):
    _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
    _sf_exec("DELETE FROM chat_sessions WHERE id=%s", (session_id,))

def db_delete_all_sessions(user_id: str):
    rows = _sf_fetch("SELECT id FROM chat_sessions WHERE user_id=%s", (user_id,))
    for row in rows:
        _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id=%s", (user_id,))

# ─────────────────────────────────────────────────────────────────
#  EMAIL (Gmail SMTP)
# ─────────────────────────────────────────────────────────────────
def send_otp_email(to_email: str, otp: str, purpose: str) -> bool:
    """Send OTP email via Gmail SMTP. Returns True on success."""
    sender    = SMTP_SENDER_EMAIL
    app_pass  = SMTP_APP_PASSWORD

    if not sender or not app_pass:
        # Dev fallback: show OTP on screen (remove in production)
        st.info(f"📧 **[DEV MODE — No SMTP configured]** Your OTP is: `{otp}`", icon="🔑")
        return True

    action = "verify your new account" if purpose == "register" else "sign in to your account"
    subject = f"Techwish AI — Your Login Code: {otp}"
    body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#050810;color:#e8eaf0;padding:32px;">
      <div style="max-width:480px;margin:0 auto;background:#0b0f1e;border-radius:18px;padding:32px;border:1px solid rgba(99,102,241,0.2);">
        <h2 style="color:#a5b4fc;margin-bottom:8px;">Techwish AI 🧠</h2>
        <p style="color:rgba(232,234,240,0.6);margin-bottom:24px;">Use the code below to {action}:</p>
        <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px 28px;text-align:center;margin-bottom:24px;">
          <span style="font-size:2.2rem;font-weight:800;letter-spacing:0.2em;color:#ffffff;font-family:monospace;">{otp}</span>
        </div>
        <p style="color:rgba(232,234,240,0.4);font-size:0.82rem;">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
        <hr style="border-color:rgba(255,255,255,0.07);margin:20px 0;">
        <p style="color:rgba(232,234,240,0.25);font-size:0.72rem;">Techwish AI Knowledge Assistant — Internal Use Only</p>
      </div>
    </body></html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_pass)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ─────────────────────────────────────────────────────────────────
#  AI & PDF SEARCH
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def embedder():
    return SentenceTransformer(EMBED_MODEL)

def pdf_text(path: str) -> str:
    doc   = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def chunk_text(text: str):
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return out

@st.cache_resource(show_spinner="📚 Building document index…")
def build_index():
    em     = embedder()
    folder = Path(DOCS_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        return None, [], []
    chunks, meta = [], []
    for p in pdfs:
        for c in chunk_text(pdf_text(str(p))):
            chunks.append(c)
            meta.append(p.name)
    embs = em.encode(chunks, batch_size=64, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, chunks, meta

def search(query: str, idx, chunks) -> tuple:
    if not idx:
        return "", False
    q = embedder().encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, TOP_K)
    SCORE_THRESHOLD = 0.30
    relevant = [chunks[i] for i, s in zip(ids[0], scores[0]) if i < len(chunks) and s >= SCORE_THRESHOLD]
    if not relevant:
        return "", False
    return "\n\n---\n\n".join(relevant), True

def ask_groq(messages: list, system: str) -> str:
    history = messages[-10:]
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [{"role": "system", "content": system}] + history,
        temperature = 0.0,
        max_tokens  = 1024,
    )
    return resp.choices[0].message.content

def ask_groq_smalltalk(prompt: str) -> str:
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "system", "content": SMALL_TALK_SYSTEM}, {"role": "user", "content": prompt}],
        temperature = 0.7, max_tokens = 200,
    )
    return resp.choices[0].message.content

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
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────
for k, v in {
    "user_id":    None,
    "user_email": None,
    "user_name":  None,
    "chat_sid":   None,
    "chat_msgs":  [],
    # Auth flow state
    "auth_step":       "email",   # email | register | login_pass | otp
    "auth_email":      "",
    "auth_purpose":    "",        # register | login
    "auth_temp_user":  None,      # temp user dict before OTP verified
    "auth_error":      "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
#  1. LOGIN / REGISTRATION FLOW
# ─────────────────────────────────────────────────────────────────
def show_login():
    st.markdown('<div class="login-bg"></div>', unsafe_allow_html=True)
    st.markdown('<div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>', unsafe_allow_html=True)

    b64 = logo_b64(LOGO_PATH)
    if b64:
        st.markdown(f'<div class="login-top-bar"><img src="{b64}" height="34" style="display:block;"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        step = st.session_state.auth_step

        # ── Step indicator ──────────────────────────────────────
        steps = ["email", "register" if step == "register" else "login_pass", "otp"]
        step_html = '<div class="step-indicator">'
        for s in ["email", "credentials", "otp"]:
            idx_map = {"email": 0, "credentials": 1, "otp": 2}
            cur_idx = {"email": 0, "register": 1, "login_pass": 1, "otp": 2}
            ci = cur_idx.get(step, 0)
            si = idx_map[s]
            cls = "active" if ci == si else ("done" if ci > si else "")
            step_html += f'<div class="step-dot {cls}"></div>'
        step_html += '</div>'

        # ── Card header ─────────────────────────────────────────
        titles = {
            "email":      ("🔐 Secure Access", "Your Intelligent Knowledge Hub", "Sign in with your @techwish.com email"),
            "register":   ("✨ Create Account", "Welcome to Techwish AI", "Set up your account to get started"),
            "login_pass": ("👋 Welcome Back", "Techwish AI", "Enter your password to continue"),
            "otp":        ("📧 Verify Your Email", "One Last Step", f"Enter the code sent to {st.session_state.auth_email}"),
        }
        badge, title, sub = titles.get(step, titles["email"])

        st.markdown(f"""
        <div class="login-card">
            <div class="login-badge">{badge}</div>
            <div class="login-title">{title}</div>
            <div class="login-sub">{sub}</div>
            {step_html}
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)
            st.session_state.auth_error = ""

        # ════════════════════════════════════════════════════════
        #  STEP 1 — Email entry
        # ════════════════════════════════════════════════════════
        if step == "email":
            email = st.text_input("Work Email", placeholder="you@techwish.com", key="inp_email")
            st.markdown("")
            if st.button("Continue →", type="primary", use_container_width=True):
                email = email.strip().lower()
                if not email:
                    st.session_state.auth_error = "Please enter your email address."
                elif not email.endswith(f"@{ALLOWED_DOMAIN}"):
                    st.session_state.auth_error = f"Only @{ALLOWED_DOMAIN} email addresses are allowed."
                else:
                    user = db_get_user(email)
                    st.session_state.auth_email = email
                    if user:
                        st.session_state.auth_step    = "login_pass"
                        st.session_state.auth_purpose = "login"
                    else:
                        st.session_state.auth_step    = "register"
                        st.session_state.auth_purpose = "register"
                    st.rerun()

            st.markdown("""
            <div class="feature-row">
                <div class="feature-chip"><div class="ficon">📄</div>PDF-powered</div>
                <div class="feature-chip"><div class="ficon">⚡</div>Instant answers</div>
                <div class="feature-chip"><div class="ficon">🔒</div>@techwish.com only</div>
            </div>
            """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════
        #  STEP 2a — Registration (new user)
        # ════════════════════════════════════════════════════════
        elif step == "register":
            st.markdown(f"<p style='font-size:0.8rem;color:rgba(255,255,255,0.4);margin-bottom:0.4rem;'>📧 {st.session_state.auth_email}</p>", unsafe_allow_html=True)
            full_name = st.text_input("Full Name", placeholder="Your full name", key="inp_name")
            password  = st.text_input("Create Password", type="password", placeholder="Min 8 characters", key="inp_pass_reg")
            confirm   = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="inp_pass_conf")
            st.markdown("")

            if st.button("Create Account & Send OTP →", type="primary", use_container_width=True):
                if not full_name.strip():
                    st.session_state.auth_error = "Please enter your full name."
                elif len(password) < 8:
                    st.session_state.auth_error = "Password must be at least 8 characters."
                elif password != confirm:
                    st.session_state.auth_error = "Passwords do not match."
                else:
                    pw_hash = _bcrypt.hashpw(password[:72].encode(), _bcrypt.gensalt()).decode()
                    # Store temp user data, create unverified account
                    uid = db_create_user(st.session_state.auth_email, full_name.strip(), pw_hash)
                    otp = generate_otp()
                    db_save_otp(st.session_state.auth_email, otp, "register")
                    sent = send_otp_email(st.session_state.auth_email, otp, "register")
                    if sent:
                        st.session_state.auth_temp_user = {"user_id": uid, "full_name": full_name.strip()}
                        st.session_state.auth_step = "otp"
                        st.rerun()

            st.markdown("")
            if st.button("← Back", type="secondary", use_container_width=True):
                st.session_state.auth_step = "email"
                st.rerun()

        # ════════════════════════════════════════════════════════
        #  STEP 2b — Login (existing user, enter password)
        # ════════════════════════════════════════════════════════
        elif step == "login_pass":
            st.markdown(f"<p style='font-size:0.8rem;color:rgba(255,255,255,0.4);margin-bottom:0.4rem;'>📧 {st.session_state.auth_email}</p>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Your password", key="inp_pass_login")
            st.markdown("")

            if st.button("Send OTP →", type="primary", use_container_width=True):
                user = db_get_user(st.session_state.auth_email)
                if not user:
                    st.session_state.auth_error = "Account not found."
                elif not _bcrypt.checkpw(password[:72].encode(), user["password_hash"].encode()):
                    st.session_state.auth_error = "Incorrect password. Please try again."
                else:
                    otp  = generate_otp()
                    db_save_otp(st.session_state.auth_email, otp, "login")
                    sent = send_otp_email(st.session_state.auth_email, otp, "login")
                    if sent:
                        st.session_state.auth_temp_user = user
                        st.session_state.auth_step = "otp"
                        st.rerun()

            st.markdown("")
            if st.button("← Back", type="secondary", use_container_width=True):
                st.session_state.auth_step = "email"
                st.rerun()

        # ════════════════════════════════════════════════════════
        #  STEP 3 — OTP verification
        # ════════════════════════════════════════════════════════
        elif step == "otp":
            st.markdown(f"""
            <div class="otp-display">
                <div class="otp-label">📨 Code sent</div>
                <div class="otp-hint">Check your inbox at <strong>{st.session_state.auth_email}</strong><br>
                Enter the {OTP_LENGTH}-character code below.</div>
                <div class="otp-timer">⏱ Expires in 10 minutes</div>
            </div>
            """, unsafe_allow_html=True)

            otp_input = st.text_input("Verification Code", placeholder="e.g. A3X9KP2B", key="inp_otp",
                                      max_chars=OTP_LENGTH)
            st.markdown("")

            if st.button("Verify & Sign In ✓", type="primary", use_container_width=True):
                purpose = st.session_state.auth_purpose
                valid   = db_verify_otp(st.session_state.auth_email, otp_input, purpose)
                if not valid:
                    st.session_state.auth_error = "Invalid or expired code. Please try again."
                    st.rerun()
                else:
                    # Mark verified if registering
                    if purpose == "register":
                        db_verify_user(st.session_state.auth_email)
                    tu = st.session_state.auth_temp_user
                    st.session_state.user_id    = tu["user_id"]
                    st.session_state.user_email = st.session_state.auth_email
                    st.session_state.user_name  = tu.get("full_name", st.session_state.auth_email.split("@")[0])
                    # Reset auth state
                    st.session_state.auth_step      = "email"
                    st.session_state.auth_email     = ""
                    st.session_state.auth_temp_user = None
                    st.rerun()

            st.markdown("")
            cols = st.columns(2)
            with cols[0]:
                if st.button("Resend Code", type="secondary", use_container_width=True):
                    otp  = generate_otp()
                    db_save_otp(st.session_state.auth_email, otp, st.session_state.auth_purpose)
                    send_otp_email(st.session_state.auth_email, otp, st.session_state.auth_purpose)
                    st.success("New code sent!")
            with cols[1]:
                if st.button("← Back", type="secondary", use_container_width=True):
                    st.session_state.auth_step = "email"
                    st.rerun()

if not st.session_state.user_id:
    show_login()
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  2. BUILD DOCUMENT INDEX
# ─────────────────────────────────────────────────────────────────
doc_idx, doc_chunks, doc_meta = build_index()

# ─────────────────────────────────────────────────────────────────
#  3. SIDEBAR
# ─────────────────────────────────────────────────────────────────
name     = st.session_state.user_name
email    = st.session_state.user_email
initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"

with st.sidebar:
    b64 = logo_b64(LOGO_PATH)
    logo_html_sb = f'<div class="sb-logo-wrap"><img src="{b64}" height="30"></div>' if b64 else \
                   f'<div class="sb-logo-wrap"><span style="font-weight:800;color:white;">{COMPANY_NAME}</span></div>'
    st.markdown(f"""
    <div class="sb-header">
        {logo_html_sb}
        <div class="sb-tagline">Ask anything about your documents</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="user-pill" style="margin-top:0.9rem;">
        <div class="user-avatar">{initials}</div>
        <div class="user-info">
            <div class="user-name">{name}</div>
            <div class="user-email">{email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn" style="padding:0 1.2rem 0.6rem;">', unsafe_allow_html=True)
    if st.button("✦  New Conversation", use_container_width=True):
        st.session_state.chat_sid  = None
        st.session_state.chat_msgs = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    pdf_count = len(list(Path(DOCS_FOLDER).glob("*.pdf"))) if Path(DOCS_FOLDER).exists() else 0
    badge_color = "rgba(255,255,255,0.28)" if pdf_count else "rgba(236,72,153,0.55)"
    badge_text  = f"📚 {pdf_count} document{'s' if pdf_count != 1 else ''} loaded" if pdf_count else "⚠️ No PDFs in docs/ folder"
    st.markdown(f'<p style="font-size:0.68rem;color:{badge_color};margin:0 0 0.2rem;text-align:center;padding:0 1.2rem;">{badge_text}</p>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">Chat History</div>', unsafe_allow_html=True)
    sessions = db_sessions(st.session_state.user_id)
    if not sessions:
        st.markdown('<p style="font-size:0.75rem;color:rgba(255,255,255,0.22);padding:0.2rem 0 0.4rem;">No conversations yet.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    for s in sessions:
        col1, col2 = st.columns([5, 1])
        label = s["title"][:22] + ("…" if len(s["title"]) > 22 else "")
        with col1:
            if st.button(f"💬  {label}", key=f"s_{s['id']}", use_container_width=True):
                st.session_state.chat_sid  = s["id"]
                st.session_state.chat_msgs = db_messages(s["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"d_{s['id']}"):
                db_delete_session(s["id"])
                if st.session_state.chat_sid == s["id"]:
                    st.session_state.chat_sid  = None
                    st.session_state.chat_msgs = []
                st.rerun()

    st.markdown('<div class="sb-bottom">', unsafe_allow_html=True)
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️  Clear All Chats", use_container_width=True):
        db_delete_all_sessions(st.session_state.user_id)
        st.session_state.chat_sid  = None
        st.session_state.chat_msgs = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("⎋  Sign Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  4. TOPBAR
# ─────────────────────────────────────────────────────────────────
b64_top = logo_b64(LOGO_PATH)
logo_html = f'<img src="{b64_top}" height="30" style="display:block;">' if b64_top else f'<span style="font-weight:800;">{COMPANY_NAME}</span>'
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        {logo_html}
        <div>
            <div class="topbar-title">Knowledge Assistant</div>
            <div class="topbar-sub">Powered by your documents</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="status-dot"></div>
        <span class="topbar-user-email">{email}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  5. CHAT AREA
# ─────────────────────────────────────────────────────────────────
if not st.session_state.chat_msgs:
    st.markdown("""
    <div class="welcome-outer">
      <div class="welcome-inner">
        <div class="welcome-orb">🧠</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-sub">
            Ask me anything about the documents uploaded to this workspace.<br>
            I'll give you precise, grounded answers — no hallucinations.
        </div>
        <div class="starter-grid">
            <div class="starter-card"><div class="starter-icon">📋</div><div class="starter-text">Summarize the key points from the uploaded documents</div></div>
            <div class="starter-card"><div class="starter-icon">🔍</div><div class="starter-text">Find specific information or policies in the docs</div></div>
            <div class="starter-card"><div class="starter-icon">❓</div><div class="starter-text">Ask any question and get a document-backed answer</div></div>
            <div class="starter-card"><div class="starter-icon">📊</div><div class="starter-text">Compare or contrast topics across multiple documents</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

for m in st.session_state.chat_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything about your documents…"):
    if not st.session_state.chat_sid:
        st.session_state.chat_sid = db_new_session(
            st.session_state.user_id, email, name, prompt[:50]
        )
    st.session_state.chat_msgs.append({"role": "user", "content": prompt})
    db_save_msg(st.session_state.chat_sid, "user", prompt)
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
        db_save_msg(st.session_state.chat_sid, "assistant", answer)
