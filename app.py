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
APP_URL           = cfg("APP_URL", "http://10.10.31.110:8501")
ALLOWED_DOMAIN    = cfg("ALLOWED_DOMAIN", "techwish.com")

# ─────────────────────────────────────────────────────────────────
#  DOCS FOLDER
# ─────────────────────────────────────────────────────────────────
DOCS_FOLDER   = "docs"

GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

NO_CONTEXT_MSG = "I'm sorry, I don't have information about that in the available documents."

# ─────────────────────────────────────────────────────────────────
#  SMALL TALK / GREETING DETECTION
# ─────────────────────────────────────────────────────────────────
SMALL_TALK_KEYWORDS = [
    "hi", "hello", "hey", "hru", "how are you", "how r u", "good morning",
    "good afternoon", "good evening", "good night", "what's up", "whats up",
    "sup", "howdy", "greetings", "thanks", "thank you", "thank u", "ty",
    "bye", "goodbye", "see you", "take care", "who are you", "what are you",
    "what can you do", "what do you do", "help me", "how can you help",
    "introduce yourself", "tell me about yourself", "nice to meet you",
    "nice", "ok", "okay", "cool", "great", "awesome", "wow", "lol", "haha",
    "good", "bad", "sad", "happy", "fine", "alright", "sure", "yes", "no",
    "yep", "nope", "please", "sorry", "excuse me", "pardon"
]

def is_small_talk(text: str) -> bool:
    t = text.lower().strip()
    # Short messages (under 6 words) that don't look like document queries
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
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
    overflow-y: auto !important;
}
section[data-testid="stMain"] { height: 100vh !important; overflow-y: auto !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ═══ SIDEBAR ══════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #0b0f1e !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 270px !important;
    max-width: 290px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    flex-direction: column !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
    background: #0b0f1e !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

/* ── Collapse toggle (<<) top-right of sidebar ── */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 14px !important;
    right: 10px !important;
    z-index: 9999 !important;
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.30) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    padding: 4px 8px !important;
    font-size: 0.75rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
button[data-testid="baseButton-headerNoPadding"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(99,102,241,0.25) !important;
    color: white !important;
}

/* Collapsed toggle — floating tab on left edge */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 9999 !important;
    background: rgba(99,102,241,0.18) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-left: none !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 14px 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 4px 0 20px rgba(99,102,241,0.2) !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(99,102,241,0.35) !important;
    padding-right: 14px !important;
}
[data-testid="collapsedControl"] svg { fill: #a5b4fc !important; }

/* ── Sidebar logo + tagline block ── */
.sb-header {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: relative;
}
.sb-logo-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
}
.sb-logo-wrap img {
    max-width: 130px !important;
    height: auto !important;
    display: block !important;
}
.sb-logo-text {
    font-size: 1.15rem;
    font-weight: 800;
    color: white;
    letter-spacing: -0.02em;
}
.sb-tagline {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    font-weight: 400;
    letter-spacing: 0.01em;
    margin-top: 2px;
}

/* ── Sidebar sections ── */
.sb-section {
    padding: 1rem 1.2rem 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-section-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.22);
    margin-bottom: 0.6rem;
}
.sb-bottom {
    padding: 0.8rem 1.2rem;
    margin-top: auto;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    flex-direction: column;
    gap: 6px;
}

/* ── User pill ── */
.user-pill {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    padding: 10px 12px;
    margin: 0 1.2rem 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
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

/* ── All sidebar buttons reset ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.70) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    padding: 0.52rem 0.85rem !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.32) !important;
    color: white !important;
}

/* New conversation */
.new-chat-btn div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.18)) !important;
    border: 1px solid rgba(99,102,241,0.32) !important;
    color: #c4b5fd !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}
.new-chat-btn div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.30), rgba(139,92,246,0.28)) !important;
    color: white !important;
}

/* Chat history items */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
    text-align: left !important;
    justify-content: flex-start !important;
}

/* Clear chat — subtle red */
.clear-btn div[data-testid="stButton"] > button {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.20) !important;
    color: rgba(248,113,113,0.85) !important;
    font-weight: 600 !important;
}
.clear-btn div[data-testid="stButton"] > button:hover {
    background: rgba(239,68,68,0.16) !important;
    border-color: rgba(239,68,68,0.45) !important;
    color: #fca5a5 !important;
}

/* Sign out */
.signout-btn div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.40) !important;
}
.signout-btn div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.09) !important;
    border-color: rgba(99,102,241,0.28) !important;
    color: #a5b4fc !important;
}

/* Section headings inside sidebar */
section[data-testid="stSidebar"] h3 {
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: rgba(255,255,255,0.22) !important;
    margin: 0 0 0.5rem !important;
    font-weight: 700 !important;
    padding: 0 !important;
}

/* ═══ LOGIN PAGE ════════════════════════════════════════════════ */
.login-bg {
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(236,72,153,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(14,165,233,0.08) 0%, transparent 70%),
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
    padding: 1rem 2rem;
    background: rgba(5,8,16,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
}
.login-card-html {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none;
    border-radius: 28px 28px 0 0;
    padding: 3rem 3rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.1), inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(24px);
}
.login-card-bottom {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: none;
    border-radius: 0 0 28px 28px;
    padding: 0.5rem 3rem 2.5rem;
    box-shadow: 0 40px 80px rgba(0,0,0,0.5), inset 0 -1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(24px);
}
.login-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px; padding: 6px 16px; margin-bottom: 1.8rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #a5b4fc;
}
.login-title {
    font-size: 2.8rem; font-weight: 800; line-height: 1.1;
    letter-spacing: -0.03em; margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #fff 0%, rgba(165,180,252,0.9) 50%, rgba(236,72,153,0.8) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.login-sub {
    font-size: 0.95rem; color: rgba(255,255,255,0.4);
    margin-bottom: 1.2rem; font-weight: 300; line-height: 1.6;
}
.login-divider-line { height: 1px; background: rgba(255,255,255,0.07); margin-bottom: 0; }
.feature-row { display: flex; gap: 12px; margin-top: 1.2rem; }
.feature-chip {
    flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px 8px; font-size: 0.7rem;
    color: rgba(255,255,255,0.4); text-align: center;
}
.feature-chip .ficon { font-size: 1.2rem; margin-bottom: 4px; }
.login-btn-wrap {
    background: rgba(255,255,255,0.03);
    border-left: 1px solid rgba(255,255,255,0.08);
    border-right: 1px solid rgba(255,255,255,0.08);
    padding: 1.4rem 3rem;
    position: relative; z-index: 500;
}
.login-btn-wrap div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: white !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    border-radius: 14px !important; padding: 0.85rem 1.5rem !important;
    border: none !important; cursor: pointer !important; width: 100% !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.45), 0 2px 0 rgba(255,255,255,0.12) inset !important;
    transition: all 0.2s ease !important; z-index: 500 !important;
}
.login-btn-wrap div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 48px rgba(99,102,241,0.6), 0 2px 0 rgba(255,255,255,0.15) inset !important;
}

div[data-testid="stButton"] > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    border-radius: 14px !important; padding: 0.8rem 1.5rem !important;
    border: none !important; cursor: pointer !important;
    transition: all 0.2s ease !important; width: 100% !important;
}

/* ═══ TOPBAR ════════════════════════════════════════════════════ */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 2rem;
    background: rgba(5,8,16,0.92);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    flex-shrink: 0; z-index: 100;
    position: sticky; top: 0;
}
.topbar-left { display: flex; align-items: center; gap: 14px; }
.topbar-title {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.01em;
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
.topbar-user-email {
    font-size: 0.82rem; font-weight: 500;
    color: rgba(255,255,255,0.75);
    white-space: nowrap;
}

/* ═══ WELCOME SCREEN ════════════════════════════════════════════ */
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
    font-size: 0.88rem; color: rgba(255,255,255,0.4);
    line-height: 1.6; margin-bottom: 1.8rem; font-weight: 300;
}
.starter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; width: 100%; }
.starter-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 12px 14px; text-align: left;
    cursor: pointer; transition: all 0.2s ease;
}
.starter-card:hover { background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.25); transform: translateY(-1px); }
.starter-icon { font-size: 1rem; margin-bottom: 4px; }
.starter-text { font-size: 0.74rem; color: rgba(255,255,255,0.5); line-height: 1.4; }

/* ═══ CHAT MESSAGES ═════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.4rem 2rem !important; max-width: 860px !important; margin: 0 auto !important;
}
[data-testid="stChatMessageContent"] {
    border-radius: 18px !important; font-size: 0.9rem !important; line-height: 1.65 !important;
}
[data-testid="chatAvatarIcon-user"]      { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; }
[data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; color: white !important; }

/* ═══ CHAT INPUT ══════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(10px) !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
}
[data-testid="stChatInput"] textarea {
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(255,255,255,0.25) !important; }
.stSpinner { color: #6366f1 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Auto-scroll anchor */
#chat-bottom { height: 1px; }
</style>

<!-- Auto-scroll to bottom on new message -->
<script>
function scrollToBottom() {
    const anchor = document.getElementById('chat-bottom');
    if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth' });
    } else {
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
}
// Run on load and observe DOM changes
const observer = new MutationObserver(() => scrollToBottom());
observer.observe(document.body, { childList: true, subtree: true });
window.addEventListener('load', scrollToBottom);
</script>
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    VARCHAR(256) PRIMARY KEY,
            email      VARCHAR(256) NOT NULL UNIQUE,
            full_name  VARCHAR(256) DEFAULT '',
            department VARCHAR(256) DEFAULT 'General',
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         VARCHAR(36)  PRIMARY KEY,
            user_id    VARCHAR(256) NOT NULL,
            user_email VARCHAR(256) NOT NULL,
            user_name  VARCHAR(256) DEFAULT '',
            title      VARCHAR(200) NOT NULL,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            role       VARCHAR(20) NOT NULL,
            content    TEXT        NOT NULL,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            id          VARCHAR(36)  PRIMARY KEY,
            filename    VARCHAR(500) NOT NULL,
            uploaded_by VARCHAR(256),
            chunk_count INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.close()

def db_upsert_user(user_id: str, email: str, name: str):
    _sf_exec("""
        MERGE INTO users AS t
        USING (SELECT %s AS user_id, %s AS email, %s AS full_name) AS s
        ON t.user_id = s.user_id
        WHEN NOT MATCHED THEN
            INSERT (user_id, email, full_name) VALUES (s.user_id, s.email, s.full_name)
    """, (user_id, email, name))

def db_sessions(user_id: str):
    rows = _sf_fetch("""
        SELECT id, title, created_at FROM chat_sessions
        WHERE user_id = %s ORDER BY created_at DESC LIMIT 30
    """, (user_id,))
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(user_id: str, email: str, name: str, title: str) -> str:
    sid = str(uuid.uuid4())
    _sf_exec("""
        INSERT INTO chat_sessions (id, user_id, user_email, user_name, title)
        VALUES (%s, %s, %s, %s, %s)
    """, (sid, user_id, email, name, title))
    return sid

def db_messages(session_id: str):
    rows = _sf_fetch("""
        SELECT role, content FROM chat_messages
        WHERE session_id = %s ORDER BY created_at
    """, (session_id,))
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save(session_id: str, role: str, content: str):
    _sf_exec("""
        INSERT INTO chat_messages (id, session_id, role, content)
        VALUES (%s, %s, %s, %s)
    """, (str(uuid.uuid4()), session_id, role, content))

def db_delete(session_id: str):
    _sf_exec("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
    _sf_exec("DELETE FROM chat_sessions WHERE id = %s", (session_id,))

def db_delete_all_sessions(user_id: str):
    """Delete ALL chat sessions and messages for a user."""
    rows = _sf_fetch("SELECT id FROM chat_sessions WHERE user_id = %s", (user_id,))
    for row in rows:
        _sf_exec("DELETE FROM chat_messages WHERE session_id = %s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id = %s", (user_id,))

def db_log_document(filename: str, uploaded_by: str, chunk_count: int):
    _sf_exec("""
        INSERT INTO document_metadata (id, filename, uploaded_by, chunk_count)
        VALUES (%s, %s, %s, %s)
    """, (str(uuid.uuid4()), filename, uploaded_by, chunk_count))

# ─────────────────────────────────────────────────────────────────
#  AI & PDF SEARCH
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

def search(query: str, idx, chunks) -> tuple[str, bool]:
    if not idx:
        return "", False
    q = embedder().encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, TOP_K)

    SCORE_THRESHOLD = 0.30
    relevant = [
        chunks[i]
        for i, s in zip(ids[0], scores[0])
        if i < len(chunks) and s >= SCORE_THRESHOLD
    ]
    if not relevant:
        return "", False
    return "\n\n---\n\n".join(relevant), True

def ask_groq(messages: list, system: str) -> str:
    history = messages[-10:]
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "system", "content": system}] + history,
        temperature = 0.0,
        max_tokens  = 1024,
    )
    return resp.choices[0].message.content

def ask_groq_smalltalk(prompt: str) -> str:
    """Dedicated call for small talk — no document context needed."""
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SMALL_TALK_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature = 0.7,
        max_tokens  = 200,
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────────────────────────────
#  SESSION INIT
# ─────────────────────────────────────────────────────────────────
for k, v in {
    "user_id": None, "user_email": None, "user_name": None,
    "chat_sid": None, "chat_msgs": [],
    "logged_out": False,   # FIX: track explicit logout
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
#  OAUTH CALLBACK — only process if not logged out
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
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Auth error: {e}")

# ─────────────────────────────────────────────────────────────────
#  1. LOGIN PAGE
# ─────────────────────────────────────────────────────────────────
if not st.session_state.user_id:
    # If there's an OAuth code in URL but user already logged out, clear it
    if "code" in st.query_params and st.session_state.logged_out:
        st.query_params.clear()

    st.markdown('<div class="login-bg"></div>', unsafe_allow_html=True)
    st.markdown('<div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>', unsafe_allow_html=True)

    b64 = logo_b64(LOGO_PATH)
    if b64:
        st.markdown(f'<div class="login-top-bar"><img src="{b64}" height="36" style="display:block;"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="login-card-html">
            <div class="login-badge">🔐 &nbsp;Secure Workspace</div>
            <div class="login-title">Your Intelligent<br>Knowledge Hub</div>
            <div class="login-sub">
                Ask anything. Get instant, precise answers<br>
                drawn directly from your organization's documents.
            </div>
            <div class="login-divider-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-btn-wrap">', unsafe_allow_html=True)
if st.button("🔑 Continue with Google", type="primary", use_container_width=True):
    # This generates the URL specifically for your Supabase project
    res = supabase_client().auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": APP_URL,  # https://docquey-techwish.streamlit.app
            "scopes": "email profile",
            "query_params": {"prompt": "select_account"}
        }
    })
    
    if res.url:
        # DO NOT use st.markdown redirect or meta-refresh. 
        # This JS is the only way to avoid 403s on Streamlit Cloud:
        st.components.v1.html(f"""
            <script>
                window.top.location.href = "{res.url}";
            </script>
        """, height=0)
        st.stop()
# ─────────────────────────────────────────────────────────────────
#  2. BUILD INDEX
# ─────────────────────────────────────────────────────────────────
doc_idx, doc_chunks, doc_meta = build_index()

# ─────────────────────────────────────────────────────────────────
#  3. SIDEBAR — collapsible/expandable, with logo, history, actions
# ─────────────────────────────────────────────────────────────────
name     = st.session_state.user_name
email    = st.session_state.user_email
initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"

with st.sidebar:
    b64 = logo_b64(LOGO_PATH)

    # ── Header: Logo + tagline ────────────────────────────────────
    if b64:
        logo_html_sb = f'<div class="sb-logo-wrap"><img src="{b64}" height="30"></div>'
    else:
        logo_html_sb = f'<div class="sb-logo-wrap"><span class="sb-logo-text">{COMPANY_NAME}</span></div>'

    st.markdown(f"""
    <div class="sb-header">
        {logo_html_sb}
        <div class="sb-tagline">Ask anything about your documents</div>
    </div>
    """, unsafe_allow_html=True)

    # ── User pill ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="user-pill" style="margin-top:0.9rem;">
        <div class="user-avatar">{initials}</div>
        <div class="user-info">
            <div class="user-name">{name}</div>
            <div class="user-email">{email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── New Conversation ──────────────────────────────────────────
    st.markdown('<div class="new-chat-btn" style="padding:0 1.2rem 0.6rem;">', unsafe_allow_html=True)
    if st.button("✦  New Conversation", use_container_width=True):
        st.session_state.chat_sid  = None
        st.session_state.chat_msgs = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PDF count badge ───────────────────────────────────────────
    pdf_count = len(list(Path(DOCS_FOLDER).glob("*.pdf"))) if Path(DOCS_FOLDER).exists() else 0
    if pdf_count:
        st.markdown(
            f'<p style="font-size:0.68rem;color:rgba(255,255,255,0.28);margin:0 0 0.2rem;text-align:center;padding:0 1.2rem;">📚 {pdf_count} document{"s" if pdf_count != 1 else ""} loaded</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="font-size:0.68rem;color:rgba(236,72,153,0.55);margin:0 0 0.2rem;text-align:center;padding:0 1.2rem;">⚠️ No PDFs in docs/ folder</p>',
            unsafe_allow_html=True
        )

    # ── Chat History ──────────────────────────────────────────────
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">Chat History</div>', unsafe_allow_html=True)
    sessions = db_sessions(st.session_state.user_id)
    if not sessions:
        st.markdown(
            '<p style="font-size:0.75rem;color:rgba(255,255,255,0.22);padding:0.2rem 0 0.4rem;">No conversations yet.</p>',
            unsafe_allow_html=True
        )
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
                db_delete(s["id"])
                if st.session_state.chat_sid == s["id"]:
                    st.session_state.chat_sid  = None
                    st.session_state.chat_msgs = []
                st.rerun()

    # ── Bottom actions: Clear Chat + Sign Out ─────────────────────
    st.markdown('<div class="sb-bottom">', unsafe_allow_html=True)

    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️  Clear Chat", use_container_width=True):
        db_delete_all_sessions(st.session_state.user_id)
        st.session_state.chat_sid  = None
        st.session_state.chat_msgs = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("⎋  Sign Out", use_container_width=True):
        try:
            supabase_client().auth.sign_out()
        except Exception:
            pass
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.logged_out = True
        st.session_state.user_id    = None
        st.session_state.user_email = None
        st.session_state.user_name  = None
        st.session_state.chat_sid   = None
        st.session_state.chat_msgs  = []
        st.query_params.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close sb-bottom

# ─────────────────────────────────────────────────────────────────
#  4. TOPBAR — green dot + name only (removed "Online" label)
# ─────────────────────────────────────────────────────────────────
b64_top   = logo_b64(LOGO_PATH)
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
#  5. SYSTEM PROMPT (document questions only)
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
#  6. CHAT AREA
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

# Scroll anchor — always rendered at the bottom of the chat
st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything about your documents…"):
    if not st.session_state.chat_sid:
        st.session_state.chat_sid = db_new_session(
            st.session_state.user_id, email, name, prompt[:50]
        )

    st.session_state.chat_msgs.append({"role": "user", "content": prompt})
    db_save(st.session_state.chat_sid, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # ── Small talk / greeting path ────────────────────────
            if is_small_talk(prompt):
                answer = ask_groq_smalltalk(prompt)
            else:
                # ── Document RAG path ─────────────────────────────
                context, has_context = search(prompt, doc_idx, doc_chunks)
                if not has_context:
                    answer = NO_CONTEXT_MSG
                else:
                    system = SYSTEM_PROMPT.format(context=context)
                    answer = ask_groq(st.session_state.chat_msgs, system)

        st.markdown(answer)
        st.session_state.chat_msgs.append({"role": "assistant", "content": answer})
        db_save(st.session_state.chat_sid, "assistant", answer)

    # Auto-scroll: inject JS to scroll to bottom after new message
    st.markdown(
        '<script>document.getElementById("chat-bottom")?.scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True
    )
