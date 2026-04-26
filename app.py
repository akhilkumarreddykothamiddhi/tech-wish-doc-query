"""
TechWish DocQuery — Flask Backend
Serves index.html (SPA) and exposes REST API consumed by the HTML frontend.

FIX: Replaced sentence-transformers (requires PyTorch/c10.dll) with
     fastembed which uses ONNX Runtime — no torch dependency, works on
     Windows even with Application Control policies blocking torch DLLs.
FIX: OTP codes are no longer printed to the terminal. In dev mode (no
     SMTP credentials) the code is written to otp_dev.log instead.
FIX: All secrets removed from source code — loaded exclusively from .env
"""

import os, uuid, random, string, smtplib, time, logging
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, redirect, url_for
)
from groq import Groq
import fitz                        # PyMuPDF
# from fastembed import TextEmbedding
# import numpy as np
import snowflake.connector
import bcrypt as _bcrypt

# ─────────────────────────────────────────────────────────────────
#  LOAD .env (only if python-dotenv is installed; harmless otherwise)
# ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on environment being pre-populated

# ─────────────────────────────────────────────────────────────────
#  LOGGING — suppress noisy library output; OTPs go to file only
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

_otp_logger = logging.getLogger("otp_dev")
_otp_logger.setLevel(logging.DEBUG)
_otp_handler = logging.FileHandler("otp_dev.log")
_otp_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_otp_logger.addHandler(_otp_handler)
_otp_logger.propagate = False

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION  — all values come from environment / .env file
#  No defaults are provided for secrets; the app will fail fast if
#  a required variable is missing.
# ─────────────────────────────────────────────────────────────────
def _require(key: str) -> str:
    """Return env var value or raise a clear error at startup."""
    v = os.environ.get(key, "").strip()
    if not v:
        raise EnvironmentError(
            f"[CONFIG] Required environment variable '{key}' is not set. "
            f"Add it to your .env file or system environment."
        )
    return v

def _optional(key: str, default: str = "") -> str:
    v = os.environ.get(key, default)
    return v.strip() if isinstance(v, str) else v

# ── Snowflake (required) ──────────────────────────────────────────
SNOWFLAKE_ACCOUNT   = _require("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = _require("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = _require("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = _optional("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE  = _optional("SNOWFLAKE_DATABASE",  "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = _optional("SNOWFLAKE_SCHEMA",     "PUBLIC")
SNOWFLAKE_ROLE      = _optional("SNOWFLAKE_ROLE",       "ACCOUNTADMIN")

# ── Groq (required) ───────────────────────────────────────────────
GROQ_API_KEY = _require("GROQ_API_KEY")

# ── App settings (required) ───────────────────────────────────────
FLASK_SECRET   = _require("FLASK_SECRET")
ALLOWED_DOMAIN = _optional("ALLOWED_DOMAIN", "techwish.com")

# ── SMTP (optional — dev mode writes OTPs to otp_dev.log) ────────
SMTP_SENDER_EMAIL = _optional("SMTP_SENDER_EMAIL")
SMTP_APP_PASSWORD = _optional("SMTP_APP_PASSWORD")

# ── Paths / model settings ─────────────────────────────────────────
# Change this line:
# DOCS_FOLDER = _optional("DOCS_FOLDER", "docs")

# To this:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, _optional("DOCS_FOLDER", "docs"))
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

OTP_EXPIRY_SECONDS = 600
OTP_LENGTH         = 8

NO_CONTEXT_MSG = (
    "I'm sorry, I don't have information about that in the available documents."
)

# ─────────────────────────────────────────────────────────────────
#  STARTUP LOGGING  (never log secret values)
# ─────────────────────────────────────────────────────────────────
logging.warning("[CONFIG] GROQ_API_KEY   : OK (len=%d)", len(GROQ_API_KEY))
logging.warning("[CONFIG] SNOWFLAKE_USER : %s", SNOWFLAKE_USER)
logging.warning("[CONFIG] SMTP sender    : %s", SMTP_SENDER_EMAIL or "NOT SET (dev mode)")

# ─────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="/static")
app.secret_key = FLASK_SECRET

# ─────────────────────────────────────────────────────────────────
#  SNOWFLAKE
# ─────────────────────────────────────────────────────────────────
_db_conn = None

def get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            role=SNOWFLAKE_ROLE,
        )
        _ensure_tables(_db_conn)
    return _db_conn

def _sf_exec(sql, params=()):
    conn = get_db(); cur = conn.cursor()
    try:
        cur.execute(sql, params); conn.commit()
    finally:
        cur.close()

def _sf_fetch(sql, params=()):
    cur = get_db().cursor()
    try:
        cur.execute(sql, params); return cur.fetchall()
    finally:
        cur.close()

def _ensure_tables(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS app_users (
        user_id VARCHAR(36) PRIMARY KEY, email VARCHAR(256) NOT NULL UNIQUE,
        full_name VARCHAR(256) DEFAULT '', password_hash VARCHAR(512) NOT NULL,
        is_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS otp_tokens (
        id VARCHAR(36) PRIMARY KEY, email VARCHAR(256) NOT NULL,
        otp_code VARCHAR(32) NOT NULL, purpose VARCHAR(32) NOT NULL,
        expires_at NUMBER NOT NULL, used BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
        id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) NOT NULL,
        user_email VARCHAR(256) NOT NULL, user_name VARCHAR(256) DEFAULT '',
        title VARCHAR(200) NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id VARCHAR(36) PRIMARY KEY, session_id VARCHAR(36) NOT NULL,
        role VARCHAR(20) NOT NULL, content TEXT NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")
    cur.close()

# ── User helpers ──────────────────────────────────────────────────
def db_get_user(email):
    rows = _sf_fetch(
        "SELECT user_id,email,full_name,password_hash,is_verified FROM app_users WHERE email=%s",
        (email,)
    )
    if rows:
        r = rows[0]
        return {
            "user_id": r[0], "email": r[1], "full_name": r[2],
            "password_hash": r[3], "is_verified": r[4],
        }
    return None

def db_create_user(email, full_name, pw_hash):
    uid = str(uuid.uuid4())
    _sf_exec(
        "INSERT INTO app_users (user_id,email,full_name,password_hash,is_verified) VALUES (%s,%s,%s,%s,%s)",
        (uid, email, full_name, pw_hash, False),
    )
    return uid

def db_verify_user(email):
    _sf_exec("UPDATE app_users SET is_verified=TRUE WHERE email=%s", (email,))

# ── OTP helpers ───────────────────────────────────────────────────
def generate_otp():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=OTP_LENGTH))

def db_save_otp(email, otp, purpose):
    _sf_exec(
        "UPDATE otp_tokens SET used=TRUE WHERE email=%s AND purpose=%s AND used=FALSE",
        (email, purpose),
    )
    _sf_exec(
        "INSERT INTO otp_tokens (id,email,otp_code,purpose,expires_at) VALUES (%s,%s,%s,%s,%s)",
        (str(uuid.uuid4()), email, otp, purpose, int(time.time()) + OTP_EXPIRY_SECONDS),
    )

def db_verify_otp(email, otp, purpose):
    rows = _sf_fetch(
        """SELECT id FROM otp_tokens
           WHERE email=%s AND otp_code=%s AND purpose=%s AND used=FALSE AND expires_at>%s
           ORDER BY created_at DESC LIMIT 1""",
        (email, otp.strip().upper(), purpose, int(time.time())),
    )
    if rows:
        _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE id=%s", (rows[0][0],))
        return True
    return False

# ── Chat helpers ──────────────────────────────────────────────────
def db_sessions(user_id):
    rows = _sf_fetch(
        "SELECT id,title,created_at FROM chat_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30",
        (user_id,),
    )
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(user_id, email, name, title):
    sid = str(uuid.uuid4())
    _sf_exec(
        "INSERT INTO chat_sessions (id,user_id,user_email,user_name,title) VALUES (%s,%s,%s,%s,%s)",
        (sid, user_id, email, name, title),
    )
    return sid

def db_messages(sid):
    rows = _sf_fetch(
        "SELECT role,content FROM chat_messages WHERE session_id=%s ORDER BY created_at",
        (sid,),
    )
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save_msg(sid, role, content):
    _sf_exec(
        "INSERT INTO chat_messages (id,session_id,role,content) VALUES (%s,%s,%s,%s)",
        (str(uuid.uuid4()), sid, role, content),
    )

def db_delete_session(sid):
    _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (sid,))
    _sf_exec("DELETE FROM chat_sessions WHERE id=%s", (sid,))

def db_delete_all_sessions(user_id):
    rows = _sf_fetch("SELECT id FROM chat_sessions WHERE user_id=%s", (user_id,))
    for row in rows:
        _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id=%s", (user_id,))

# ─────────────────────────────────────────────────────────────────
#  EMAIL  (OTPs never printed to stdout/stderr)
# ─────────────────────────────────────────────────────────────────
def _smtp_send(sender, app_pass, mime_msg, to_email):
    """Try SSL/465 first, fall back to STARTTLS/587."""
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(sender, app_pass)
            s.sendmail(sender, to_email, mime_msg.as_string())
        return True
    except Exception as e1:
        logging.warning("SMTP port 465 failed (%s), trying 587…", e1)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.ehlo(); s.starttls(); s.ehlo()
            s.login(sender, app_pass)
            s.sendmail(sender, to_email, mime_msg.as_string())
        return True
    except Exception as e2:
        logging.error("SMTP port 587 also failed: %s", e2)
        return False

def _otp_html(otp, headline):
    return f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:32px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:18px;padding:36px;
              border:1px solid #e0e0e0;box-shadow:0 4px 20px rgba(0,0,0,0.07);">
    <h2 style="color:#e85d4a;margin-bottom:4px;">TechWish DocQuery</h2>
    <p style="color:#666;font-size:13px;margin-bottom:24px;">{headline}</p>
    <div style="background:#fff8f7;border:2px solid rgba(232,93,74,0.35);border-radius:14px;
                padding:24px;text-align:center;margin-bottom:20px;">
      <span style="font-size:2.4rem;font-weight:900;letter-spacing:0.25em;
                   color:#e85d4a;font-family:monospace;">{otp}</span>
    </div>
    <p style="color:#aaa;font-size:12px;">Expires in <strong>10 minutes</strong>. Do not share this code.</p>
  </div></body></html>"""

def send_otp_email(to_email, otp, purpose):
    sender   = SMTP_SENDER_EMAIL
    app_pass = SMTP_APP_PASSWORD
    if not sender or not app_pass:
        _otp_logger.debug("OTP for %s (%s): %s", to_email, purpose, otp)
        return True
    action = "verify your new account" if purpose == "register" else "sign in to your account"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TechWish DocQuery — Your Verification Code"
    msg["From"]    = f"TechWish DocQuery <{sender}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(_otp_html(otp, f"Use the code below to {action}:"), "html"))
    ok = _smtp_send(sender, app_pass, msg, to_email)
    if not ok:
        _otp_logger.debug("EMAIL FAILED — OTP for %s (%s): %s", to_email, purpose, otp)
    return ok

def send_reset_otp_email(to_email, otp):
    sender   = SMTP_SENDER_EMAIL
    app_pass = SMTP_APP_PASSWORD
    if not sender or not app_pass:
        _otp_logger.debug("RESET OTP for %s: %s", to_email, otp)
        return True
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TechWish DocQuery — Password Reset Code"
    msg["From"]    = f"TechWish DocQuery <{sender}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(_otp_html(otp, "Use the code below to reset your password:"), "html"))
    ok = _smtp_send(sender, app_pass, msg, to_email)
    if not ok:
        _otp_logger.debug("EMAIL FAILED — RESET OTP for %s: %s", to_email, otp)
    return ok

# ─────────────────────────────────────────────────────────────────
#  AI / DOCUMENT INDEX  (fastembed — pure ONNX, no PyTorch)
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
#  LIGHTWEIGHT DOCUMENT SEARCH (RAM FRIENDLY)
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
#  LIGHTWEIGHT DOCUMENT SEARCH (FIXED)
# ─────────────────────────────────────────────────────────────────
_cached_text = ""

def get_all_doc_text():
    global _cached_text
    if _cached_text:
        return _cached_text
    
    # Path logic for Render
    folder = Path(DOCS_FOLDER)
    if not folder.exists():
        logging.warning(f"[INDEX] Folder not found: {folder.absolute()}")
        return ""

    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        logging.warning(f"[INDEX] No PDFs found in: {folder.absolute()}")
        return ""
    
    extracted_text = ""
    for p in pdfs:
        try:
            logging.warning(f"[INDEX] Reading: {p.name}")
            doc = fitz.open(str(p))
            for page in doc:
                extracted_text += page.get_text()
            doc.close()
        except Exception as e:
            logging.error(f"[INDEX] Error reading {p.name}: {e}")
            
    # Clean up the text: remove excess whitespace
    extracted_text = " ".join(extracted_text.split())
    
    # Limit characters for Groq Context (staying safe under RAM limits)
    _cached_text = extracted_text[:18000] 
    logging.warning(f"[INDEX] Success! Extracted {len(_cached_text)} characters.")
    return _cached_text

def doc_search(query):
    context = get_all_doc_text()
    if not context or len(context.strip()) < 10:
        return "", False
    return context, True
SMALL_TALK_KEYWORDS = [
    "hi","hello","hey","hru","how are you","good morning","good afternoon",
    "good evening","good night","what's up","whats up","sup","howdy","greetings",
    "thanks","thank you","thank u","ty","bye","goodbye","see you","take care",
    "who are you","what are you","what can you do","help me","introduce yourself",
    "tell me about yourself","nice to meet you","ok","okay","cool","great","awesome",
    "lol","haha","good","bad","sad","happy","fine","alright","sure","yes","no","yep","nope",
]

def is_small_talk(text):
    t = text.lower().strip()
    if len(t.split()) <= 6:
        for kw in SMALL_TALK_KEYWORDS:
            if kw in t:
                return True
    return False

SMALL_TALK_SYSTEM = (
    "You are the TechWish DocQuery AI — a friendly, professional assistant for TechWish employees. "
    "Respond warmly, briefly, and professionally. Keep replies 1–3 sentences."
)

SYSTEM_PROMPT = """You are the TechWish DocQuery Knowledge Assistant.
You ONLY answer using the Context provided below.

RULES:
- Answer ONLY from the provided Context. Never use outside knowledge.
- If the Context does not contain a clear answer, respond ONLY with: "I'm sorry, I don't have information about that in the available documents."
- Do NOT say "According to the context". Answer directly and professionally.
- Be concise, clear, and helpful.

Context:
{context}"""

def ask_groq(messages, system):
    history = messages[-10:]
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}] + history,
        temperature=0.0, max_tokens=1024,
    )
    return resp.choices[0].message.content

def ask_groq_smalltalk(prompt):
    resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SMALL_TALK_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7, max_tokens=200,
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────────────────────────────
#  AUTH HELPER
# ─────────────────────────────────────────────────────────────────
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return {
        "user_id":   uid,
        "email":     session.get("user_email"),
        "full_name": session.get("user_name"),
    }

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user():
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# ─────────────────────────────────────────────────────────────────
#  ROUTES — Pages
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index_page():
    return send_from_directory(".", "index.html")

@app.route("/chat")
def chat_page():
    if not current_user():
        return redirect("/")
    return send_from_directory(".", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ─────────────────────────────────────────────────────────────────
#  ROUTES — Auth API
# ─────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    data      = request.json or {}
    email     = (data.get("email") or "").strip().lower()
    password  = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()

    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        return jsonify({"error": f"Only @{ALLOWED_DOMAIN} emails are allowed."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if not full_name:
        return jsonify({"error": "Full name is required."}), 400
    if db_get_user(email):
        return jsonify({"error": "An account with this email already exists."}), 409

    pw_hash = _bcrypt.hashpw(password[:72].encode(), _bcrypt.gensalt()).decode()
    uid     = db_create_user(email, full_name, pw_hash)
    otp     = generate_otp()
    db_save_otp(email, otp, "register")
    send_otp_email(email, otp, "register")

    session["pending_uid"]     = uid
    session["pending_email"]   = email
    session["pending_name"]    = full_name
    session["pending_purpose"] = "register"

    return jsonify({"otp_sent": True, "email": email})


@app.route("/api/login", methods=["POST"])
def api_login():
    data     = request.json or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        return jsonify({"error": f"Only @{ALLOWED_DOMAIN} emails are allowed."}), 400

    user = db_get_user(email)
    if not user:
        return jsonify({"error": "No account found with this email."}), 404
    if not _bcrypt.checkpw(password[:72].encode(), user["password_hash"].encode()):
        return jsonify({"error": "Incorrect password."}), 401

    otp = generate_otp()
    db_save_otp(email, otp, "login")
    send_otp_email(email, otp, "login")

    session["pending_uid"]     = user["user_id"]
    session["pending_email"]   = email
    session["pending_name"]    = user["full_name"]
    session["pending_purpose"] = "login"

    return jsonify({"otp_sent": True, "email": email})


@app.route("/api/verify-otp", methods=["POST"])
def api_verify_otp():
    data  = request.json or {}
    email = (data.get("email") or "").strip().lower()
    otp   = (data.get("otp") or "").strip().upper()

    pending_purpose = session.get("pending_purpose", "login")
    valid = db_verify_otp(email, otp, pending_purpose)
    if not valid:
        return jsonify({"error": "Invalid or expired code. Please try again."}), 400

    if pending_purpose == "register":
        db_verify_user(email)

    session["user_id"]    = session.pop("pending_uid", None)
    session["user_email"] = email
    session["user_name"]  = session.pop("pending_name", email.split("@")[0])
    session.pop("pending_purpose", None)
    session.pop("pending_email", None)

    return jsonify({"success": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/resend-otp", methods=["POST"])
def api_resend_otp():
    email   = session.get("pending_email", "")
    purpose = session.get("pending_purpose", "")
    if not email or not purpose:
        return jsonify({"error": "Session expired. Please log in again."}), 400
    otp = generate_otp()
    db_save_otp(email, otp, purpose)
    ok  = send_otp_email(email, otp, purpose)
    if ok:
        return jsonify({"resent": True, "email": email})
    return jsonify({"error": "Failed to send email. Check otp_dev.log on the server."}), 500


@app.route("/api/me")
@require_auth
def api_me():
    u = current_user()
    return jsonify({"user_id": u["user_id"], "email": u["email"], "full_name": u["full_name"]})


@app.route("/api/forgot-password", methods=["POST"])
def api_forgot_password():
    data  = request.json or {}
    email = (data.get("email") or "").strip().lower()
    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        return jsonify({"error": f"Only @{ALLOWED_DOMAIN} emails are allowed."}), 400
    user = db_get_user(email)
    if user:
        otp = generate_otp()
        db_save_otp(email, otp, "reset")
        send_reset_otp_email(email, otp)
        session["reset_email"] = email
    return jsonify({"otp_sent": True})


@app.route("/api/reset-password", methods=["POST"])
def api_reset_password():
    data     = request.json or {}
    email    = (data.get("email") or "").strip().lower()
    otp      = (data.get("otp") or "").strip().upper()
    password = data.get("password") or ""

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    valid = db_verify_otp(email, otp, "reset")
    if not valid:
        return jsonify({"error": "Invalid or expired reset code."}), 400

    pw_hash = _bcrypt.hashpw(password[:72].encode(), _bcrypt.gensalt()).decode()
    _sf_exec("UPDATE app_users SET password_hash=%s WHERE email=%s", (pw_hash, email))
    session.pop("reset_email", None)
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────
#  ROUTES — Chat API
# ─────────────────────────────────────────────────────────────────
@app.route("/api/sessions", methods=["GET"])
@require_auth
def api_get_sessions():
    u = current_user()
    return jsonify(db_sessions(u["user_id"]))


@app.route("/api/sessions", methods=["DELETE"])
@require_auth
def api_delete_all_sessions():
    u = current_user()
    db_delete_all_sessions(u["user_id"])
    return jsonify({"success": True})


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
@require_auth
def api_get_messages(sid):
    return jsonify(db_messages(sid))


@app.route("/api/sessions/<sid>", methods=["DELETE"])
@require_auth
def api_delete_session(sid):
    db_delete_session(sid)
    return jsonify({"success": True})


@app.route("/api/chat", methods=["POST"])
@require_auth
def api_chat():
    u    = current_user()
    data = request.json or {}
    msg  = (data.get("message") or "").strip()
    sid  = data.get("session_id")

    if not msg:
        return jsonify({"error": "Empty message."}), 400

    if not sid:
        sid = db_new_session(u["user_id"], u["email"], u["full_name"], msg[:50])

    db_save_msg(sid, "user", msg)

    history          = db_messages(sid)
    history_for_groq = [{"role": m["role"], "content": m["content"]} for m in history]

    if is_small_talk(msg):
        answer = ask_groq_smalltalk(msg)
    else:
        context, has_ctx = doc_search(msg)
        if not has_ctx:
            answer = NO_CONTEXT_MSG
        else:
            system = SYSTEM_PROMPT.format(context=context)
            answer = ask_groq(history_for_groq[:-1], system)

    db_save_msg(sid, "assistant", answer)
    return jsonify({"answer": answer, "session_id": sid})

# ─────────────────────────────────────────────────────────────────
#  STARTUP (Move this OUTSIDE the if block)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
