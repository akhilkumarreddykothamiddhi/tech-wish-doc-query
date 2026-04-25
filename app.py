"""
TechWish DocQuery — Flask Backend
Serves index.html / chat.html and exposes REST API consumed by the HTML frontend.
All original Streamlit logic is preserved; only the delivery layer changes.
"""

import os, uuid, random, string, smtplib, time
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, redirect, url_for
)
from groq import Groq
import fitz                        # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import snowflake.connector
import bcrypt as _bcrypt

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION  (mirrors original app.py)
# ─────────────────────────────────────────────────────────────────
def cfg(key, default=""):
    return os.environ.get(key, default)

SNOWFLAKE_ACCOUNT   = cfg("SNOWFLAKE_ACCOUNT",   "TCFIWLF-SJ78956")
SNOWFLAKE_USER      = cfg("SNOWFLAKE_USER",       "AKHILREDDY2")
SNOWFLAKE_PASSWORD  = cfg("SNOWFLAKE_PASSWORD",   "Reddy@1614421a")
SNOWFLAKE_WAREHOUSE = cfg("SNOWFLAKE_WAREHOUSE",  "COMPUTE_WH")
SNOWFLAKE_DATABASE  = cfg("SNOWFLAKE_DATABASE",   "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = cfg("SNOWFLAKE_SCHEMA",     "PUBLIC")
SNOWFLAKE_ROLE      = cfg("SNOWFLAKE_ROLE",       "ACCOUNTADMIN")

GROQ_API_KEY   = cfg("GROQ_API_KEY")
ALLOWED_DOMAIN = cfg("ALLOWED_DOMAIN", "techwish.com")

SMTP_SENDER_EMAIL = cfg("SMTP_SENDER_EMAIL", "")
SMTP_APP_PASSWORD = cfg("SMTP_APP_PASSWORD", "")

DOCS_FOLDER   = cfg("DOCS_FOLDER", "docs")
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 4

OTP_EXPIRY_SECONDS = 600
OTP_LENGTH         = 8

NO_CONTEXT_MSG = "I'm sorry, I don't have information about that in the available documents."

# ─────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="/static")
app.secret_key = cfg("FLASK_SECRET", os.urandom(32))

# ─────────────────────────────────────────────────────────────────
#  SNOWFLAKE
# ─────────────────────────────────────────────────────────────────
_db_conn = None

def get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT, user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD, warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE, schema=SNOWFLAKE_SCHEMA,
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
    rows = _sf_fetch("SELECT user_id,email,full_name,password_hash,is_verified FROM app_users WHERE email=%s", (email,))
    if rows:
        r = rows[0]
        return {"user_id": r[0], "email": r[1], "full_name": r[2], "password_hash": r[3], "is_verified": r[4]}
    return None

def db_create_user(email, full_name, pw_hash):
    uid = str(uuid.uuid4())
    _sf_exec("INSERT INTO app_users (user_id,email,full_name,password_hash,is_verified) VALUES (%s,%s,%s,%s,%s)",
             (uid, email, full_name, pw_hash, False))
    return uid

def db_verify_user(email):
    _sf_exec("UPDATE app_users SET is_verified=TRUE WHERE email=%s", (email,))

# ── OTP helpers ───────────────────────────────────────────────────
def generate_otp():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=OTP_LENGTH))

def db_save_otp(email, otp, purpose):
    _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE email=%s AND purpose=%s AND used=FALSE", (email, purpose))
    _sf_exec("INSERT INTO otp_tokens (id,email,otp_code,purpose,expires_at) VALUES (%s,%s,%s,%s,%s)",
             (str(uuid.uuid4()), email, otp, purpose, int(time.time()) + OTP_EXPIRY_SECONDS))

def db_verify_otp(email, otp, purpose):
    rows = _sf_fetch("""
        SELECT id FROM otp_tokens
        WHERE email=%s AND otp_code=%s AND purpose=%s AND used=FALSE AND expires_at>%s
        ORDER BY created_at DESC LIMIT 1""",
        (email, otp.strip().upper(), purpose, int(time.time())))
    if rows:
        _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE id=%s", (rows[0][0],))
        return True
    return False

# ── Chat helpers ──────────────────────────────────────────────────
def db_sessions(user_id):
    rows = _sf_fetch("SELECT id,title,created_at FROM chat_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 30", (user_id,))
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(user_id, email, name, title):
    sid = str(uuid.uuid4())
    _sf_exec("INSERT INTO chat_sessions (id,user_id,user_email,user_name,title) VALUES (%s,%s,%s,%s,%s)",
             (sid, user_id, email, name, title))
    return sid

def db_messages(sid):
    rows = _sf_fetch("SELECT role,content FROM chat_messages WHERE session_id=%s ORDER BY created_at", (sid,))
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save_msg(sid, role, content):
    _sf_exec("INSERT INTO chat_messages (id,session_id,role,content) VALUES (%s,%s,%s,%s)",
             (str(uuid.uuid4()), sid, role, content))

def db_delete_session(sid):
    _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (sid,))
    _sf_exec("DELETE FROM chat_sessions WHERE id=%s", (sid,))

def db_delete_all_sessions(user_id):
    rows = _sf_fetch("SELECT id FROM chat_sessions WHERE user_id=%s", (user_id,))
    for row in rows:
        _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id=%s", (user_id,))

# ─────────────────────────────────────────────────────────────────
#  EMAIL
# ─────────────────────────────────────────────────────────────────
def send_otp_email(to_email, otp, purpose):
    sender   = SMTP_SENDER_EMAIL
    app_pass = SMTP_APP_PASSWORD
    if not sender or not app_pass:
        print(f"[DEV] OTP for {to_email}: {otp}")
        return True
    action  = "verify your new account" if purpose == "register" else "sign in to your account"
    subject = f"Techwish AI — Your Login Code: {otp}"
    body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#3a3a3a;color:#e8eaf0;padding:32px;">
      <div style="max-width:480px;margin:0 auto;background:#2e2e2e;border-radius:18px;padding:32px;border:1px solid rgba(232,93,74,0.2);">
        <h2 style="color:#e85d4a;margin-bottom:8px;">TechWish DocQuery</h2>
        <p style="color:rgba(232,234,240,0.6);margin-bottom:24px;">Use the code below to {action}:</p>
        <div style="background:rgba(232,93,74,0.1);border:1px solid rgba(232,93,74,0.3);border-radius:12px;padding:20px 28px;text-align:center;margin-bottom:24px;">
          <span style="font-size:2.2rem;font-weight:800;letter-spacing:0.2em;color:#ffffff;font-family:monospace;">{otp}</span>
        </div>
        <p style="color:rgba(232,234,240,0.4);font-size:0.82rem;">Expires in <strong>10 minutes</strong>. Do not share it.</p>
      </div>
    </body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject; msg["From"] = sender; msg["To"] = to_email
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(sender, app_pass); srv.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────
#  AI / DOCUMENT INDEX
# ─────────────────────────────────────────────────────────────────
_embedder   = None
_doc_idx    = None
_doc_chunks = []

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def pdf_text(path):
    doc   = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def chunk_text(text):
    out, i = [], 0
    while i < len(text):
        out.append(text[i: i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return out

def build_index():
    global _doc_idx, _doc_chunks
    em     = get_embedder()
    folder = Path(DOCS_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs   = list(folder.glob("*.pdf"))
    if not pdfs:
        _doc_idx = None; _doc_chunks = []; return

    chunks = []
    for p in pdfs:
        for c in chunk_text(pdf_text(str(p))):
            chunks.append(c)

    embs = em.encode(chunks, batch_size=64, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    _doc_idx = idx; _doc_chunks = chunks
    print(f"[INDEX] Built index with {len(chunks)} chunks from {len(pdfs)} PDF(s)")

def doc_search(query):
    if not _doc_idx:
        return "", False
    q = get_embedder().encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = _doc_idx.search(q, TOP_K)
    relevant = [_doc_chunks[i] for i, s in zip(ids[0], scores[0])
                if i < len(_doc_chunks) and s >= 0.30]
    if not relevant:
        return "", False
    return "\n\n---\n\n".join(relevant), True

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
        messages=[{"role": "system", "content": SMALL_TALK_SYSTEM}, {"role": "user", "content": prompt}],
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
    return {"user_id": uid, "email": session.get("user_email"), "full_name": session.get("user_name")}

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
    if current_user():
        return redirect("/chat")
    return send_from_directory(".", "index.html")

@app.route("/chat")
def chat_page():
    if not current_user():
        return redirect("/")
    return send_from_directory(".", "chat.html")

# Serve logo and other static assets from current directory
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ─────────────────────────────────────────────────────────────────
#  ROUTES — Auth API
# ─────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json or {}
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

    session["pending_uid"]      = uid
    session["pending_email"]    = email
    session["pending_name"]     = full_name
    session["pending_purpose"]  = "register"

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


@app.route("/api/me")
@require_auth
def api_me():
    u = current_user()
    return jsonify({"user_id": u["user_id"], "email": u["email"], "full_name": u["full_name"]})

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

    # Create session if needed
    if not sid:
        sid = db_new_session(u["user_id"], u["email"], u["full_name"], msg[:50])

    db_save_msg(sid, "user", msg)

    history = db_messages(sid)  # includes the message we just saved
    history_for_groq = [{"role": m["role"], "content": m["content"]} for m in history]

    # Generate answer
    if is_small_talk(msg):
        answer = ask_groq_smalltalk(msg)
    else:
        context, has_ctx = doc_search(msg)
        if not has_ctx:
            answer = NO_CONTEXT_MSG
        else:
            system = SYSTEM_PROMPT.format(context=context)
            answer = ask_groq(history_for_groq[:-1], system)  # exclude last user msg (already in system prompt context)

    db_save_msg(sid, "assistant", answer)

    return jsonify({"answer": answer, "session_id": sid})

# ─────────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[STARTUP] Building document index…")
    build_index()
    print("[STARTUP] Starting TechWish DocQuery Flask server…")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
