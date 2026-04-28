"""
TechWish DocQuery — Flask Backend (MERGED PRODUCTION)

Key behaviour:
- Documents are indexed ONCE into Snowflake (DOCUMENT_INDEX table).
- Every subsequent user login / app restart skips re-indexing — it checks
  the DB first.  A new PDF dropped into /docs is picked up automatically
  on the next restart, but existing files are never re-processed.
- Embeddings live in Snowflake; fastembed (ONNX) is used locally only to
  embed the *query* at search time — very fast, no PyTorch needed.
- OTPs are never printed to stdout; dev mode writes them to otp_dev.log.
- All secrets come from .env / environment variables — nothing hardcoded.

ACCURACY FIXES (v2):
  1. Embeddings are L2-normalised before storage AND at query time,
     so Snowflake dot-product == true cosine similarity.
  2. Similarity threshold raised to 0.50 (was 0.20).
  3. Risky "use best match anyway" fallback removed.
  4. CHUNK_SIZE raised to 1200 (was 500), CHUNK_OVERLAP to 200 (was 80).
     Re-index required: TRUNCATE TABLE DOCUMENT_INDEX; then restart.
"""

import os, uuid, random, string, smtplib, time, logging
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, redirect,
)
from groq import Groq
import fitz                     # PyMuPDF
# from fastembed import TextEmbedding
# import numpy as np
import snowflake.connector
import bcrypt as _bcrypt

# ─────────────────────────────────────────────────────────────────
#  .env  (optional dependency — harmless if missing)
# ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

_otp_logger = logging.getLogger("otp_dev")
_otp_logger.setLevel(logging.DEBUG)
_otp_handler = logging.FileHandler("otp_dev.log")
_otp_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_otp_logger.addHandler(_otp_handler)
_otp_logger.propagate = False

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
def _require(key: str) -> str:
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

# Snowflake
SNOWFLAKE_ACCOUNT   = _require("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = _require("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = _require("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = _optional("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE  = _optional("SNOWFLAKE_DATABASE",  "ORGMIND_AI")
SNOWFLAKE_SCHEMA    = _optional("SNOWFLAKE_SCHEMA",     "PUBLIC")
SNOWFLAKE_ROLE      = _optional("SNOWFLAKE_ROLE",       "ACCOUNTADMIN")

# Groq
GROQ_API_KEY = _require("GROQ_API_KEY")

# App
FLASK_SECRET   = _require("FLASK_SECRET")
ALLOWED_DOMAIN = _optional("ALLOWED_DOMAIN", "techwish.com")

# SMTP (optional — dev writes OTPs to otp_dev.log)
SMTP_SENDER_EMAIL = _optional("SMTP_SENDER_EMAIL")
SMTP_APP_PASSWORD = _optional("SMTP_APP_PASSWORD")

# Paths / model
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER   = os.path.join(BASE_DIR, _optional("DOCS_FOLDER", "docs"))
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"

# ── FIX #3: Larger chunks retain full policy context ─────────────
CHUNK_SIZE    = 1200   # was 500  — bigger chunks = richer context per retrieval
CHUNK_OVERLAP = 200    # was 80   — more overlap prevents sentences being split

TOP_K         = 6

OTP_EXPIRY_SECONDS = 600
OTP_LENGTH         = 8

# ── FIX #2: Raised threshold — only genuinely relevant chunks pass ─
SIMILARITY_THRESHOLD = 0.50   # was 0.20; works correctly after L2 normalisation

NO_CONTEXT_MSG = (
    "I'm sorry, I don't have information about that in the available documents."
)

# ─────────────────────────────────────────────────────────────────
#  STARTUP LOGGING
# ─────────────────────────────────────────────────────────────────
logging.warning("[CONFIG] GROQ_API_KEY   : OK (len=%d)", len(GROQ_API_KEY))
logging.warning("[CONFIG] SNOWFLAKE_USER : %s", SNOWFLAKE_USER)
logging.warning("[CONFIG] SMTP sender    : %s", SMTP_SENDER_EMAIL or "NOT SET (dev mode)")
logging.warning("[CONFIG] DOCS_FOLDER    : %s", DOCS_FOLDER)
logging.warning("[CONFIG] CHUNK_SIZE     : %d", CHUNK_SIZE)
logging.warning("[CONFIG] THRESHOLD      : %.2f", SIMILARITY_THRESHOLD)

# ─────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="/static")
app.secret_key = FLASK_SECRET

# ─────────────────────────────────────────────────────────────────
#  SNOWFLAKE  — single persistent connection with auto-reconnect
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
    """Execute a write statement, reconnecting once on stale-session errors."""
    global _db_conn
    def _run(conn):
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            conn.commit()
        finally:
            cur.close()
    try:
        _run(get_db())
    except Exception as e:
        err = str(e)
        if any(x in err for x in ("390114", "Authentication token", "session")):
            _db_conn = None
            _run(get_db())
        else:
            raise

def _sf_fetch(sql, params=()):
    """Execute a read statement, reconnecting once on stale-session errors."""
    global _db_conn
    def _run(conn):
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            cur.close()
    try:
        return _run(get_db())
    except Exception as e:
        err = str(e)
        if any(x in err for x in ("390114", "Authentication token", "session")):
            _db_conn = None
            return _run(get_db())
        raise

def _ensure_tables(conn):
    cur = conn.cursor()

    # Auth tables
    cur.execute("""CREATE TABLE IF NOT EXISTS app_users (
        user_id VARCHAR(36) PRIMARY KEY,
        email VARCHAR(256) NOT NULL UNIQUE,
        full_name VARCHAR(256) DEFAULT '',
        password_hash VARCHAR(512) NOT NULL,
        is_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")

    cur.execute("""CREATE TABLE IF NOT EXISTS otp_tokens (
        id VARCHAR(36) PRIMARY KEY,
        email VARCHAR(256) NOT NULL,
        otp_code VARCHAR(32) NOT NULL,
        purpose VARCHAR(32) NOT NULL,
        expires_at NUMBER NOT NULL,
        used BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")

    # Chat tables
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
        id VARCHAR(36) PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        user_email VARCHAR(256) NOT NULL,
        user_name VARCHAR(256) DEFAULT '',
        title VARCHAR(200) NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")

    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id VARCHAR(36) PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")

    # ── Document index table ──────────────────────────────────────
    # EMBEDDING stored as a JSON array (VARIANT).
    # Vectors are L2-normalised before storage so dot product == cosine similarity.
    cur.execute("""CREATE TABLE IF NOT EXISTS DOCUMENT_INDEX (
        ID VARCHAR(36) PRIMARY KEY,
        FILENAME VARCHAR(512) NOT NULL,
        CHUNK_TEXT TEXT NOT NULL,
        EMBEDDING VARIANT NOT NULL,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())""")

    cur.close()
    conn.commit()
    logging.warning("[DB] Tables verified / created.")

# ─────────────────────────────────────────────────────────────────
#  USER HELPERS
# ─────────────────────────────────────────────────────────────────
def db_get_user(email):
    rows = _sf_fetch(
        "SELECT user_id,email,full_name,password_hash,is_verified "
        "FROM app_users WHERE email=%s",
        (email,),
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
        "INSERT INTO app_users "
        "(user_id,email,full_name,password_hash,is_verified) "
        "VALUES (%s,%s,%s,%s,%s)",
        (uid, email, full_name, pw_hash, False),
    )
    return uid

def db_verify_user(email):
    _sf_exec("UPDATE app_users SET is_verified=TRUE WHERE email=%s", (email,))

# ─────────────────────────────────────────────────────────────────
#  OTP HELPERS
# ─────────────────────────────────────────────────────────────────
def generate_otp():
    return "".join(
        random.choices(string.ascii_uppercase + string.digits, k=OTP_LENGTH)
    )

def db_save_otp(email, otp, purpose):
    _sf_exec(
        "UPDATE otp_tokens SET used=TRUE "
        "WHERE email=%s AND purpose=%s AND used=FALSE",
        (email, purpose),
    )
    _sf_exec(
        "INSERT INTO otp_tokens "
        "(id,email,otp_code,purpose,expires_at) VALUES (%s,%s,%s,%s,%s)",
        (str(uuid.uuid4()), email, otp, purpose,
         int(time.time()) + OTP_EXPIRY_SECONDS),
    )

def db_verify_otp(email, otp, purpose):
    rows = _sf_fetch(
        """SELECT id FROM otp_tokens
           WHERE email=%s AND otp_code=%s AND purpose=%s
             AND used=FALSE AND expires_at>%s
           ORDER BY created_at DESC LIMIT 1""",
        (email, otp.strip().upper(), purpose, int(time.time())),
    )
    if rows:
        _sf_exec("UPDATE otp_tokens SET used=TRUE WHERE id=%s", (rows[0][0],))
        return True
    return False

# ─────────────────────────────────────────────────────────────────
#  CHAT HELPERS
# ─────────────────────────────────────────────────────────────────
def db_sessions(user_id):
    rows = _sf_fetch(
        "SELECT id,title,created_at FROM chat_sessions "
        "WHERE user_id=%s ORDER BY created_at DESC LIMIT 30",
        (user_id,),
    )
    return [{"id": r[0], "title": r[1], "date": str(r[2])[:10]} for r in rows]

def db_new_session(user_id, email, name, title):
    sid = str(uuid.uuid4())
    _sf_exec(
        "INSERT INTO chat_sessions "
        "(id,user_id,user_email,user_name,title) VALUES (%s,%s,%s,%s,%s)",
        (sid, user_id, email, name, title),
    )
    return sid

def db_messages(sid):
    rows = _sf_fetch(
        "SELECT role,content FROM chat_messages "
        "WHERE session_id=%s ORDER BY created_at",
        (sid,),
    )
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_save_msg(sid, role, content):
    _sf_exec(
        "INSERT INTO chat_messages (id,session_id,role,content) "
        "VALUES (%s,%s,%s,%s)",
        (str(uuid.uuid4()), sid, role, content),
    )

def db_delete_session(sid):
    _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (sid,))
    _sf_exec("DELETE FROM chat_sessions WHERE id=%s", (sid,))

def db_delete_all_sessions(user_id):
    rows = _sf_fetch(
        "SELECT id FROM chat_sessions WHERE user_id=%s", (user_id,)
    )
    for row in rows:
        _sf_exec("DELETE FROM chat_messages WHERE session_id=%s", (row[0],))
    _sf_exec("DELETE FROM chat_sessions WHERE user_id=%s", (user_id,))

# ─────────────────────────────────────────────────────────────────
#  EMAIL
# ─────────────────────────────────────────────────────────────────
def _smtp_send(sender, app_pass, mime_msg, to_email):
    """Try SSL/465 first, then STARTTLS/587."""
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
    <p style="color:#aaa;font-size:12px;">
      Expires in <strong>10 minutes</strong>. Do not share this code.
    </p>
  </div></body></html>"""

def send_otp_email(to_email, otp, purpose):
    sender   = SMTP_SENDER_EMAIL
    app_pass = SMTP_APP_PASSWORD
    if not sender or not app_pass:
        _otp_logger.debug("OTP for %s (%s): %s", to_email, purpose, otp)
        return True
    action = (
        "verify your new account"
        if purpose == "register"
        else "sign in to your account"
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TechWish DocQuery — Your Verification Code"
    msg["From"]    = f"TechWish DocQuery <{sender}>"
    msg["To"]      = to_email
    msg.attach(
        MIMEText(_otp_html(otp, f"Use the code below to {action}:"), "html")
    )
    ok = _smtp_send(sender, app_pass, msg, to_email)
    if not ok:
        _otp_logger.debug(
            "EMAIL FAILED — OTP for %s (%s): %s", to_email, purpose, otp
        )
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
    msg.attach(
        MIMEText(
            _otp_html(otp, "Use the code below to reset your password:"),
            "html",
        )
    )
    ok = _smtp_send(sender, app_pass, msg, to_email)
    if not ok:
        _otp_logger.debug(
            "EMAIL FAILED — RESET OTP for %s: %s", to_email, otp
        )
    return ok

# ─────────────────────────────────────────────────────────────────
#  EMBEDDER  (fastembed — ONNX, no PyTorch)
# ─────────────────────────────────────────────────────────────────
_embedder = None

def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        logging.warning("[EMBED] Loading fastembed model: %s", EMBED_MODEL)
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
        logging.warning("[EMBED] Model loaded.")
    return _embedder


# ── FIX #1: L2 normalisation helper ──────────────────────────────
def _normalise(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector (L2 norm = 1). Safe against zero vectors."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ─────────────────────────────────────────────────────────────────
#  PDF HELPERS
# ─────────────────────────────────────────────────────────────────
def pdf_text(path: str) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)

def chunk_text(text: str) -> list:
    """
    Split text into overlapping chunks.
    FIX #3: CHUNK_SIZE=1200, CHUNK_OVERLAP=200 for richer context.
    """
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i: i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]

# ─────────────────────────────────────────────────────────────────
#  DOCUMENT INDEXING
#
#  build_index() runs ONCE at app startup.
#  Per-file logic:
#    1. Check DOCUMENT_INDEX for rows with FILENAME = <pdf_name>.
#    2. If rows exist  → skip (already indexed).
#    3. If no rows     → extract, chunk, embed (normalised), insert.
#
#  NOTE: After changing CHUNK_SIZE you must truncate DOCUMENT_INDEX
#  in Snowflake before restarting:
#      TRUNCATE TABLE DOCUMENT_INDEX;
# ─────────────────────────────────────────────────────────────────
def build_index():
    folder = Path(DOCS_FOLDER)
    if not folder.exists():
        logging.warning("[INDEX] Docs folder not found: %s", DOCS_FOLDER)
        return

    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        logging.warning("[INDEX] No PDFs found in %s", DOCS_FOLDER)
        return

    em = get_embedder()

    for pdf_path in pdfs:
        filename = pdf_path.name

        # ── STEP 1: Check if already indexed ────────────────────
        try:
            rows = _sf_fetch(
                "SELECT COUNT(*) FROM DOCUMENT_INDEX WHERE FILENAME = %s",
                (filename,),
            )
            already_indexed = (rows[0][0] > 0) if rows else False
        except Exception as e:
            logging.error("[INDEX] DB check failed for '%s': %s", filename, e)
            already_indexed = False

        if already_indexed:
            logging.warning("[INDEX] SKIP — '%s' already indexed.", filename)
            continue

        # ── STEP 2: Index new file ───────────────────────────────
        logging.warning("[INDEX] Indexing '%s' …", filename)
        try:
            text   = pdf_text(str(pdf_path))
            chunks = chunk_text(text)
            if not chunks:
                logging.warning(
                    "[INDEX] No extractable text in '%s'. Skipping.", filename
                )
                continue

            raw_embeddings = list(em.embed(chunks))

            for chunk, raw_emb in zip(chunks, raw_embeddings):
                # ── FIX #1: Normalise before storage ────────────
                emb      = _normalise(np.array(raw_emb))
                vec_json = str(emb.tolist())

                _sf_exec(
                    "INSERT INTO DOCUMENT_INDEX "
                    "(ID, FILENAME, CHUNK_TEXT, EMBEDDING) "
                    "SELECT %s, %s, %s, PARSE_JSON(%s)",
                    (str(uuid.uuid4()), filename, chunk, vec_json),
                )

            logging.warning(
                "[INDEX] Done — '%s' (%d chunks inserted).",
                filename, len(chunks),
            )

        except Exception as e:
            logging.error("[INDEX] Failed to index '%s': %s", filename, e)

    logging.warning("[INDEX] build_index() complete.")

# ─────────────────────────────────────────────────────────────────
#  DOCUMENT SEARCH
#
#  FIX #1: Query vector is also L2-normalised, so the Snowflake
#          dot product equals true cosine similarity (both sides unit).
#  FIX #2: Threshold raised to SIMILARITY_THRESHOLD (0.50).
#          Fallback "use best match anyway" removed — bad context is
#          worse than no context; the LLM handles the no-context case
#          cleanly via NO_CONTEXT_MSG.
# ─────────────────────────────────────────────────────────────────
def doc_search(query: str) -> tuple:
    """Return (context_string, has_results: bool, error: str|None)."""
    if not query.strip():
        return "", False, None

    # ── Embed & normalise the query ──────────────────────────────
    try:
        em      = get_embedder()
        raw_emb = list(em.embed([query]))[0]
        q_emb   = _normalise(np.array(raw_emb))   # FIX #1
        q_vec_json = str(q_emb.tolist())
    except Exception as e:
        logging.error("[SEARCH] Embedding failed: %s", e)
        return "", False, f"Embedding error: {e}"

    # ── Check index is populated ─────────────────────────────────
    try:
        count_rows   = _sf_fetch("SELECT COUNT(*) FROM DOCUMENT_INDEX", ())
        total_chunks = count_rows[0][0] if count_rows else 0
        if total_chunks == 0:
            logging.warning("[SEARCH] DOCUMENT_INDEX is empty.")
            return "", False, "no_index"
    except Exception as e:
        logging.error("[SEARCH] Index count check failed: %s", e)
        return "", False, f"DB error: {e}"

    # ── Vector similarity search (dot product == cosine after normalisation) ─
    sql = """
        WITH query_vec AS (
            SELECT index, value::FLOAT AS val
            FROM TABLE(FLATTEN(input => PARSE_JSON(%s)))
        )
        SELECT
            d.CHUNK_TEXT,
            SUM(d_v.value::FLOAT * q.val) AS score
        FROM DOCUMENT_INDEX d,
             LATERAL FLATTEN(input => d.EMBEDDING) d_v
        JOIN query_vec q ON d_v.index = q.index
        GROUP BY d.ID, d.CHUNK_TEXT
        ORDER BY score DESC
        LIMIT %s
    """

    try:
        results = _sf_fetch(sql, (q_vec_json, TOP_K))
        logging.warning(
            "[SEARCH] Top-%d scores: %s",
            TOP_K,
            [round(r[1], 4) for r in results] if results else "no results",
        )

        # ── FIX #2: Proper threshold, no risky fallback ──────────
        relevant = [r[0] for r in results if r[1] >= SIMILARITY_THRESHOLD]

        if not relevant:
            logging.warning(
                "[SEARCH] No chunks above threshold %.2f — returning no context.",
                SIMILARITY_THRESHOLD,
            )
            return "", False, None

        return "\n\n---\n\n".join(relevant), True, None

    except Exception as e:
        logging.error("[SEARCH] Vector search SQL error: %s", e)
        return "", False, f"Search SQL error: {e}"

# ─────────────────────────────────────────────────────────────────
#  SMALL-TALK DETECTION
# ─────────────────────────────────────────────────────────────────
_SMALL_TALK_KW = {
    "hi","hello","hey","hru","how are you","good morning","good afternoon",
    "good evening","good night","what's up","whats up","sup","howdy","greetings",
    "thanks","thank you","thank u","ty","bye","goodbye","see you","take care",
    "who are you","what are you","what can you do","help me","introduce yourself",
    "tell me about yourself","nice to meet you","ok","okay","cool","great",
    "awesome","lol","haha","good","bad","sad","happy","fine","alright","sure",
    "yes","no","yep","nope",
}

def is_small_talk(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) <= 6:
        for kw in _SMALL_TALK_KW:
            if kw in t:
                return True
    return False

# ─────────────────────────────────────────────────────────────────
#  GROQ CALLS
# ─────────────────────────────────────────────────────────────────
_SMALL_TALK_SYSTEM = (
    "You are the TechWish DocQuery AI — a friendly, professional assistant "
    "for TechWish employees. Respond warmly, briefly, and professionally. "
    "Keep replies 1–3 sentences."
)

_DOC_SYSTEM = """You are TechWish DocQuery, an intelligent knowledge assistant for TechWish employees.
Your job is to read the provided document context carefully and give accurate, helpful answers.

STRICT RULES:
1. Answer ONLY using information found in the Context section below.
2. Never use your own training knowledge to fill in gaps — only use the Context.
3. If the answer is clearly present in the Context, answer it directly, confidently, and completely.
4. If the answer is partially present, share what is available and note what is missing.
5. If the Context does not contain relevant information at all, say exactly:
   "I'm sorry, I don't have information about that in the available documents."
6. Do NOT say phrases like "According to the context", "The document says", or "Based on the provided context".
   Just answer naturally and professionally, as if you already know it.
7. Format your answer clearly — use bullet points or numbered lists when listing multiple items (e.g. leave types, policies, rules).
8. Keep answers concise but complete. Do not truncate important details.
9. If the question is about a policy, procedure, or rule — always include the specific details (days, conditions, approvals, etc.) if present in the Context.

Context:
{context}"""

def _trim_history(messages: list, max_chars: int = 6000) -> list:
    """
    Keep the most recent messages that fit within max_chars total.
    Always keeps the last user message. Trims assistant answers to
    300 chars max so long policy responses don't explode future tokens.
    """
    trimmed = []
    for m in messages:
        role    = m["role"]
        content = m["content"]
        if role == "assistant" and len(content) > 300:
            content = content[:300] + "…"
        trimmed.append({"role": role, "content": content})

    kept  = []
    total = 0
    for m in reversed(trimmed):
        total += len(m["content"])
        if total > max_chars:
            break
        kept.insert(0, m)

    if not kept:
        for m in reversed(trimmed):
            if m["role"] == "user":
                kept = [m]
                break
    return kept


def _is_groq_rate_limit(err: str) -> bool:
    keywords = ("429", "rate limit", "ratelimit", "too many requests",
                 "tokens per minute", "requests per minute", "quota")
    return any(k in err.lower() for k in keywords)


def _is_groq_context_too_long(err: str) -> bool:
    keywords = ("context_length", "context length", "maximum context",
                 "too long", "reduce the length", "tokens in your prompt")
    return any(k in err.lower() for k in keywords)


def ask_groq(messages: list, system: str) -> str:
    """
    Call Groq with history trimming, rate-limit retry, and context-length fallback.
    """
    max_retries = 3
    history     = _trim_history(messages)

    for attempt in range(max_retries):
        try:
            payload = [{"role": "system", "content": system}] + history
            logging.warning(
                "[GROQ] Sending %d messages, ~%d chars (attempt %d)",
                len(payload),
                sum(len(m["content"]) for m in payload),
                attempt + 1,
            )
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL,
                messages=payload,
                temperature=0.1,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e)
            logging.error("[GROQ] ask_groq error (attempt %d): %s", attempt + 1, err_str)

            if _is_groq_rate_limit(err_str) and attempt < max_retries - 1:
                wait = (attempt + 1) * 20
                logging.warning("[GROQ] Rate limited — sleeping %ds…", wait)
                time.sleep(wait)
                continue

            if _is_groq_context_too_long(err_str) and attempt < max_retries - 1:
                logging.warning("[GROQ] Context too long — trimming history further.")
                history = _trim_history(messages, max_chars=2000)
                continue

            raise


def ask_groq_smalltalk(prompt: str) -> str:
    """Call Groq for small-talk with retry on rate-limit."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SMALL_TALK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            logging.error("[GROQ] ask_groq_smalltalk error (attempt %d): %s", attempt + 1, err_str)
            if _is_groq_rate_limit(err_str) and attempt < max_retries - 1:
                wait = (attempt + 1) * 20
                logging.warning("[GROQ] Rate limited — sleeping %ds…", wait)
                time.sleep(wait)
                continue
            raise

# ─────────────────────────────────────────────────────────────────
#  AUTH HELPERS
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
    if not db_verify_otp(email, otp, pending_purpose):
        return jsonify({"error": "Invalid or expired code. Please try again."}), 400

    if pending_purpose == "register":
        db_verify_user(email)

    session["user_id"]    = session.pop("pending_uid",  None)
    session["user_email"] = email
    session["user_name"]  = session.pop("pending_name", email.split("@")[0])
    session.pop("pending_purpose", None)
    session.pop("pending_email",   None)

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
    return jsonify({
        "error": "Failed to send email. Check otp_dev.log on the server."
    }), 500


@app.route("/api/me")
@require_auth
def api_me():
    u = current_user()
    return jsonify({
        "user_id":   u["user_id"],
        "email":     u["email"],
        "full_name": u["full_name"],
    })


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
    if not db_verify_otp(email, otp, "reset"):
        return jsonify({"error": "Invalid or expired reset code."}), 400

    pw_hash = _bcrypt.hashpw(password[:72].encode(), _bcrypt.gensalt()).decode()
    _sf_exec(
        "UPDATE app_users SET password_hash=%s WHERE email=%s",
        (pw_hash, email),
    )
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
def api_delete_all_sessions_route():
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
        sid = db_new_session(
            u["user_id"], u["email"], u["full_name"], msg[:50]
        )

    db_save_msg(sid, "user", msg)

    history          = db_messages(sid)
    history_for_groq = [
        {"role": m["role"], "content": m["content"]}
        for m in history[:-1]
    ]

    answer = None

    try:
        if is_small_talk(msg):
            answer = ask_groq_smalltalk(msg)
        else:
            context, has_ctx, search_err = doc_search(msg)

            if search_err == "no_index":
                answer = (
                    "The document index is still being built or no documents have "
                    "been uploaded yet. Please try again in a moment."
                )
            elif search_err:
                logging.error(
                    "[CHAT] doc_search error — user=%s msg='%s' err=%s",
                    u["email"], msg[:60], search_err,
                )
                answer = (
                    "I ran into a technical issue while searching the documents. "
                    "Please try again in a moment."
                )
            elif not has_ctx:
                answer = NO_CONTEXT_MSG
            else:
                full_history = history_for_groq + [{"role": "user", "content": msg}]
                system       = _DOC_SYSTEM.format(context=context)
                answer       = ask_groq(full_history, system)

    except Exception as e:
        err_str = str(e)
        logging.error(
            "[CHAT] Unhandled error — user=%s msg='%s' err=%s",
            u["email"], msg[:60], err_str,
        )
        if _is_groq_rate_limit(err_str):
            answer = (
                "The AI service is currently rate-limited due to high traffic. "
                "Please wait 30 seconds and try again."
            )
        elif _is_groq_context_too_long(err_str):
            answer = (
                "Your conversation history is very long. "
                "Please start a new chat session and try your question again."
            )
        else:
            answer = (
                "Something went wrong on our end. Please try again in a moment."
            )

    db_save_msg(sid, "assistant", answer)
    return jsonify({"answer": answer, "session_id": sid})

# ─────────────────────────────────────────────────────────────────
#  MODULE-LEVEL STARTUP
# ─────────────────────────────────────────────────────────────────
logging.warning("[STARTUP] Initialising DB connection and tables…")
get_db()

logging.warning("[STARTUP] Running build_index() — already-indexed files are skipped…")
build_index()

logging.warning("[STARTUP] TechWish DocQuery is ready.")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
