# 🧠 OrgMind AI — Internal Organization Edition

Runs on your server at `10.10.10.73`.
Employees access via `http://10.10.10.73:8501` in their browser.

---

## What you need on your server (10.10.10.73)

- Python 3.10+ installed
- ODBC Driver 17 for SQL Server installed
  → Download: https://aka.ms/odbc17
- Your MS SQL Server already running (it is — you have the connection string)
- Port 8501 open in your server's firewall (for employees to access the app)

---

## Setup — 5 steps

### Step 1 — Install Python dependencies

Open a command prompt on your server and run:

```cmd
pip install -r requirements.txt
```

---

### Step 2 — Get Groq API key (free LLM)

1. Go to https://console.groq.com
2. Sign up free → Create API Key
3. Copy the key (starts with `gsk_...`)

---

### Step 3 — Set up Google Login (free, via Supabase)

1. Go to https://supabase.com → create free account → New Project
2. Go to Authentication → Providers → Google → toggle ON
3. Go to https://console.cloud.google.com:
   - New project → APIs & Services → Credentials
   - Create OAuth 2.0 Client ID → Web application
   - Authorized redirect URI:
     `https://xxxxxxxxxxxx.supabase.co/auth/v1/callback`
4. Paste Client ID and Secret into Supabase → Save
5. In Supabase → Authentication → URL Configuration:
   - Add to Redirect URLs: `http://10.10.10.73:8501`
6. Copy your Supabase Project URL and anon key

---

### Step 4 — Fill in secrets.toml

Edit `.streamlit\secrets.toml` — it's already pre-filled with your SQL Server details.
Just add:
- Your Supabase URL and anon key
- Your Groq API key
- Your org email domain (e.g. `yourcompany.com`)

---

### Step 5 — Add your PDFs

```
orgmind/
├── app.py
├── docs/
│   ├── hr/       ← drop HR policy PDFs here
│   └── it/       ← drop IT policy PDFs here
```

---

### Step 6 — Run the app

```cmd
cd C:\orgmind
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Employees can now open: **http://10.10.10.73:8501**

---

### Step 7 — Run as a Windows Service (auto-start on reboot)

Download NSSM: https://nssm.cc/download

```cmd
nssm install OrgMindAI "C:\Python311\Scripts\streamlit.exe" "run C:\orgmind\app.py --server.port 8501 --server.address 0.0.0.0"
nssm set OrgMindAI AppDirectory C:\orgmind
nssm start OrgMindAI
```

---

## Security recommendation — create a dedicated SQL user

Instead of using `sa`, create a limited user for the app:

```sql
-- Run this in SSMS
USE master;
CREATE LOGIN orgmind_app WITH PASSWORD = 'NewAppPassword123!';
USE OrgMindAI;
CREATE USER orgmind_app FOR LOGIN orgmind_app;
ALTER ROLE db_datareader ADD MEMBER orgmind_app;
ALTER ROLE db_datawriter ADD MEMBER orgmind_app;
```

Then update `secrets.toml`:
```toml
SQL_USER     = "orgmind_app"
SQL_PASSWORD = "NewAppPassword123!"
```

---

## Updating PDFs

1. Copy new PDFs into `docs/hr/` or `docs/it/`
2. Click **"Reload PDFs"** in the app sidebar
3. Done — no restart needed

---

## Troubleshooting

**"ODBC driver not found"**
→ Install ODBC Driver 17: https://aka.ms/odbc17

**"Login failed for user sa"**
→ Enable SQL Server Authentication in SSMS:
  Right-click server → Properties → Security → SQL Server and Windows Authentication

**"Cannot connect to DB"**
→ Make sure SQL Server service is running: `services.msc` → SQL Server (MSSQLSERVER)

**"Employees can't open the app"**
→ Open port 8501 in Windows Firewall:
  `netsh advfirewall firewall add rule name="OrgMind" dir=in action=allow protocol=TCP localport=8501`
