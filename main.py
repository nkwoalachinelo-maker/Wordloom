"""
DeAilize backend — FastAPI + SQLite, using Groq's OpenAI-compatible API.
Training data (every rewrite's input/output) is logged to a Google Sheet,
not SQLite — see the Google Sheets setup below.

Setup:
    pip install fastapi uvicorn openai pydantic[email] PyJWT bcrypt gspread google-auth

Required environment variables:
    GROQ_API_KEY            Your Groq API key (console.groq.com)
    JWT_SECRET              Long random string used to sign session tokens
    GOOGLE_SHEET_ID         The ID from your Google Sheet's URL (the long
                            string between /d/ and /edit)
Optional:
    GROQ_MODEL              Defaults to "openai/gpt-oss-20b" (as before)
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
                            Optional: point at a different OpenAI-compatible
                            provider (e.g. Gemini) without touching the code
    DB_PATH                 Defaults to "wordloom.db"
    CORS_ORIGINS            Comma-separated allowed origins, defaults to "*"
    GOOGLE_SERVICE_ACCOUNT_FILE
                            Path to the service account JSON key. Defaults to
                            "/etc/secrets/google-service-account.json" — the
                            path Render gives Secret Files. Upload the key
                            there instead of pasting it into a plain env var,
                            since it's multi-line JSON and easy to corrupt by
                            hand-pasting on mobile.

Google Sheets setup (one-time):
    1. console.cloud.google.com → create a project (or use an existing one).
    2. APIs & Services → Library → enable "Google Sheets API".
    3. APIs & Services → Credentials → Create Credentials → Service Account.
    4. Open the new service account → Keys tab → Add Key → JSON. This
       downloads a .json file — that's GOOGLE_SERVICE_ACCOUNT_FILE.
    5. Open the JSON file, find "client_email" — copy that address.
    6. Create a new Google Sheet, click Share, paste that client_email in,
       give it Editor access.
    7. Copy the Sheet's ID from its URL and set GOOGLE_SHEET_ID.
    8. On Render: Settings → Secret Files → add a file named
       google-service-account.json with the full JSON key's contents, then
       add GOOGLE_SHEET_ID as a normal environment variable.

If Sheets isn't configured yet, rewrites still work fine — training data
logging is skipped with a warning in the logs, nothing breaks.

Note: a stiffness gate (code only, no model) flags the sentences that read stiff;
only those go to the model, section by section (word pass + rewrite, plus a
retry or a final restructure pass if needed). Sentences that read naturally are
never touched. Rate-limit errors (HTTP 429) are retried automatically.

Run:
    uvicorn main:app --reload
"""

import asyncio
import hashlib
import json
import logging
import os
import queue
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

import bcrypt
import gspread
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.oauth2.service_account import Credentials
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, EmailStr, Field

from rewrite_engine import load_tells, load_wordset, rewrite_post, set_tells

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wordloom")

# ========== CONFIG ==========
# Groq exposes an OpenAI-compatible API, so we use the plain OpenAI client
# pointed at Groq's base URL instead of AzureOpenAI.
# Defaults to Groq. To try another OpenAI-compatible provider later, set
# LLM_BASE_URL + LLM_API_KEY + LLM_MODEL on Render; nothing else changes.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = (os.environ["LLM_API_KEY"] if "LLM_BASE_URL" in os.environ
                else os.environ["GROQ_API_KEY"])
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))   # lower it (e.g. 1024) for a small self-hosted model
WORD_PASS = os.environ.get("LLM_WORD_PASS", "on").lower() != "off"   # "off" skips the slow word-synonym step
MODEL = os.environ.get("LLM_MODEL") or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 24
DB_PATH = os.environ.get("DB_PATH", "wordloom.db")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_FILE", "/etc/secrets/google-service-account.json"
)

# Daily rewrite allowance per plan. Kept low by default since Groq's free
# tier has its own rate limits on top of this app's own limit.
PLAN_LIMITS = {"free": 5, "pro": 200}

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=LLM_BASE_URL,
    max_retries=0,   # _chat() retries itself so it can honour Retry-After
)


# ========== WORD LIST ==========

# words_alpha.txt (370k English words, one per line) is the dictionary of REAL
# words. The model proposes and ranks 30 candidates per word; only candidates
# that are real words survive, and garbled/invented words in a rewrite are
# rejected. Commit it next to main.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
WORDSET = load_wordset(os.path.join(_HERE, "words_alpha.txt"))
# Stock words the stiffness gate flags (edit ai_tells.txt, no code needed)
set_tells(load_tells(os.path.join(_HERE, "ai_tells.txt")))
if WORDSET:
    logger.info("Word guard on: %d words from words_alpha.txt", len(WORDSET))
else:
    logger.warning("words_alpha.txt missing or too small: word guard is OFF")




# Each entry is a list ranked from MOST AI-used (index 0) to
# LEAST AI-used (last index). The weighted picker below makes
# choices near the END of each list far more likely — so the
# engine tends toward vocabulary AI would almost never choose.


# Sort longest first so multi-word phrases match before their component words








def apply_contractions(text: str) -> str:
    """Apply common contractions — humans use these naturally, AI doesn't."""
    pairs = [
        (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"),
        (r"\bdid not\b", "didn't"), (r"\bwill not\b", "won't"),
        (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"),
        (r"\bwould not\b", "wouldn't"), (r"\bshould not\b", "shouldn't"),
        (r"\bcould not\b", "couldn't"), (r"\bI am\b", "I'm"),
        (r"\bI have\b", "I've"), (r"\bI will\b", "I'll"),
        (r"\bI would\b", "I'd"), (r"\bthey are\b", "they're"),
        (r"\bwe are\b", "we're"), (r"\byou are\b", "you're"),
        (r"\bhe is\b", "he's"), (r"\bshe is\b", "she's"),
        (r"\bit is\b", "it's"), (r"\bthat is\b", "that's"),
        (r"\bthere is\b", "there's"), (r"\bwhat is\b", "what's"),
        (r"\bwho is\b", "who's"), (r"\bare not\b", "aren't"),
        (r"\bis not\b", "isn't"), (r"\bwas not\b", "wasn't"),
        (r"\bwere not\b", "weren't"),
    ]
    for pattern, replacement in pairs:
        if random.random() < 0.75:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text




_training_worksheet = None
_training_worksheet_tried = False


def get_training_worksheet():
    """Lazily connects to the training-data Google Sheet. Returns None (and
    logs why) if it isn't configured or the connection fails — callers must
    treat that as "skip logging", not as an error worth failing the request over."""
    global _training_worksheet, _training_worksheet_tried
    if _training_worksheet is not None:
        return _training_worksheet
    if _training_worksheet_tried:
        return None
    _training_worksheet_tried = True

    if not GOOGLE_SHEET_ID:
        logger.warning("GOOGLE_SHEET_ID not set — training data will not be logged.")
        return None
    if not os.path.exists(GOOGLE_SERVICE_ACCOUNT_FILE):
        logger.warning(
            "Google service account file not found at %s — training data will not be logged.",
            GOOGLE_SERVICE_ACCOUNT_FILE,
        )
        return None

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = spreadsheet.worksheet("TrainingData")
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title="TrainingData", rows=1000, cols=8)
            ws.append_row(
                ["timestamp", "user_id", "input_text", "voice_samples", "location", "style", "output_text"]
            )
        _training_worksheet = ws
        return ws
    except Exception as exc:
        logger.error("Could not connect to Google Sheets: %s", exc)
        return None

def hash_password(password: str) -> str:
    # SHA-256 pre-hash avoids bcrypt's 72-byte input limit entirely, so
    # any password length is safe to hash. bcrypt itself still provides
    # the slow, salted hashing that makes this safe to store.
    pre_hashed = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(pre_hashed, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    pre_hashed = hashlib.sha256(password.encode("utf-8")).digest()
    try:
        return bcrypt.checkpw(pre_hashed, hashed.encode("utf-8"))
    except ValueError:
        return False


app = FastAPI(title="DeAilize API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== DATABASE ==========
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free',
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, day),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.commit()


init_db()


# ========== MODELS ==========
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    plan: str


class MeResponse(BaseModel):
    email: str
    plan: str
    used_today: int
    daily_limit: int


class VoiceDNA(BaseModel):
    tone: str
    location: str
    local_refs: list[str] = Field(default_factory=list)
    currency: str
    avg_words: float
    filler: list[str] = Field(default_factory=list)
    vocab: list[str] = Field(default_factory=list)  # personal word choices
    sentence_starters: list[str] = Field(default_factory=list)  # how they open sentences
    punctuation_style: str = ""  # e.g. "uses dashes often", "short paragraphs"


class HumanizeRequest(BaseModel):
    text: str
    voice_samples: str
    location: str
    style: str = "match my voice"  # e.g. "professional", "casual", "persuasive"
    add_story: bool = True   # weave in a short illustrative mini-story (casual-type styles only)
    story_note: str = ""     # optional: a REAL anecdote from the writer to use instead
    persona: str = ""        # optional voice to write in, e.g. "a tired but upbeat student", "a CEO"


class HumanizeResponse(BaseModel):
    humanized_text: str
    voice_dna: VoiceDNA
    used_today: int
    daily_limit: int
    segments: list[dict] = []   # [{"text", "status"}]: kept | rewritten | stiff | locked | story | break
    stats: dict = {}


class ReviewCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    rating: int = Field(ge=1, le=5)
    comment: str = Field(min_length=1, max_length=500)


class ReviewOut(BaseModel):
    name: str
    rating: int
    comment: str
    created_at: str


# ========== AUTH HELPERS ==========
def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_user(authorization: str = Header(default="")) -> sqlite3.Row:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

    with closing(get_db()) as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (payload["sub"],)
        ).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


def get_usage_today(conn: sqlite3.Connection, user_id: int) -> int:
    today = date.today().isoformat()
    row = conn.execute(
        "SELECT count FROM usage WHERE user_id = ? AND day = ?", (user_id, today)
    ).fetchone()
    return row["count"] if row else 0


def increment_usage(conn: sqlite3.Connection, user_id: int) -> int:
    today = date.today().isoformat()
    conn.execute(
        """INSERT INTO usage (user_id, day, count) VALUES (?, ?, 1)
           ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1""",
        (user_id, today),
    )
    conn.commit()
    return get_usage_today(conn, user_id)


# ========== AUTH ENDPOINTS ==========
@app.post("/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (req.email,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        password_hash = hash_password(req.password)
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, plan, created_at) VALUES (?, ?, 'free', ?)",
            (req.email, password_hash, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        user_id = cursor.lastrowid

    token = create_token(user_id, req.email)
    return AuthResponse(token=token, email=req.email, plan="free")


@app.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    with closing(get_db()) as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (req.email,)
        ).fetchone()

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_token(user["id"], user["email"])
    return AuthResponse(token=token, email=user["email"], plan=user["plan"])


@app.get("/me", response_model=MeResponse)
def me(user: sqlite3.Row = Depends(get_current_user)):
    with closing(get_db()) as conn:
        used_today = get_usage_today(conn, user["id"])
    return MeResponse(
        email=user["email"],
        plan=user["plan"],
        used_today=used_today,
        daily_limit=PLAN_LIMITS.get(user["plan"], PLAN_LIMITS["free"]),
    )


# ========== HUMANIZE PIPELINE ==========
def extract_voice_dna(samples: str, location: str) -> VoiceDNA:
    prompt = f"""Analyze this writing and return JSON with these keys:
tone (string), location (string), local_refs (array, max 3 strings),
currency (3-letter code), avg_words (number), filler (array, max 4 strings),
vocab (array of 10 distinctive personal words), sentence_starters (array of 5 strings),
punctuation_style (one short sentence).

Text: {samples[:800]}
Location: {location}"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = res.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Voice DNA extraction returned invalid JSON: %s", raw)
        raise HTTPException(status_code=502, detail="Voice DNA extraction failed") from exc
    return VoiceDNA(**data)


def _retry_after(exc: Exception, default: float) -> float:
    """Seconds the provider asked us to wait (Retry-After header), else `default`."""
    try:
        raw = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return min(max(float(raw), 1.0), 70.0) if raw else default
    except Exception:  # noqa: BLE001
        return default


def _chat(prompt: str) -> str:
    """One model call. Rate limits (HTTP 429) wait exactly as long as the
    provider says; other errors back off exponentially."""
    last_exc: Exception | None = None
    extra = {"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in MODEL else {}
    for attempt in range(6):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                **extra,
            )
            return res.choices[0].message.content or ""
        except RateLimitError as exc:
            last_exc = exc
            wait = _retry_after(exc, default=min(2 ** attempt * 4, 60))
            logger.warning("Rate limited; waiting %.0fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait + 0.5)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt * 2
            logger.warning("LLM call failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise HTTPException(status_code=503, detail="The model is busy right now. Please try again in a minute.") from last_exc


STYLE_INSTRUCTIONS = {
    "match my voice": "",
    "professional": "Write in a polished, professional register suitable for business or formal correspondence.",
    "casual": "Write in a relaxed, conversational register, like talking to a friend.",
    "persuasive": "Write persuasively, building a clear case and calling the reader to action.",
    "friendly": "Write in a warm, approachable, friendly register.",
    "academic": "Write in a precise, formal, academic register with careful qualifications.",
}














_SPOKEN_STYLES = ("match my voice", "casual", "friendly")
_STORY_STYLES = ("match my voice", "casual", "friendly", "persuasive")


def smart_rewrite(text: str, style: str, location: str, dna: "VoiceDNA | None" = None,
                  add_story: bool = True, story_note: str = "", progress=None,
                  persona: str = "") -> tuple[str, list[dict], dict]:
    """The full flow: stiffness gate -> chaos lengths for the flagged sentences ->
    personality (whole-section context + persona + story) -> word flipper ->
    judge. Sentences that already read naturally are never touched.
    Returns (text, per-sentence segments for highlighting, stats)."""
    key = style.lower().strip()
    style_line = STYLE_INSTRUCTIONS.get(key, "")
    voice_hint = ""
    if dna is not None:
        voice_hint = (f"tone={dna.tone}; favourite words={', '.join(dna.vocab[:8])}; "
                      f"typical openers={', '.join(dna.sentence_starters[:4])}; {dna.punctuation_style}")
    result = rewrite_post(
        text, _chat, style_line=style_line, voice_hint=voice_hint, persona=persona.strip(),
        location=location, wordset=WORDSET,
        story=add_story and key in _STORY_STYLES,   # formal styles never get an invented story
        story_note=story_note,                       # a real anecdote is always honoured
        casual_words=key in _SPOKEN_STYLES,
        progress=progress, word_pass=WORD_PASS,
    )
    logger.info("flow: %d/%d sentences flagged, %d rewritten, %d still stiff, %.0f%% on length, "
                "burstiness %.2f, story=%s, %d call(s)", result.n_red, result.n_sentences,
                result.n_rewritten, result.n_still_stiff, result.on_target * 100,
                result.burstiness, result.story_added, result.attempts)
    if result.missing:
        logger.warning("%d flagged sentence(s) kept their original text: %s",
                       len(result.missing), result.missing[:5])
    out = result.text
    if key in ("match my voice", "casual", "friendly", "persuasive"):
        out = apply_contractions(out)
    stats = {
        "sentences": result.n_sentences, "flagged": result.n_red, "rewritten": result.n_rewritten,
        "still_stiff": result.n_still_stiff, "story_added": result.story_added,
        "burstiness": round(result.burstiness, 2),
    }
    return out, result.segments, stats


@app.post("/humanize", response_model=HumanizeResponse)
def humanize(req: HumanizeRequest, user: sqlite3.Row = Depends(get_current_user)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not req.voice_samples.strip():
        raise HTTPException(status_code=400, detail="voice_samples must not be empty")

    limit = PLAN_LIMITS.get(user["plan"], PLAN_LIMITS["free"])
    with closing(get_db()) as conn:
        used_today = get_usage_today(conn, user["id"])
        if used_today >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({limit} rewrites/day on the {user['plan']} plan). Try again tomorrow or upgrade.",
            )

    dna = extract_voice_dna(req.voice_samples, req.location)
    text, segments, stats = smart_rewrite(req.text, req.style, req.location, dna, req.add_story,
                                          req.story_note, persona=req.persona)

    # Final safety net — never return empty
    if not text.strip():
        text = req.text

    with closing(get_db()) as conn:
        used_today = increment_usage(conn, user["id"])

    # Logged to Google Sheets by default to improve DeAilize's rewriting —
    # see Terms of Service. If Sheets isn't configured yet, this is skipped
    # (with a warning in the logs) rather than failing the rewrite.
    try:
        ws = get_training_worksheet()
        if ws:
            ws.append_row(
                [
                    datetime.now(timezone.utc).isoformat(),
                    user["id"],
                    req.text,
                    req.voice_samples,
                    req.location,
                    req.style,
                    text,
                ]
            )
    except Exception as exc:
        logger.error("Failed to log training data to Sheets: %s", exc)

    return HumanizeResponse(
        humanized_text=text,
        voice_dna=dna,
        used_today=used_today,
        daily_limit=limit,
        segments=segments,
        stats=stats,
    )


# ========== STREAMING ENDPOINT ==========
def _sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@app.post("/humanize/stream")
async def humanize_stream(
    req: HumanizeRequest,
    user: sqlite3.Row = Depends(get_current_user),
):
    """Streaming version of /humanize. Sends Server-Sent Events so the
    browser sees progress during the pipeline and Render's 30-second
    hard timeout never fires (the connection stays alive via stage events
    between each blocking Groq call)."""

    limit = PLAN_LIMITS.get(user["plan"], PLAN_LIMITS["free"])
    with closing(get_db()) as conn:
        used_today = get_usage_today(conn, user["id"])
        if used_today >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({limit} rewrites/day on the {user['plan']} plan). "
                       "Try again tomorrow or upgrade.",
            )

    async def _with_progress(call):
        """Run a blocking call in a thread; yield its progress messages and a
        keepalive ping so long posts never look dead to the browser or Render."""
        q: "queue.Queue[str]" = queue.Queue()
        task = asyncio.create_task(asyncio.to_thread(call, q.put))
        last_ping = time.monotonic()
        while not task.done():
            while not q.empty():
                yield ("stage", q.get_nowait())
            if time.monotonic() - last_ping > 10:
                yield ("ping", "")
                last_ping = time.monotonic()
            await asyncio.sleep(0.5)
        while not q.empty():
            yield ("stage", q.get_nowait())
        yield ("result", task.result())

    async def generate():
        text = req.text  # safe fallback throughout
        segments: list[dict] = []
        stats: dict = {}
        dna = None
        used = used_today
        is_free = user["plan"] == "free"

        try:
            if is_free:
                # FREE: word pass + rewrite (2 model calls), no Voice DNA
                yield _sse("stage", {"message": "Checking which sentences read stiff…"})
                async for kind, val in _with_progress(
                        lambda cb: smart_rewrite(req.text, req.style, req.location, None,
                                                 req.add_story, req.story_note, cb, req.persona)):
                    if kind == "stage":
                        yield _sse("stage", {"message": val})
                    elif kind == "ping":
                        yield ": keepalive\n\n"
                    else:
                        text, segments, stats = val
            else:
                # PAID: Voice DNA + word pass + rewrite (3 model calls)
                yield _sse("stage", {"message": "Extracting your voice profile…"})
                dna = await asyncio.to_thread(extract_voice_dna, req.voice_samples, req.location)
                yield _sse("stage", {"message": "Rewriting your post…"})
                async for kind, val in _with_progress(
                        lambda cb: smart_rewrite(req.text, req.style, req.location, dna,
                                                 req.add_story, req.story_note, cb, req.persona)):
                    if kind == "stage":
                        yield _sse("stage", {"message": val})
                    elif kind == "ping":
                        yield ": keepalive\n\n"
                    else:
                        text, segments, stats = val

            # Safety net
            if not text.strip():
                text = req.text

            # ---- Log to Sheets ----
            try:
                ws = get_training_worksheet()
                if ws:
                    await asyncio.to_thread(
                        ws.append_row,
                        [
                            datetime.now(timezone.utc).isoformat(),
                            user["id"],
                            req.text,
                            req.voice_samples,
                            req.location,
                            req.style,
                            text,
                        ],
                    )
            except Exception as exc:
                logger.error("Sheets log failed: %s", exc)

            # ---- Increment usage ----
            with closing(get_db()) as conn:
                used = increment_usage(conn, user["id"])

            # ---- Stream result word by word ----
            yield _sse("stage", {"message": "Done — streaming result…"})
            words = text.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield _sse("token", {"text": token})
                await asyncio.sleep(0.015)  # pacing for visual effect

            # ---- Done signal ----
            yield _sse("done", {
                "used_today": used,
                "daily_limit": limit,
                "plan": user["plan"],
                "voice_dna": dna.model_dump() if dna else {},
                "segments": segments,
                "stats": stats,
            })

        except Exception as exc:
            logger.error("Stream pipeline error: %s", exc)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tells nginx/Render not to buffer
            "Connection": "keep-alive",
        },
    )



@app.post("/reviews", response_model=ReviewOut)
def create_review(req: ReviewCreate):
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO reviews (name, rating, comment, created_at) VALUES (?, ?, ?, ?)",
            (req.name.strip(), req.rating, req.comment.strip(), created_at),
        )
        conn.commit()
    return ReviewOut(name=req.name.strip(), rating=req.rating, comment=req.comment.strip(), created_at=created_at)


@app.get("/reviews", response_model=list[ReviewOut])
def list_reviews():
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT name, rating, comment, created_at FROM reviews ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return [ReviewOut(**dict(row)) for row in rows]


@app.get("/")
def health():
    return {"status": "DeAilize API running"}
