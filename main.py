"""
Wordloom backend — FastAPI + SQLite, using Groq's OpenAI-compatible API.

Setup:
    pip install fastapi uvicorn openai pydantic[email] PyJWT "passlib[bcrypt]"

Required environment variables:
    GROQ_API_KEY    Your Groq API key (console.groq.com)
    JWT_SECRET      Long random string used to sign session tokens
Optional:
    GROQ_MODEL      Defaults to "llama-3.3-70b-versatile" — see
                     console.groq.com/docs/models for the current list
    DB_PATH         Defaults to "wordloom.db"
    CORS_ORIGINS    Comma-separated allowed origins, defaults to "*"

Note: Groq's free tier enforces its own requests/tokens-per-minute limits,
separate from this app's PLAN_LIMITS. A single rewrite makes ~4 model
calls (voice DNA extraction + up to 3 rewrite passes), so you may hit
those limits faster than expected — if a rewrite fails with a rate-limit
error from Groq, that's the free tier, not a bug here.

Run:
    uvicorn main:app --reload
"""

import json
import logging
import os
import random
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wordloom")

# ========== CONFIG ==========
# Groq exposes an OpenAI-compatible API, so we use the plain OpenAI client
# pointed at Groq's base URL instead of AzureOpenAI.
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 24
DB_PATH = os.environ.get("DB_PATH", "wordloom.db")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Daily rewrite allowance per plan. Kept low by default since Groq's free
# tier has its own rate limits on top of this app's own limit.
PLAN_LIMITS = {"free": 5, "pro": 200}

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Wordloom API")
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


class HumanizeRequest(BaseModel):
    text: str
    voice_samples: str
    location: str


class HumanizeResponse(BaseModel):
    humanized_text: str
    voice_dna: VoiceDNA
    used_today: int
    daily_limit: int


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

        password_hash = pwd_context.hash(req.password)
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

    if not user or not pwd_context.verify(req.password, user["password_hash"]):
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
    prompt = f"""Analyze this writing sample and location. Return a JSON object with
exactly these keys: tone (string), location (string), local_refs (array of strings,
local slang/references for the given location), currency (3-letter code), avg_words
(number, average sentence length in words), filler (array of strings, filler words/
phrases typical of this writer).

Text: {samples}
Location: {location}
"""
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


def _chat(prompt: str) -> str:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content


def llm_rewrite(text: str, dna: VoiceDNA) -> str:
    out1 = _chat(f"Rewrite this preserving 100% of the meaning:\n\n{text}")
    out2 = _chat(
        f"""Rewrite using this Voice DNA: {dna.model_dump_json()}
Add one natural reference to {dna.location}. Use currency {dna.currency} if money
is mentioned. Write in this tone: {dna.tone}.

Text: {out1}
"""
    )
    out3 = _chat(
        f"Cut unnecessary fluff. Add one clear opinion and one specific example. "
        f"Text: {out2}"
    )
    return out3


def chaos_engine(text: str, dna: VoiceDNA) -> str:
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    if not sentences:
        return text

    for i, sentence in enumerate(sentences):
        if random.random() < 0.3 and len(sentence) > 10:
            words = sentence.split()
            cut = max(1, len(words) // 2)
            sentences[i] = " ".join(words[:cut]) + "..."

    if dna.filler and random.random() < 0.5:
        insert_at = random.randint(0, len(sentences))
        sentences.insert(insert_at, random.choice(dna.filler))

    if dna.local_refs:
        sentences.append(f"That's how we do it in {dna.location.split(',')[0]}.")

    if random.random() < 0.2 and sentences:
        sentences[0] = "And " + sentences[0][0].lower() + sentences[0][1:]

    return " ".join(sentences)


def passes_quality_check(text: str, dna: VoiceDNA) -> bool:
    has_location = dna.location.split(",")[0].lower() in text.lower()
    has_opinion = " i " in f" {text.lower()} " or "honestly" in text.lower()
    return has_location and has_opinion


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
    text = llm_rewrite(req.text, dna)
    text = chaos_engine(text, dna)
    if not passes_quality_check(text, dna):
        text = llm_rewrite(text, dna)

    with closing(get_db()) as conn:
        used_today = increment_usage(conn, user["id"])

    return HumanizeResponse(
        humanized_text=text,
        voice_dna=dna,
        used_today=used_today,
        daily_limit=limit,
    )


@app.get("/")
def health():
    return {"status": "Wordloom API running"}
