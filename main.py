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
    GROQ_MODEL              Defaults to "openai/gpt-oss-120b" — see
                            console.groq.com/docs/models for the current list
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

Note: Groq's free tier enforces its own requests/tokens-per-minute limits,
separate from this app's PLAN_LIMITS. A single rewrite makes ~4 model
calls (voice DNA extraction + up to 3 rewrite passes), so you may hit
those limits faster than expected — if a rewrite fails with a rate-limit
error from Groq, that's the free tier, not a bug here.

Run:
    uvicorn main:app --reload
"""

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

import bcrypt
import gspread
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2.service_account import Credentials
from openai import OpenAI
from pydantic import BaseModel, EmailStr, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wordloom")

# ========== CONFIG ==========
# Groq exposes an OpenAI-compatible API, so we use the plain OpenAI client
# pointed at Groq's base URL instead of AzureOpenAI.
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
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
    base_url="https://api.groq.com/openai/v1",
)


# ========== GOOGLE SHEETS (training data log) ==========
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


class HumanizeResponse(BaseModel):
    humanized_text: str
    voice_dna: VoiceDNA
    used_today: int
    daily_limit: int


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
    prompt = f"""Analyze this writing sample and location deeply. Return a JSON
object with exactly these keys:

- tone (string): the overall emotional register
- location (string): the writer's location
- local_refs (array of strings): local slang, references, place names natural to {location}
- currency (3-letter string): currency used in this location
- avg_words (number): average sentence length in words
- filler (array of strings): filler words/phrases this writer uses, e.g. "you know", "honestly", "like"
- vocab (array of 20 strings): the most distinctive, personal, non-generic words
  this writer uses — words that mark their individual style, NOT common words
  like "the", "and", "is". Think: unusual verbs, adjectives, specific nouns
  they reach for. These will be used to substitute AI-preferred vocabulary.
- sentence_starters (array of 8 strings): the actual words/phrases this writer
  uses to begin sentences, e.g. "Look,", "The thing is", "And yet", "I've always"
- punctuation_style (string): one sentence describing how this writer uses
  punctuation — e.g. "uses em-dashes for asides, short paragraphs, rarely uses semicolons"

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


STYLE_INSTRUCTIONS = {
    "match my voice": "",
    "professional": "Write in a polished, professional register suitable for business or formal correspondence.",
    "casual": "Write in a relaxed, conversational register, like talking to a friend.",
    "persuasive": "Write persuasively, building a clear case and calling the reader to action.",
    "friendly": "Write in a warm, approachable, friendly register.",
    "academic": "Write in a precise, formal, academic register with careful qualifications.",
}


def rewrite_sentence_by_sentence(text: str, dna: VoiceDNA, style: str = "match my voice") -> str:
    """Split the text into sentences and rewrite each one individually,
    then stitch them back together. This catches AI patterns that survive
    whole-paragraph rewrites because the model averages them out."""
    style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "")
    style_instruction = f" Style register: {style_line}." if style_line else ""

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return text

    # Group into small chunks of 2-3 sentences so the model has
    # enough context to vary rhythm between sentences, but not so
    # much that it averages everything out again
    chunk_size = 2
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    def rewrite_chunk(args):
        i, chunk = args
        chunk_text = " ".join(chunk)
        is_first = i == 0
        is_last = i == len(chunks) - 1

        prompt = f"""Rewrite these {len(chunk)} sentence(s) to sound like a specific human.

Writer profile:
- Tone: {dna.tone}
- Personal vocabulary (use these words where natural): {dna.vocab}
- Sentence starters they use: {dna.sentence_starters}
- Filler words/phrases: {dna.filler}
- Punctuation style: {dna.punctuation_style}
- Location: {dna.location}{style_instruction}

STRICT rules for these sentences:
- NEVER use: "it's worth noting", "additionally", "furthermore", "in conclusion",
  "importantly", "notably", "overall", "in summary", "clearly", "obviously",
  "it should be noted", "undoubtedly", "certainly", "one must", "this highlights"
- NO bullet points or lists
- Use contractions: don't, it's, won't, can't, I've, we're
- Vary sentence length from the surrounding context
- {"Start with one of the writer's natural sentence starters." if is_first else ""}
- {"End with an abrupt or unexpected sentence — no tidy conclusion." if is_last else ""}
- Replace generic verbs/adjectives with words from the writer's personal vocabulary
- Preserve every fact and all meaning

Sentences to rewrite:
{chunk_text}

Return ONLY the rewritten sentences as plain text. No labels, no explanation."""

        return i, _chat(prompt).strip()

    # Run all chunks in parallel — cuts wall-clock time from
    # (n_chunks × latency) to roughly (max_latency of any single chunk)
    rewritten_chunks = [""] * len(chunks)
    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as executor:
        futures = {executor.submit(rewrite_chunk, (i, chunk)): i
                   for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            i, result = future.result()
            rewritten_chunks[i] = result

    return " ".join(rewritten_chunks)


def llm_rewrite(text: str, dna: VoiceDNA, style: str = "match my voice") -> str:
    style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "")
    style_instruction = f"\n- Style register: {style_line}" if style_line else ""

    prompt = f"""You are ghostwriting for a specific human. Their writing profile:
{dna.model_dump_json()}

Your job is to rewrite the text below so it reads as genuinely human-written.
The biggest mistakes AI makes that get it caught:

BANNED — never use these:
- Bullet points, numbered lists, headers of any kind
- "it's worth noting", "additionally", "furthermore", "in conclusion",
  "it is important to", "one must consider", "this highlights", "this
  underscores", "overall", "in summary", "notably", "importantly",
  "it should be noted", "as previously mentioned", "in other words",
  "to summarize", "clearly", "obviously", "undoubtedly", "certainly"
- Starting consecutive sentences with the same word
- Smooth logical transitions between every sentence — humans jump topics

REQUIRED — do all of these:
- Mix sentence lengths violently: some 3-5 words, some 20-25 words, never
  two similar lengths in a row
- Write at least one sentence that starts mid-thought, as if continuing
  something unsaid
- Include one slightly imprecise or colloquial word choice where a more
  "correct" word exists — the kind of choice a real person makes
- Add one throwaway observation in parentheses or after a dash that feels
  like a genuine aside, not a crafted example
- Write one sentence that mildly contradicts or complicates what came before
  — humans are inconsistent, AI isn't
- Use contractions aggressively: don't, it's, won't, can't, I've, they're
- Use the writer's filler words: {dna.filler}
- Use the writer's personal vocabulary: {dna.vocab}
- Use the writer's sentence starters: {dna.sentence_starters}
- Match the writer's punctuation style: {dna.punctuation_style}
- Reference {dna.location} once, woven in naturally not bolted on
- One very short paragraph of 1-2 sentences followed by a longer one
- Preserve every fact and the full meaning of the original{style_instruction}

Return ONLY the rewritten text. Nothing else.

Original:
{text}"""
    return _chat(prompt)


def chaos_engine(text: str, dna: VoiceDNA) -> str:
    """Post-processing pass that adds human imperfection patterns
    the LLM won't produce on its own."""
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    if not sentences:
        return text

    result = []
    for i, sentence in enumerate(sentences):
        words = sentence.split()

        # Randomly contract some formal phrases
        contractions = {
            "do not": "don't", "it is": "it's", "they are": "they're",
            "we are": "we're", "you are": "you're", "I am": "I'm",
            "cannot": "can't", "will not": "won't", "would not": "wouldn't",
            "should not": "shouldn't", "does not": "doesn't",
            "did not": "didn't", "is not": "isn't", "are not": "aren't",
        }
        s = sentence
        for formal, contracted in contractions.items():
            if random.random() < 0.6 and formal in s:
                s = s.replace(formal, contracted, 1)

        # Occasionally split a long sentence into two with a dash or
        # em-dash for a more natural, mid-thought feel
        if len(words) > 18 and random.random() < 0.35:
            mid = len(words) // 2
            s = " ".join(words[:mid]) + " — " + words[mid].lower() + " " + " ".join(words[mid + 1:])

        result.append(s)

        # Insert a short punchy follow-up after a long sentence
        if len(words) > 15 and random.random() < 0.25 and dna.filler:
            result.append(random.choice(dna.filler))

    # Occasional imperfect opener — humans don't always start formally
    if result and random.random() < 0.3:
        starters = ["Look,", "Honestly,", "Real talk —", "Here's the thing:"]
        result[0] = random.choice(starters) + " " + result[0][0].lower() + result[0][1:]

    return " ".join(result)


def diagnose_ai_patterns(text: str) -> list[str]:
    """Ask the model to identify which specific AI-detection patterns
    are present in the text. Returns a list of pattern names found."""
    prompt = f"""You are an AI-detection expert. Analyze this text and identify
which of these specific AI writing patterns are present. Return ONLY a JSON
array of pattern names that apply. Choose from:
- "lack_of_personal_voice"
- "repetitive_keywords"
- "excessive_qualification"
- "similar_sentence_rhythms"
- "predictable_sentence_endings"
- "excessive_list_formatting"
- "high_phrase_predictability"
- "smooth_transitions"
- "no_contradictions"
- "overly_formal_diction"

Text to analyze:
{text}

Return ONLY valid JSON like: ["pattern1", "pattern2"]
No explanation, no preamble."""
    try:
        raw = _chat(prompt).strip()
        # Strip any markdown code fences the model might add
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
        patterns = json.loads(raw)
        return patterns if isinstance(patterns, list) else []
    except Exception:
        return []


def targeted_fix(text: str, dna: VoiceDNA, patterns: list[str]) -> str:
    """Fix only the specific AI patterns diagnosed, leaving everything else alone."""
    if not patterns:
        return text

    fix_instructions = {
        "lack_of_personal_voice": "Add first-person opinions and reactions ('I think', 'I've found', 'honestly'). Make the writer's perspective clear.",
        "repetitive_keywords": "Replace repeated key terms with synonyms, pronouns, or restructured sentences that avoid reusing the same words.",
        "excessive_qualification": "Remove hedging words like 'might', 'could', 'perhaps', 'possibly', 'may'. Make statements more direct and confident.",
        "similar_sentence_rhythms": "Break the rhythm pattern — add a 3-word sentence, then a 25-word one. Make consecutive sentences structurally different.",
        "predictable_sentence_endings": "End some sentences abruptly, mid-thought, or with a question. Avoid always ending on a complete noun phrase.",
        "excessive_list_formatting": "Convert any remaining lists to flowing prose. No bullets or numbered items.",
        "high_phrase_predictability": "Replace predictable phrase combinations with unexpected word choices. Use colloquial or imprecise words where a 'correct' word feels too clean.",
        "smooth_transitions": "Remove smooth connective tissue between some sentences. Let two sentences sit next to each other without an explicit link.",
        "no_contradictions": "Add one sentence that pushes back on or slightly contradicts a claim made earlier in the text.",
        "overly_formal_diction": "Replace formal vocabulary with conversational equivalents. Use contractions wherever possible.",
    }

    instructions = "\n".join(
        f"- {fix_instructions[p]}" for p in patterns if p in fix_instructions
    )

    prompt = f"""You are editing text to remove specific AI-detection patterns.
ONLY fix the patterns listed below. Do NOT rewrite the whole text.
Change the minimum number of words/sentences needed to fix each pattern.
Preserve all facts, meaning, and the overall voice.

Patterns to fix:
{instructions}

Text:
{text}

Return ONLY the fixed text. No explanation."""
    return _chat(prompt)


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

    # Pass 1: whole-text rewrite to establish voice and structure
    text = llm_rewrite(req.text, dna, req.style)

    # Pass 2: sentence-by-sentence pass with personal vocab injection
    # This catches patterns the whole-text rewrite averages out
    text = rewrite_sentence_by_sentence(text, dna, req.style)
    text = chaos_engine(text, dna)

    # Pass 3: diagnosis-and-fix loop — identify remaining AI patterns
    # and surgically fix only those. Cap at 1 round to stay within
    # Groq's free tier rate limits (pipeline is already 4+ calls deep).
    patterns = diagnose_ai_patterns(text)
    logger.info("AI patterns detected after sentence pass: %s", patterns)
    if patterns:
        text = targeted_fix(text, dna, patterns)
        text = chaos_engine(text, dna)

    if not passes_quality_check(text, dna):
        text = llm_rewrite(text, dna, req.style)

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
