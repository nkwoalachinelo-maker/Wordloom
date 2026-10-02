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

import asyncio
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
from fastapi.responses import StreamingResponse
from google.oauth2.service_account import Credentials
from openai import OpenAI
from pydantic import BaseModel, EmailStr, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wordloom")

# ========== CONFIG ==========
# Groq exposes an OpenAI-compatible API, so we use the plain OpenAI client
# pointed at Groq's base URL instead of AzureOpenAI.
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
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


# ========== CHAOS ENGINE + WORD LIST ==========
# Loads words_alpha.txt if it exists in the repo (commit the 4MB file to
# GitHub so Render can find it). Falls back to a built-in simple-word list
# so the pipeline never crashes if the file isn't there.
_FALLBACK_WORDS = [
    "use", "help", "get", "make", "good", "need", "want", "think", "know",
    "see", "find", "give", "take", "try", "work", "look", "ask", "go", "come",
    "say", "tell", "show", "turn", "keep", "let", "put", "mean", "start",
    "feel", "move", "live", "change", "bring", "happen", "write", "read",
    "set", "hold", "run", "hear", "talk", "call", "build", "cut", "buy",
    "small", "big", "real", "free", "easy", "hard", "fast", "slow", "early",
    "late", "long", "short", "high", "low", "old", "young", "new", "right",
    "wrong", "clear", "dark", "light", "open", "close", "full", "empty",
    "strong", "weak", "safe", "busy", "ready", "done", "true", "false",
    "money", "time", "day", "night", "year", "people", "thing", "place",
    "world", "life", "house", "hand", "face", "door", "room", "road",
    "mind", "body", "voice", "heart", "word", "name", "story", "reason",
    "point", "side", "part", "group", "lot", "bit", "step", "move",
    "plan", "idea", "chance", "choice", "deal", "problem", "question",
]

try:
    _words_path = os.path.join(os.path.dirname(__file__), "words_alpha.txt")
    with open(_words_path, "r") as f:
        WORD_LIST = [w.strip() for w in f if len(w.strip()) > 2]
    logger.info("Loaded %d words from words_alpha.txt", len(WORD_LIST))
except Exception:
    WORD_LIST = _FALLBACK_WORDS
    logger.info("words_alpha.txt not found — using built-in word list (%d words)", len(WORD_LIST))


def generate_chaos_lengths(n: int = 1000) -> list[int]:
    """Mix of 8 logistic-map trajectories with different starting points
    and r-values. This gives more variety than a single trajectory and
    produces the full 3-30 word range instead of just 5-25."""
    trajectories = [
        [0.10, 3.57], [0.30, 3.72], [0.50, 3.85],
        [0.70, 3.95], [0.90, 3.63], [0.20, 4.00],
        [0.60, 3.78], [0.40, 3.91],
    ]
    lengths = []
    for i in range(n):
        t = trajectories[i % len(trajectories)]
        r = t[1] + (i // len(trajectories)) * 0.0002
        t[0] = r * t[0] * (1 - t[0])
        # Map 0-1 → 3-30 words for more aggressive burstiness
        lengths.append(int(3 + t[0] * 27))
    return lengths


# Pre-generate once at startup — reused across all requests
CHAOS_LENGTHS = generate_chaos_lengths(1000)
logger.info("Generated %d chaos sentence-length targets", len(CHAOS_LENGTHS))


# ========== WORD FLIP ENGINE ==========
# Each entry is a list ranked from MOST AI-used (index 0) to
# LEAST AI-used (last index). The weighted picker below makes
# choices near the END of each list far more likely — so the
# engine tends toward vocabulary AI would almost never choose.
AI_FLIP_MAP: dict[str, list[str]] = {
    # Ranked: index 0 = AI uses a lot → last = AI almost never uses.
    # Quadratic weighting strongly favours the tail end.
    # ---- Verbs ----
    "utilize":      ["use", "apply", "work with", "make use of", "put to work", "get some mileage from"],
    "leverage":     ["use", "tap", "lean on", "bank on", "make the most of", "squeeze what you can from"],
    "delve":        ["look at", "dig into", "get into", "wade through", "poke around in", "really get into the weeds of"],
    "implement":    ["do", "put in place", "roll out", "get going", "set up", "get it off the ground"],
    "facilitate":   ["help", "make easier", "smooth the way", "open the door for", "get out of the way so it can happen"],
    "commence":     ["start", "begin", "kick off", "get going", "get the ball rolling", "get the show started"],
    "endeavor":     ["try", "attempt", "give it a shot", "have a go", "push for", "take a real crack at it"],
    "ascertain":    ["find out", "check", "pin down", "get to the bottom of", "figure out exactly"],
    "elucidate":    ["explain", "clarify", "break down", "lay out", "spell it out", "make it so anyone can follow"],
    "exemplify":    ["show", "prove", "make the case for", "drive home", "put a real face on"],
    "epitomize":    ["capture", "nail", "sum up perfectly", "be the poster child for", "say everything about"],
    "underscore":   ["show", "stress", "hammer home", "drive the point", "make you sit up and notice"],
    "necessitate":  ["need", "require", "call for", "mean we need", "force the question of"],
    "streamline":   ["simplify", "cut down", "trim the fat from", "clean up", "stop overcomplicating"],
    "optimize":     ["improve", "fine-tune", "get right", "tweak", "stop leaving stuff on the table"],
    "harness":      ["use", "tap into", "channel", "put to work", "not waste"],
    "foster":       ["grow", "build", "push", "help along", "give room to breathe"],
    "mitigate":     ["reduce", "cut down", "soften", "take the edge off", "stop it getting worse"],
    "proliferate":  ["spread", "grow fast", "pop up everywhere", "take off", "start showing up constantly"],
    "constitute":   ["make up", "form", "count as", "add up to", "basically be"],
    "culminate":    ["end up", "peak", "come to a head", "finish with", "land on"],
    "exacerbate":   ["worsen", "make worse", "add fuel to the fire", "pour petrol on it"],
    "precipitate":  ["trigger", "cause", "set off", "kick off", "be what tips it over"],
    "disseminate":  ["spread", "share", "push out", "get out there", "make sure it reaches people"],
    "conceptualize":["think up", "imagine", "map out", "come up with", "start to picture"],
    # ---- Adjectives ----
    "comprehensive":["full", "complete", "wall-to-wall", "soup-to-nuts", "the whole nine yards", "nothing left out"],
    "substantial":  ["big", "real", "sizable", "not small", "pretty hefty", "hard to ignore"],
    "significant":  ["big", "major", "worth noting", "real", "not nothing", "the kind you can't ignore"],
    "paramount":    ["key", "top", "number one", "the big one", "front and center", "everything else comes after this"],
    "optimal":      ["best", "ideal", "right", "spot-on", "just what's needed", "the sweet spot"],
    "innovative":   ["new", "fresh", "different", "unlike what came before", "a proper break from the norm"],
    "robust":       ["strong", "solid", "tough", "built to last", "not fragile", "won't fall apart on you"],
    "intricate":    ["complex", "detailed", "fiddly", "tricky", "layered", "takes a while to fully get"],
    "meticulous":   ["careful", "thorough", "precise", "obsessively detailed", "nothing slips through"],
    "crucial":      ["key", "vital", "make-or-break", "the thing that matters", "don't mess this part up"],
    "pivotal":      ["key", "turning-point", "the one that changes things", "where everything hinges"],
    "imperative":   ["needed", "essential", "non-negotiable", "can't skip this", "you have to do this one"],
    "multifaceted": ["complex", "many-sided", "all over the map", "not simple", "way more going on than it looks"],
    "nuanced":      ["subtle", "careful", "not black-and-white", "layered", "more complicated than people think"],
    "groundbreaking":["new", "first-of-its-kind", "never-been-done", "a genuine first", "nobody had pulled this off before"],
    "holistic":     ["full", "all-around", "top-to-bottom", "the whole picture", "leaves nothing out"],
    "unprecedented":["never seen before", "new territory", "first time ever", "nothing like it existed before"],
    "transformative":["life-changing", "game-shifting", "a real shift", "the kind of change that sticks"],
    "cutting-edge": ["latest", "newest", "bleeding-edge", "right at the front", "what everyone will be using in two years"],
    "state-of-the-art":["latest", "newest", "as good as it gets right now", "the best you can get today"],
    # ---- Adverbs ----
    "seamlessly":   ["smoothly", "without a hitch", "clean", "no fuss", "like it was always supposed to work that way"],
    "undoubtedly":  ["no doubt", "for sure", "without question", "clearly", "you'd be hard-pressed to argue otherwise"],
    "fundamentally":["at the core", "deep down", "at its heart", "basically", "if you strip everything back"],
    "inherently":   ["by nature", "naturally", "at its core", "built-in", "baked in from the start"],
    "ultimately":   ["in the end", "when it's all said and done", "at the end of it", "when the dust settles"],
    "essentially":  ["basically", "when you strip it back", "more or less", "if we're being honest about it"],
    # ---- Nouns ----
    "synergy":      ["teamwork", "working together", "clicking as a team", "the way they make each other better"],
    "paradigm":     ["model", "way of thinking", "the playbook", "the frame", "how people have been looking at it"],
    "tapestry":     ["mix", "blend", "patchwork", "mishmash", "jumble", "a real mix of everything"],
    "framework":    ["setup", "approach", "structure", "way of doing it", "how they've laid it out"],
    "ecosystem":    ["world", "space", "environment", "whole setup", "everything that's grown up around it"],
    "landscape":    ["field", "space", "world", "what's out there", "how things look right now"],
    "cornerstone":  ["foundation", "bedrock", "backbone", "the thing it's built on", "where it all starts"],
    "realm":        ["world", "area", "space", "territory", "that whole side of things"],
    "facet":        ["side", "part", "piece", "angle", "one slice of it"],
    "endeavour":    ["try", "attempt", "push", "effort", "real crack at it"],
    # ---- Phrases (longest first) ----
    "it is important to note that":    ["note that", "worth knowing", "heads up —", "one thing —", "quick flag:"],
    "it's worth noting that":          ["keep in mind", "just so you know", "quick note —", "one thing though:"],
    "it is worth noting":              ["note that", "worth knowing", "heads up", "flag this:"],
    "it's worth noting":               ["just so you know", "keep in mind", "note —", "quick one:"],
    "in conclusion":                   ["so", "all in all", "bottom line", "look —", "here's where we end up:"],
    "in summary":                      ["basically", "in short", "so here's the thing", "to put it plainly"],
    "to summarize":                    ["in short", "basically", "to put it plainly", "quick version:"],
    "as previously mentioned":         ["as I said", "going back to what I said", "again", "like I pointed out"],
    "it should be noted":              ["note that", "keep in mind", "worth flagging", "heads up on this:"],
    "in other words":                  ["meaning", "basically", "put differently", "to say it another way"],
    "needless to say":                 ["obviously", "of course", "naturally", "you probably already know"],
    "at the end of the day":           ["when it comes down to it", "in the end", "honestly", "strip it all back and"],
    "moving forward":                  ["from here", "going ahead", "next", "what happens now"],
    "in light of":                     ["given", "because of", "with", "seeing as"],
    "with regard to":                  ["on", "about", "when it comes to", "as for"],
    "in terms of":                     ["for", "on", "when it comes to", "re:", "on the question of"],
    "due to the fact that":            ["because", "since", "given that", "seeing as"],
    "in the event that":               ["if", "when", "should", "in case"],
    "plays a crucial role":            ["matters a lot", "is key", "does the heavy lifting", "is the piece that makes it work"],
    "plays a pivotal role":            ["is key", "makes the difference", "carries a lot of weight", "is really what turns it"],
    "a wide range of":                 ["lots of", "all kinds of", "many", "a bunch of", "every sort of"],
    "a variety of":                    ["lots of", "many", "all sorts of", "a mix of"],
    "it is essential":                 ["you need to", "this matters", "don't skip", "this one's not optional"],
    "it is necessary":                 ["you need to", "this is needed", "must", "no getting around this"],
}

# Sort longest first so multi-word phrases match before their component words
_FLIP_SORTED = sorted(AI_FLIP_MAP.keys(), key=len, reverse=True)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _weighted_flip_pick(options: list[str]) -> str:
    """Pick from a ranked list where index 0 = most AI-like, last = least.
    Quadratic weighting makes the tail (least AI-like) far more likely:
    a 5-option list gives weights [1, 4, 9, 16, 25] — last option is
    25x more likely than first, strongly pushing toward AI-never-used vocab."""
    n = len(options)
    if n == 1:
        return options[0]
    weights = [(i + 1) ** 2 for i in range(n)]
    total = sum(weights)
    r = random.random() * total
    for option, weight in zip(options, weights):
        r -= weight
        if r <= 0:
            return option
    return options[-1]


def word_flip_engine(text: str) -> str:
    """Replace AI-preferred vocabulary using the ranked flip map.
    Picks replacements with a strong bias toward the least AI-like end
    of each word's list. Case-preserving."""
    result = text
    for phrase in _FLIP_SORTED:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for match in reversed(list(pattern.finditer(result))):
            replacement = _weighted_flip_pick(AI_FLIP_MAP[phrase])
            if match.group()[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def chaos_burstiness_engine(text: str) -> str:
    """Restructure sentence lengths to match chaos-derived targets.
    Splits long sentences at natural joints and merges very short ones.
    No LLM required — pure string manipulation."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return text

    chaos_targets = random.sample(CHAOS_LENGTHS, min(len(sentences), len(CHAOS_LENGTHS)))
    result: list[str] = []
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        words = sentence.split()
        word_count = len(words)
        target = chaos_targets[i % len(chaos_targets)]

        if word_count > target + 8:
            # Too long — try to split at a natural joint
            split_patterns = [" but ", " and ", " which ", " that ", " because ",
                               " however ", " so ", " yet ", " while "]
            split_done = False
            for splitter in split_patterns:
                idx = sentence.lower().find(splitter, target * 4)  # look past target point
                if idx != -1:
                    first = sentence[:idx].strip()
                    second = sentence[idx + len(splitter):].strip()
                    if first and second:
                        # Capitalise the split-off second half
                        second = second[0].upper() + second[1:]
                        result.append(first + ".")
                        sentences.insert(i + 1, second)
                        chaos_targets.insert(i + 1, random.choice(CHAOS_LENGTHS))
                        split_done = True
                        break
            if not split_done:
                result.append(sentence)

        elif word_count < 5 and i + 1 < len(sentences):
            # Very short — merge with next if it also short, otherwise keep
            next_words = sentences[i + 1].split()
            if len(next_words) < 10:
                merged = sentence.rstrip(".!?") + " — " + sentences[i + 1][0].lower() + sentences[i + 1][1:]
                result.append(merged)
                i += 2
                continue
            else:
                result.append(sentence)
        else:
            result.append(sentence)

        i += 1

    return " ".join(result)


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


def free_tier_rewrite(text: str, location: str, style: str = "match my voice") -> str:
    """Free tier: single Groq call supervised by the chaos engine and
    words_alpha word list. No Voice DNA (that's paid tier only).
    The LLM reads the chaos targets and word samples and uses them to
    guide how it rewrites — it's not a blind find/replace."""

    chaos_sample = random.sample(CHAOS_LENGTHS, 5)
    chaos_str = ", ".join(map(str, chaos_sample))
    sample_size = min(30, len(WORD_LIST))
    simple_words = ", ".join(random.sample(WORD_LIST, sample_size))

    style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "")
    style_instruction = f"\nStyle register: {style_line}." if style_line else ""

    prompt = f"""You are a text humanizer. Rewrite the text below using these rules:

WORD FLIP — for every word, ask: would an AI pick this? If yes, swap it for a
simpler alternative. Preferred vocabulary to draw from: {simple_words}
NEVER USE: delve, utilize, leverage, furthermore, moreover, synergy, tapestry,
robust, seamlessly, paradigm, holistic, groundbreaking, "it's worth noting",
"in conclusion", "importantly", "notably", "clearly", "obviously"

CHAOS SENTENCE LENGTHS — your sentences must hit these word counts in order:
{chaos_str}
Example: lengths [7, 18, 9] → first sentence 7 words, second 18 words, third 9.
This unpredictable rhythm is what makes text feel genuinely human.

RULES:
- No bullet points, no numbered lists, no headers — prose only
- Use contractions throughout: don't, it's, won't, can't, I've, they're
- Add one casual aside using a dash or parentheses
- One very short sentence (under 6 words), one long one (over 18 words)
- Start at least one sentence mid-thought, as if continuing something unsaid
- Reference {location} naturally if the topic allows
- Preserve every fact and the full meaning{style_instruction}

Return ONLY the rewritten text. Nothing else.

Text: {text}"""

    result = _chat(prompt)
    # Python post-processing safety net — catches anything Groq missed
    result = word_flip_engine(result)
    result = apply_contractions(result)
    return result


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

        # Each chunk gets its own chaos-derived target lengths
        chunk_chaos = random.sample(CHAOS_LENGTHS, min(len(chunk), 3))
        chunk_chaos_str = ", ".join(map(str, chunk_chaos))
        chunk_words = ", ".join(random.sample(WORD_LIST, min(8, len(WORD_LIST))))

        prompt = f"""Rewrite these {len(chunk)} sentence(s) like this human writer.
Tone: {dna.tone} | Location: {dna.location} | Starters: {dna.sentence_starters}
Filler: {dna.filler} | Vocab: {dna.vocab}{style_instruction}

RULES: No lists. Use contractions. Chaos lengths (words): {chunk_chaos_str}.
Prefer: {chunk_words}. Ban: "furthermore","notably","clearly","utilize","delve".
{"First sentence — use a natural starter." if is_first else ""}
{"Last sentence — end abruptly, no neat conclusion." if is_last else ""}
Preserve all meaning.

Rewrite: {chunk_text}

Return ONLY the rewritten text."""

        try:
            result = _chat(prompt).strip()
            # Only use result if non-empty, otherwise keep original chunk
            return i, result if result else chunk_text
        except Exception as exc:
            logger.warning("Chunk %d rewrite failed (%s) — keeping original", i, exc)
            return i, chunk_text

    # Run sequentially (max_workers=1) to avoid bursting openai/gpt-oss-120b's
    # 8000 TPM free-tier limit. Parallel workers caused rate-limit 429 errors.
    rewritten_chunks = [""] * len(chunks)
    original_chunks = [" ".join(c) for c in chunks]
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(rewrite_chunk, (i, chunk)): i
                   for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            try:
                i, result = future.result()
                rewritten_chunks[i] = result if result.strip() else original_chunks[i]
            except Exception as exc:
                logger.error("Chunk rewrite failed: %s", exc)
                i = futures[future]
                rewritten_chunks[i] = original_chunks[i]

    return " ".join(rewritten_chunks)


def llm_rewrite(text: str, dna: VoiceDNA, style: str = "match my voice") -> str:
    style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "")
    style_instruction = f"\nStyle: {style_line}" if style_line else ""

    chaos_sample = random.sample(CHAOS_LENGTHS, 5)
    chaos_str = ", ".join(map(str, chaos_sample))
    sample_size = min(15, len(WORD_LIST))
    simple_words = ", ".join(random.sample(WORD_LIST, sample_size))

    prompt = f"""Rewrite this text to sound like the human writer below. No lists, no headers.

Writer: tone={dna.tone}, location={dna.location}, filler={dna.filler},
vocab={dna.vocab}, starters={dna.sentence_starters}, punctuation={dna.punctuation_style}{style_instruction}

BANNED WORDS: delve, moreover, furthermore, utilize, leverage, synergy, tapestry,
"it's worth noting", "in conclusion", "importantly", "notably", "clearly", "obviously"

CHAOS SENTENCE LENGTHS (follow in order): {chaos_str} words per sentence.
PREFER SIMPLE WORDS like: {simple_words}
USE contractions: don't, it's, won't, can't, I've
ADD one contradiction, one aside (dash or parentheses), reference {dna.location} once.
Preserve all facts.{style_instruction}

Return ONLY the rewritten text.

Text: {text}"""
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
    if not text.strip():
        text = req.text  # fallback to original if rewrite fails

    # Pass 2: sentence-by-sentence pass with personal vocab injection
    sentence_result = rewrite_sentence_by_sentence(text, dna, req.style)
    if sentence_result.strip():  # only apply if we got something back
        text = sentence_result
    text = chaos_engine(text, dna)

    # Pass 3: diagnosis-and-fix — identify remaining AI patterns and
    # surgically fix only those. Guard against empty returns at every step.
    patterns = diagnose_ai_patterns(text)
    logger.info("AI patterns detected after sentence pass: %s", patterns)
    if patterns:
        fixed = targeted_fix(text, dna, patterns)
        if fixed.strip():  # only apply fix if we got something back
            text = fixed
        text = chaos_engine(text, dna)

    if not passes_quality_check(text, dna):
        fallback = llm_rewrite(text, dna, req.style)
        if fallback.strip():
            text = fallback

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

    async def generate():
        text = req.text  # safe fallback throughout
        dna = None
        used = used_today
        is_free = user["plan"] == "free"

        try:
            if is_free:
                # ======================================================
                # FREE TIER: single Groq call supervised by the chaos
                # engine + words_alpha word list. No Voice DNA.
                # Groq reads the chaos targets and word samples and uses
                # them to guide the rewrite — not a blind string replace.
                # Post-processing catches anything Groq missed.
                # ======================================================
                yield _sse("stage", {"message": "Running chaos-guided word flip…"})
                text = await asyncio.to_thread(
                    free_tier_rewrite, req.text, req.location, req.style
                )
                # Chaos burstiness as a final Python pass on top of Groq output
                yield _sse("stage", {"message": "Applying sentence burstiness…"})
                text = await asyncio.to_thread(chaos_burstiness_engine, text)
                minimal_dna = VoiceDNA(
                    tone="conversational", location=req.location,
                    currency="NGN", avg_words=12.0
                )
                text = await asyncio.to_thread(chaos_engine, text, minimal_dna)

            else:
                # ======================================================
                # PAID TIER: full AI pipeline with Voice DNA.
                # Voice DNA → LLM rewrite → sentence pass → diagnose/fix.
                # ======================================================

                # ---- Stage 1: Voice DNA ----
                yield _sse("stage", {"message": "Extracting your voice profile…"})
                dna = await asyncio.to_thread(
                    extract_voice_dna, req.voice_samples, req.location
                )
                yield _sse("stage", {"message": "Voice DNA captured. Rewriting your text…"})

                # ---- Stage 2: Full rewrite ----
                text = await asyncio.to_thread(llm_rewrite, req.text, dna, req.style)
                yield _sse("stage", {"message": "First pass done. Running sentence-level refinement…"})

                # ---- Stage 3: Word flip + contractions on top of LLM output ----
                # Catches any AI vocabulary the LLM missed on its own pass
                text = await asyncio.to_thread(word_flip_engine, text)
                text = await asyncio.to_thread(apply_contractions, text)
                text = await asyncio.to_thread(chaos_burstiness_engine, text)

                # ---- Stage 4: Sentence-by-sentence ----
                text = await asyncio.to_thread(
                    rewrite_sentence_by_sentence, text, dna, req.style
                )
                text = await asyncio.to_thread(chaos_engine, text, dna)
                yield _sse("stage", {"message": "Diagnosing remaining AI patterns…"})

                # ---- Stage 5: Diagnose + fix ----
                patterns = await asyncio.to_thread(diagnose_ai_patterns, text)
                logger.info("Stream: patterns detected: %s", patterns)
                if patterns:
                    yield _sse("stage", {"message": f"Fixing {len(patterns)} AI pattern(s)…"})
                    text = await asyncio.to_thread(targeted_fix, text, dna, patterns)
                    text = await asyncio.to_thread(chaos_engine, text, dna)

                # ---- Stage 6: Quality check ----
                if not await asyncio.to_thread(passes_quality_check, text, dna):
                    yield _sse("stage", {"message": "Final quality pass…"})
                    text = await asyncio.to_thread(llm_rewrite, text, dna, req.style)

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
