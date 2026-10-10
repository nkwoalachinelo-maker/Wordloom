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
import hmac
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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
PLAN_LIMITS = {"free": 3, "pro": 200}

# Pro = paid on Selar ($9/month). Access lasts PRO_DAYS from each payment.
ADMIN_KEY = os.environ.get("ADMIN_KEY")                    # for admin.html / manual grants
SELAR_WEBHOOK_SECRET = os.environ.get("SELAR_WEBHOOK_SECRET")
PRO_DAYS = 31

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
    """Mix of 8 logistic-map trajectories: x = r * x * (1 - x).
    r is capped at 3.99 — above 4.0 the map diverges to infinity.
    x is clamped to (0.01, 0.99) to recover from any near-zero collapse.
    Produces sentence-length targets in the 3-30 word range."""
    trajectories = [
        [0.10, 3.57], [0.30, 3.72], [0.50, 3.85],
        [0.70, 3.95], [0.90, 3.63], [0.20, 3.99],
        [0.60, 3.78], [0.40, 3.91],
    ]
    lengths = []
    for i in range(n):
        t = trajectories[i % len(trajectories)]
        r = min(t[1] + (i // len(trajectories)) * 0.0001, 3.99)  # never exceed 3.99
        x = r * t[0] * (1 - t[0])
        x = max(0.01, min(0.99, x))  # clamp so it never collapses or explodes
        t[0] = x
        lengths.append(int(3 + x * 27))  # 3-30 word range
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

    prompt = f"""You are a text humanizer. Your goal: make this text pass as human-written.
Here is the exact 5-engine method you must apply:

ENGINE 1 — BURSTINESS (30/20 rule):
30% of sentences MUST be under 8 words. Short. Punchy. Final.
20% of sentences MUST be over 22 words, the kind that trail a bit before landing.
Never write three consecutive sentences of similar length.
Target sentence lengths in order: {chaos_str}

ENGINE 2 — PERPLEXITY BREAKERS:
Add 3 of these fillers naturally: "honestly", "like", "you know", "kinda", "actually"
Start one sentence with "And", "But", or "Like"
Add at least one em-dash used mid-thought — like this
Use casual/broken phrasing at least once: "Another time I was..." or "She said it didn't..."

ENGINE 3 — WORD FLIPPER:
Preferred vocabulary: {simple_words}
DELETE these words entirely — do not replace, just remove or restructure:
crucial, delve, tapestry, realm, underscore, moreover, furthermore,
utilize, leverage, paradigm, synergy, holistic, groundbreaking, robust,
"it's worth noting", "it is important", "in conclusion", "in summary",
"notably", "clearly", "obviously", "seamlessly"

ENGINE 4 — HUMAN MEMORY INJECTION:
Add 1 specific real-world detail: a place name ({location} if relevant),
a person ("my mate", "this designer I know"), a specific time ("last week",
"that morning", "30 minutes"). Replace any hypothetical ("imagine someone
who...") with a real anecdote format ("Like this one time...").

ENGINE 5 — KILL THE ACADEMIC STRUCTURE:
Do NOT write a summary or conclusion paragraph.
Do NOT use: "In conclusion", "Ultimately", "To summarize", "In summary",
"Overall", "In closing". Just end. The last sentence is a statement, not a wrap-up.{style_instruction}

Preserve every fact and meaning from the original.
Return ONLY the rewritten text. Nothing else.

Text: {_truncate(text)}"""

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
        try:
            conn.execute("ALTER TABLE users ADD COLUMN pro_until TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pending_pro (
                email TEXT PRIMARY KEY, pro_until TEXT NOT NULL)"""
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
    pro_until: str | None = None


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


class GrantRequest(BaseModel):
    email: EmailStr
    days: int = Field(default=31, ge=1, le=366)


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def effective_plan(user) -> str:
    """'pro' only while a paid period is still running."""
    try:
        if user["plan"] == "pro" and user["pro_until"] and datetime.fromisoformat(user["pro_until"]) > _now():
            return "pro"
    except (ValueError, TypeError):
        pass
    return "free"


def require_pro(user: sqlite3.Row = Depends(get_current_user)) -> sqlite3.Row:
    if effective_plan(user) != "pro":
        raise HTTPException(status_code=403, detail="Pro plan required. Upgrade on the DeAilize landing page.")
    return user


def grant_pro(email: str, days: int = PRO_DAYS) -> str:
    """Start or extend Pro for an email. If they haven't signed up yet, the
    grant waits in pending_pro and is applied when they create the account."""
    email = email.strip().lower()
    with closing(get_db()) as conn:
        row = conn.execute("SELECT id, pro_until FROM users WHERE lower(email) = ?", (email,)).fetchone()
        base = _now()
        existing = row["pro_until"] if row else None
        if existing:
            try:
                base = max(base, datetime.fromisoformat(existing))
            except ValueError:
                pass
        until = (base + timedelta(days=days)).isoformat()
        if row:
            conn.execute("UPDATE users SET plan = 'pro', pro_until = ? WHERE id = ?", (until, row["id"]))
        else:
            conn.execute("INSERT OR REPLACE INTO pending_pro (email, pro_until) VALUES (?, ?)", (email, until))
        conn.commit()
    return until


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
        pend = conn.execute("SELECT pro_until FROM pending_pro WHERE email = ?", (req.email.strip().lower(),)).fetchone()
        plan = "free"
        if pend:
            conn.execute("UPDATE users SET plan = 'pro', pro_until = ? WHERE id = ?", (pend["pro_until"], user_id))
            conn.execute("DELETE FROM pending_pro WHERE email = ?", (req.email.strip().lower(),))
            conn.commit()
            plan = "pro"

    token = create_token(user_id, req.email)
    return AuthResponse(token=token, email=req.email, plan=plan)


@app.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    with closing(get_db()) as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (req.email,)
        ).fetchone()

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_token(user["id"], user["email"])
    return AuthResponse(token=token, email=user["email"], plan=effective_plan(user))


@app.get("/me", response_model=MeResponse)
def me(user: sqlite3.Row = Depends(get_current_user)):
    with closing(get_db()) as conn:
        used_today = get_usage_today(conn, user["id"])
    return MeResponse(
        email=user["email"],
        plan=effective_plan(user),
        used_today=used_today,
        daily_limit=PLAN_LIMITS[effective_plan(user)],
        pro_until=user["pro_until"] if effective_plan(user) == "pro" else None,
    )


# ========== HUMANIZE PIPELINE ==========
def extract_voice_dna(samples: str, location: str) -> VoiceDNA:
    prompt = f"""Analyze this writing and return JSON with these keys:
tone (string), location (string), local_refs (array, max 3 strings),
currency (3-letter code), avg_words (number), filler (array, max 4 strings),
vocab (array of 10 distinctive personal words), sentence_starters (array of 5 strings),
punctuation_style (one short sentence).

Text: {_truncate(samples, 600)}
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


# Safe word cap for text sent to the LLM in a single request.
# openai/gpt-oss-20b has an 8000 TPM per-request limit.
# Prompt overhead (~500 tokens) + completion (~1000 tokens) leaves ~6500
# tokens for input text. 1 token ≈ 0.75 words → 6500 * 0.75 = ~4875 words.
# We cap at 4000 words to stay comfortably under.
MAX_INPUT_WORDS = 4000


def _truncate(text: str, max_words: int = MAX_INPUT_WORDS) -> str:
    """Truncate text to max_words, breaking at a sentence boundary where
    possible so the result doesn't end mid-sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at the last sentence boundary
    last_stop = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_stop > len(truncated) * 0.7:  # only use if it's in the last 30%
        truncated = truncated[:last_stop + 1]
    return truncated + " [document truncated — upload shorter sections for best results]"


def _chat(prompt: str) -> str:
    """Single LLM call. Priority order:
    1. LOCAL_LLM_URL — llama.cpp server on Oracle Cloud running
       Phi-3.5-mini-instruct-Q4_K_M.gguf (or any GGUF model).
       Set LOCAL_LLM_URL=http://<your-oracle-ip>:8080 in Render env vars.
       Set LOCAL_LLM_MODEL to the model name (e.g. phi-3.5-mini).
    2. HUGGINGFACE_API_KEY + HUGGINGFACE_MODEL — HuggingFace Inference API.
    3. GROQ_API_KEY — Groq (current default, rate-limited on free tier).
    """
    local_url = os.environ.get("LOCAL_LLM_URL")
    local_model = os.environ.get("LOCAL_LLM_MODEL", "phi-3.5-mini")
    hf_key = os.environ.get("HUGGINGFACE_API_KEY")
    hf_model = os.environ.get("HUGGINGFACE_MODEL")

    if local_url:
        from openai import OpenAI as _OpenAI
        local_client = _OpenAI(
            api_key="local",  # llama.cpp doesn't need a real key
            base_url=local_url.rstrip("/") + "/v1",
        )
        res = local_client.chat.completions.create(
            model=local_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
    elif hf_key and hf_model:
        from openai import OpenAI as _OpenAI
        hf_client = _OpenAI(
            api_key=hf_key,
            base_url="https://api-inference.huggingface.co/v1/",
        )
        res = hf_client.chat.completions.create(
            model=hf_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
    else:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
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


# ========== PROMPT FILE ==========
# All of the main rewrite prompt lives in engine_prompt.txt (edit that file, not this code).
PROMPT_FILE = os.environ.get("PROMPT_FILE", os.path.join(os.path.dirname(__file__), "engine_prompt.txt"))


def load_prompt(section: str) -> str | None:
    """Return one ### SECTION of the prompt file, or None if the file/section is missing."""
    try:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        logger.warning("Prompt file %s not found; using built-in prompt", PROMPT_FILE)
        return None
    kept = "\n".join(l for l in raw.splitlines() if not l.startswith("%%"))
    parts = re.split(r"^###\s*(\w+)\s*$", kept, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        if parts[i].upper() == section.upper():
            return parts[i + 1].strip()
    logger.warning("Section %s missing in %s", section, PROMPT_FILE)
    return None


def fill_prompt(template: str, **values) -> str:
    for key, val in values.items():
        template = template.replace("{{" + key + "}}", str(val))
    return template


def llm_rewrite(text: str, dna: VoiceDNA, style: str = "match my voice", answers: str = "") -> str:
    template = load_prompt("REWRITE")
    if template:
        essay = _truncate(text)
        style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "") or "match the writer's voice"
        return _chat(fill_prompt(
            template,
            essay=essay,
            answers=answers.strip() or "(none given)",
            word_count=len(essay.split()),
            tone=dna.tone, location=dna.location, filler=dna.filler, vocab=dna.vocab,
            style=style_line,
        ))
    # Fallback: original built-in prompt (used only if engine_prompt.txt is missing)
    style_line = STYLE_INSTRUCTIONS.get(style.lower().strip(), "")
    style_instruction = f"\nStyle: {style_line}" if style_line else ""

    chaos_sample = random.sample(CHAOS_LENGTHS, 5)
    chaos_str = ", ".join(map(str, chaos_sample))
    sample_size = min(15, len(WORD_LIST))
    simple_words = ", ".join(random.sample(WORD_LIST, sample_size))

    prompt = f"""Ghostwrite this text as the specific human writer described below.
Apply all 5 engines:

WRITER PROFILE: tone={dna.tone}, location={dna.location},
filler words={dna.filler}, personal vocab={dna.vocab},
sentence starters={dna.sentence_starters},
punctuation style={dna.punctuation_style}{style_instruction}

ENGINE 1 — BURSTINESS (30/20 rule):
30% of sentences under 8 words. 20% over 22 words.
Never three consecutive sentences of similar length.
Target lengths in order: {chaos_str}

ENGINE 2 — PERPLEXITY BREAKERS:
Add 3 fillers naturally from: {dna.filler} or "honestly", "like", "you know"
Start one sentence with "And", "But", or "Like"
One em-dash used mid-thought — like this
One casual/broken phrase: "Another time I was..." or "She said it didn't..."

ENGINE 3 — WORD FLIPPER:
Use vocab: {dna.vocab} and these simple words: {simple_words}
DELETE entirely: crucial, delve, tapestry, realm, underscore, moreover,
furthermore, utilize, leverage, paradigm, synergy, holistic, groundbreaking,
robust, "it's worth noting", "in conclusion", "in summary", "notably",
"clearly", "obviously", "seamlessly"

ENGINE 4 — HUMAN MEMORY:
Add 1 specific real detail referencing {dna.location} or a person/time.
Replace any hypothetical ("imagine someone...") with real anecdote format.

ENGINE 5 — KILL THE STRUCTURE:
No conclusion paragraph. No "In conclusion", "Ultimately", "To summarize".
Just end on a plain statement.

Preserve every fact. Return ONLY the rewritten text.

Text: {_truncate(text)}"""
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
def humanize(req: HumanizeRequest, user: sqlite3.Row = Depends(require_pro)):
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
    text = llm_rewrite(req.text, dna, req.style, req.voice_samples)
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
        fallback = llm_rewrite(text, dna, req.style, req.voice_samples)
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
    user: sqlite3.Row = Depends(require_pro),
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
                text = await asyncio.to_thread(llm_rewrite, req.text, dna, req.style, req.voice_samples)
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
                    text = await asyncio.to_thread(llm_rewrite, text, dna, req.style, req.voice_samples)

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


# ========== FREE TIER + PRO ACCESS ==========
@app.post("/free/use")
def free_use(user: sqlite3.Row = Depends(get_current_user)):
    """Landing page calls this before each free rewrite (which runs on the
    Hugging Face Space). Counts against the daily free limit."""
    plan = effective_plan(user)
    limit = PLAN_LIMITS[plan]
    with closing(get_db()) as conn:
        used = get_usage_today(conn, user["id"])
        if plan == "free" and used >= limit:
            raise HTTPException(status_code=429, detail=f"You've used your {limit} free rewrites today. Come back tomorrow or go Pro.")
        if plan == "free":
            used = increment_usage(conn, user["id"])
    return {"used_today": used, "daily_limit": limit, "plan": plan}


def _check_admin(key: str):
    if not ADMIN_KEY:
        raise HTTPException(status_code=503, detail="ADMIN_KEY is not set on the server.")
    if not hmac.compare_digest(key or "", ADMIN_KEY):
        raise HTTPException(status_code=403, detail="Wrong admin key.")


@app.post("/admin/grant-pro")
def admin_grant_pro(req: GrantRequest, x_admin_key: str = Header(default="")):
    _check_admin(x_admin_key)
    return {"email": req.email, "pro_until": grant_pro(req.email, req.days)}


def _find_email(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ("email", "customer_email", "buyer_email") and isinstance(v, str) and "@" in v:
                return v
        for v in obj.values():
            found = _find_email(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_email(v)
            if found:
                return found
    return None


@app.post("/webhooks/selar")
async def selar_webhook(request: Request, secret: str = ""):
    """Best-effort Selar hook: URL is /webhooks/selar?secret=<SELAR_WEBHOOK_SECRET>.
    Looks for the buyer's email anywhere in the JSON body and grants Pro.
    Check Selar's current webhook docs/payload; use admin.html if it doesn't fit."""
    if not SELAR_WEBHOOK_SECRET or not hmac.compare_digest(secret, SELAR_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Expected JSON.")
    logger.info("Selar webhook received")
    email = _find_email(body)
    if not email:
        raise HTTPException(status_code=422, detail="No buyer email found in payload.")
    return {"ok": True, "pro_until": grant_pro(email)}


class QuestionsRequest(BaseModel):
    essay: str = Field(min_length=1)


@app.post("/engine/questions")
def engine_questions(req: QuestionsRequest, user: sqlite3.Row = Depends(require_pro)):
    """Step 1 of the prompt file: questions about the essay for the writer to answer.
    Send the answers back in the `voice_samples` field of /humanize."""
    template = load_prompt("QUESTIONS")
    if not template:
        raise HTTPException(status_code=503, detail="Prompt file or QUESTIONS section is missing.")
    raw = _chat(fill_prompt(template, essay=_truncate(req.essay)))
    lines = [re.sub(r"^\s*(?:\d+[\).:-]|[-*•])\s*", "", l).strip() for l in raw.splitlines() if l.strip()]
    return {"questions": lines[:5]}


@app.get("/")
def health():
    return {"status": "DeAilize API running"}
