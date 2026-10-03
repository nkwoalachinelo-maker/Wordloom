"""
rewrite_engine.py — DeAilize's rewrite core.

Pipeline (nothing here is a hardcoded word list):

  1. WORD PASS.  The model reads the WHOLE post and picks words worth
     reconsidering. For each one it proposes 30 same-meaning candidates that
     fit that exact sentence, ranked 1 (plain, everyday) to 30 (the wording an
     AI would most likely use). Every candidate is checked against
     words_alpha.txt, so only real words survive. We take rank ~5.
  2. CHAOS PLAN. A Lorenz attractor decides two things only: sentence size and
     punctuation (dash / semicolon / colon / parentheses). Nothing else.
  3. REWRITE.  One model call gets the whole post, the chosen swaps and the
     chaos plan. The model applies a swap only where the meaning stays intact.
  4. GUARDS.  Numbers, links, quotes and names must survive, and invented or
     garbled words (checked against words_alpha.txt) are rejected and retried.
     If a rewrite can't pass, the original text is returned untouched.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("wordloom.engine")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*")


# ---------------------------------------------------------------- chaos ----
def lorenz_sequence(n: int, seed: int | None = None, dt: float = 0.01,
                    steps_per_sample: int = 25) -> list[tuple[float, float, float]]:
    """Integrate the Lorenz system (sigma=10, rho=28, beta=8/3) and sample it.
    A tiny seed-dependent nudge gives a different trajectory per request."""
    import random
    rng = random.Random(seed)
    x = 1.0 + rng.uniform(-0.5, 0.5)
    y = 1.0 + rng.uniform(-0.5, 0.5)
    z = 1.0 + rng.uniform(-0.5, 0.5)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    warmup = 500  # let the trajectory settle onto the attractor
    out: list[tuple[float, float, float]] = []
    for i in range(warmup + n * steps_per_sample):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dx * dt, y + dy * dt, z + dz * dt
        if i >= warmup and (i - warmup) % steps_per_sample == 0:
            out.append((x, y, z))
    return out[:n]


def _norm(v: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


MARKS = ("dash", "semicolon", "colon", "parentheses")
_MARK_HELP = {
    "dash": "an em dash (—) for an aside or a turn",
    "semicolon": "a semicolon joining two closely related clauses",
    "colon": "a colon introducing the point",
    "parentheses": "a brief aside in parentheses",
}


@dataclass
class ChaosPlan:
    lengths: list[int] = field(default_factory=list)   # target words per sentence
    marks: dict[int, str] = field(default_factory=dict)  # sentence index -> punctuation mark


def chaos_plan(n_sentences: int, seed: int | None = None, min_len: int = 4,
               max_len: int = 30, mark_share: float = 0.30) -> ChaosPlan:
    """x -> sentence size, y -> which sentences get a special mark,
    z -> which mark. That is all the chaos engine decides."""
    n = max(1, n_sentences)
    pts = lorenz_sequence(n, seed)

    lengths: list[int] = []
    for x, _, _ in pts:
        length = int(round(min_len + _norm(x, -20, 20) * (max_len - min_len)))
        if len(lengths) >= 2:  # never let three sentences in a row be the same size
            window = lengths[-2:] + [length]
            if max(window) - min(window) < 4:
                length = lengths[-1] + 10 if lengths[-1] < 17 else max(min_len, lengths[-1] - 10)
        lengths.append(max(min_len, min(max_len, length)))

    marks: dict[int, str] = {}
    if n >= 3:
        k = max(1, round(n * mark_share))
        ranked = sorted(range(n), key=lambda i: _norm(pts[i][1], -25, 25), reverse=True)
        for i in ranked:
            if len(marks) >= k:
                break
            if i - 1 in marks or i + 1 in marks:  # no two marked sentences in a row
                continue
            marks[i] = MARKS[min(len(MARKS) - 1, int(_norm(pts[i][2], 0, 50) * len(MARKS)))]
    return ChaosPlan(lengths=lengths, marks=marks)


# ---------------------------------------------------------- fact guard ----
_URL = re.compile(r"https?://[^\s)\]]+")
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_QUOTE = re.compile(r"[\"“]([^\"”]{4,})[\"”]")
_NAME = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_NAME_STOP = {"The", "This", "That", "These", "Those", "And", "But", "So", "When",
              "While", "What", "Why", "How", "Here", "There", "They", "Then", "Also"}


def extract_facts(text: str) -> list[str]:
    """Things that must survive a rewrite unchanged."""
    facts: list[str] = []
    facts += _URL.findall(text)
    facts += [m.strip().rstrip(".,") for m in _NUM.findall(text)]
    facts += _QUOTE.findall(text)
    for sentence in _SENT_SPLIT.split(text):
        words = sentence.split()
        tail = " ".join(words[1:])  # skip the first word: capitalised anyway
        for name in _NAME.findall(tail):
            if name not in _NAME_STOP and name != "I":
                facts.append(name)
    seen: set[str] = set()
    return [f for f in facts if f and not (f in seen or seen.add(f))]


def missing_facts(facts: list[str], rewritten: str) -> list[str]:
    return [f for f in facts if f not in rewritten]


# ------------------------------------------------------ word list guard ----
def load_wordset(path: str) -> "frozenset[str] | None":
    """Load words_alpha.txt (one word per line, CRLF-safe). Returns None if the
    file is missing or too small to trust, which disables the word checks."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            words = frozenset(w.strip().lower() for w in f if w.strip())
    except OSError:
        return None
    return words if len(words) > 50_000 else None


def _known(word: str, wordset: "frozenset[str]") -> bool:
    """True if `word` is a real word, allowing the regular endings the list
    may not spell out (plurals, -ed, -ing, -ly)."""
    w = word.lower()
    if w in wordset or len(w) <= 2:
        return True
    forms = []
    for suffix, repl in (("s", ""), ("es", ""), ("ies", "y"), ("ed", ""), ("ed", "e"),
                         ("d", ""), ("ing", ""), ("ing", "e"), ("ly", ""), ("ily", "y"),
                         ("er", ""), ("est", ""), ("ers", ""), ("ness", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 2:
            stem = w[: -len(suffix)] + repl
            forms.append(stem)
            if len(stem) > 2 and stem[-1] == stem[-2]:  # running -> run
                forms.append(stem[:-1])
    return any(f in wordset for f in forms)


def is_real_phrase(phrase: str, wordset: "frozenset[str] | None") -> bool:
    """Every word in `phrase` must be a real dictionary word."""
    tokens = _TOKEN.findall(phrase)
    if not tokens:
        return False
    if not wordset:
        return True
    return all(_known(p, wordset) for t in tokens for p in re.split(r"['’\-]", t.lower()) if p)


def unknown_words(original: str, rewritten: str, wordset: "frozenset[str] | None") -> list[str]:
    """New words in the rewrite that are neither in the original nor in the
    word list: garbled or invented words. Proper nouns are ignored."""
    if not wordset:
        return []
    seen_original = {t.lower() for t in _TOKEN.findall(original)}
    bad: list[str] = []
    for sentence in _SENT_SPLIT.split(rewritten):
        for i, tok in enumerate(_TOKEN.findall(sentence)):
            low = tok.lower()
            if low in seen_original or (i > 0 and tok[0].isupper()):
                continue
            if is_real_phrase(low, wordset):
                continue
            if low not in bad:
                bad.append(low)
    return bad


# ------------------------------------------------- transition density ----
_OPENER = re.compile(r"^[\"'“(]*[A-Z][A-Za-z'’-]+,\s")


def transition_heavy(text: str, max_share: float = 0.2, min_count: int = 3) -> bool:
    """True if too many sentences open with a one-word lead-in plus comma
    ("Moreover, ...", "However, ..."). Structure-based, so no word list."""
    sentences = [x for x in _SENT_SPLIT.split(text.strip()) if x.strip()]
    if len(sentences) < 5:
        return False
    hits = sum(1 for x in sentences if _OPENER.match(x.strip()))
    return hits >= min_count and hits / len(sentences) > max_share


# ------------------------------------------------------------ word pass ----
@dataclass
class Swap:
    word: str          # exact text in the post
    replacement: str   # candidate we picked
    rank: int          # its rank in the model's 1..30 list (30 = most AI-typical)


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def pick_by_rank(candidates: list[str], original: str,
                 wordset: "frozenset[str] | None", target_rank: int = 5) -> tuple[str, int] | None:
    """candidates are ordered 1 (plainest) .. N (most AI-typical). Take the
    valid one closest to `target_rank`, preferring the plainer side on ties."""
    order = sorted(range(len(candidates)), key=lambda i: (abs((i + 1) - target_rank), i))
    for i in order:
        cand = str(candidates[i]).strip()
        if cand and cand.lower() != original.lower() and is_real_phrase(cand, wordset):
            return cand, i + 1
    return None


def plan_swaps(text: str, chat: Callable[[str], str], wordset: "frozenset[str] | None",
               facts: list[str], target_rank: int = 5, n_candidates: int = 30) -> list[Swap]:
    """Ask the model for ranked candidates per word, keep only real words,
    pick the one near `target_rank`. Any failure just means no swaps."""
    n_words = len(text.split())
    n_targets = max(3, min(30, n_words // 10))
    prompt = (
        f"Read this whole post. Pick about {n_targets} words or short phrases that are the "
        "most formal, abstract or over-polished wording, where a more natural word would "
        "keep the exact meaning in THIS post. Skip names, numbers, quotes and technical "
        "terms the author clearly needs.\n"
        f"For each, give {n_candidates} replacement candidates that fit that exact sentence "
        "(same part of speech and same inflection as the original, meaning unchanged), "
        "ranked from 1 = plain everyday wording to "
        f"{n_candidates} = the wording an AI model would most likely write.\n"
        'Return ONLY JSON: {"words":[{"word":"<exact text from the post>","candidates":["...", ...]}]}\n\n'
        f"POST:\n{text}"
    )
    try:
        data = _parse_json(chat(prompt))
    except Exception as exc:  # rate limit etc: rewrite without swaps
        logger.warning("word pass failed, continuing without swaps: %s", exc)
        return []

    swaps: list[Swap] = []
    used: set[str] = set()
    protected = {f.lower() for f in facts}
    for item in data.get("words", []):
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        cands = item.get("candidates", [])
        low = word.lower()
        if (len(word) < 3 or low in used or low in protected or not isinstance(cands, list)
                or not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)):
            continue
        picked = pick_by_rank([str(c) for c in cands], word, wordset, target_rank)
        if picked:
            swaps.append(Swap(word=word, replacement=picked[0], rank=picked[1]))
            used.add(low)
    return swaps


# ---------------------------------------------------------------- prompt ----
def count_sentences(text: str) -> int:
    return len([s for s in _SENT_SPLIT.split(text.strip()) if s.strip()])


def _build_prompt(text: str, plan: ChaosPlan, swaps: list[Swap], style_line: str,
                  voice_hint: str, location: str, story: bool = False,
                  story_note: str = "", casual_words: bool = False) -> str:
    steps = []
    for i, n in enumerate(plan.lengths):
        mark = plan.marks.get(i)
        steps.append(f"#{i + 1}: ~{n} words" + (f", {mark}" if mark else ""))
    parts = [
        "Rewrite the blog post below so it reads like a thoughtful person wrote it: "
        "specific, natural, with a varied rhythm. You can see the WHOLE post, so use "
        "that context to keep every reference clear.",
        "",
        "HARD RULES",
        "- Keep every fact, number, name, link and quote exactly as written. Add no new claims.",
        "- Keep the paragraph breaks. No headers, bullets or lists unless the original has them.",
        "- Meaning and flow come first. If anything below would hurt clarity, ignore it.",
        "",
        "EDITING CHECKLIST (fix these common weaknesses of generic prose, without changing "
        "what the writer actually argues):",
        "- Balance: don't force both-sides symmetry or pros-and-cons pairs. Where the writer takes a "
        "position, state it plainly and commit to it. Never invent opinions the writer didn't hold.",
        "- Predictable phrasing: replace stock, expected phrases with the concrete detail the post "
        "already contains. Avoid the first phrase that comes to mind.",
        "- Personal voice: speak to the reader directly, use the writer's own phrasing and rhythm, "
        "and keep contractions where the tone allows.",
        "- Conclusion: no template wrap-up that restates the points. End on one specific, "
        "concrete thought that grows out of the content.",
        "- Transitions: use few connecting words. Let the order of ideas carry the logic; at most one "
        "explicit transition per paragraph, and never several sentences in a row opening with one.",
        "",
        "SENTENCE PLAN (soft guide: size in words, plus a punctuation mark where listed; "
        "follow in order and cycle if the post is longer):",
        "; ".join(steps),
        "Punctuation marks: " + "; ".join(f"{k} = {v}" for k, v in _MARK_HELP.items())
        + ". Use one only where it reads naturally.",
    ]
    if swaps:
        parts += ["", "WORD SWAPS (apply each only where the meaning stays exactly the same "
                      "in that spot, otherwise keep the original word):"]
        parts += [f'- "{s.word}" -> "{s.replacement}"' for s in swaps]
    if style_line:
        parts += ["", f"STYLE: {style_line}"]
    if voice_hint:
        parts += ["", f"WRITER VOICE: {voice_hint}"]
    if location:
        parts += ["", f"LOCATION: mention {location} only if it fits naturally. Never invent facts about it."]
    if story_note.strip():
        parts += ["", "STORY: at a natural point in the middle, work in this real anecdote from the "
                      "writer, in 2-4 sentences, using only the details given. Add no extra facts "
                      f"to it: {story_note.strip()}"]
    elif story:
        parts += ["", "STORY: somewhere in the middle, add one short illustrative mini-story "
                      "(2-4 sentences, about 60-90 words) that brings a point from the post to life "
                      "and gives it a personal feel. Frame it as an example (\"Picture someone who...\", "
                      "\"Say you're...\"), NOT as the writer's real experience, and do not invent real "
                      "people, places, numbers or statistics. It must flow into the surrounding text."]
    if casual_words:
        parts += ["", "SPOKEN TOUCHES: where the tone allows, work in 1-3 casual, spoken-style words "
                      "you would not find in a formal dictionary, such as a y'all-style contraction or "
                      "a playful coinage (like a labradoodle-type blend). Vary them from post to post, "
                      "and only where they fit naturally and the meaning stays clear."]
    parts += ["", "Return ONLY the rewritten post.", "", "POST:", text]
    return "\n".join(parts)


# ------------------------------------------------------------- main API ----
@dataclass
class RewriteResult:
    text: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    attempts: int = 0
    swaps: list[Swap] = field(default_factory=list)


def rewrite_post(text: str, chat: Callable[[str], str], *, style_line: str = "",
                 voice_hint: str = "", location: str = "",
                 wordset: "frozenset[str] | None" = None, target_rank: int = 5,
                 story: bool = False, story_note: str = "", casual_words: bool = False,
                 max_attempts: int = 3, seed: int | None = None) -> RewriteResult:
    """Word pass -> chaos plan -> whole-post rewrite -> guards (with retry)."""
    original = text.strip()
    if not original:
        return RewriteResult(text=text, ok=False)

    facts = extract_facts(original)
    swaps = plan_swaps(original, chat, wordset, facts, target_rank)
    plan = chaos_plan(count_sentences(original), seed)
    base_prompt = _build_prompt(original, plan, swaps, style_line, voice_hint, location,
                                story, story_note, casual_words)
    in_words = len(original.split())
    extra = 140 if (story or story_note.strip()) else 0   # room for the story
    tolerate = 3 if casual_words else 1                    # non-dictionary words allowed

    feedback = ""
    last_missing: list[str] = []
    best_ok: tuple[int, int, str] | None = None  # (odd words, choppy flag, text): facts intact
    for attempt in range(1, max_attempts + 1):
        try:
            out = (chat(base_prompt + feedback) or "").strip()
        except Exception as exc:  # network / rate limit: let the caller decide
            logger.error("rewrite attempt %d failed: %s", attempt, exc)
            if attempt == max_attempts:
                raise
            continue

        out_words = len(out.split())
        miss = missing_facts(facts, out)
        bad_len = not (0.6 * in_words <= out_words <= 1.5 * in_words + extra)
        weird = unknown_words(original, out, wordset) if out else []
        choppy = transition_heavy(out) if out else False
        if out and not miss and not bad_len:
            if len(weird) <= tolerate and not choppy:  # slang / coinages allowed in casual styles
                return RewriteResult(text=out, ok=True, attempts=attempt, swaps=swaps)
            if best_ok is None or (len(weird), int(choppy)) < best_ok[:2]:
                best_ok = (len(weird), int(choppy), out)

        last_missing = miss
        problems = []
        if miss:
            problems.append("you dropped or changed these, put them back exactly: " + "; ".join(miss[:15]))
        if bad_len:
            problems.append(f"keep the length close to the original (~{in_words} words{'; one short story may be added' if extra else ''})")
        if choppy:
            problems.append("too many sentences open with a one-word transition and a comma; "
                            "cut most of them and let the ideas follow each other")
        if len(weird) > tolerate:
            problems.append("these are not real English words, use normal words instead: " + ", ".join(weird[:10]))
        feedback = "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: " + " | ".join(problems)
        logger.info("rewrite attempt %d rejected: %s", attempt, problems)

    if best_ok is not None and best_ok[0] <= tolerate + 2:  # facts intact, a few odd words
        logger.warning("accepting rewrite with %d unrecognised words", best_ok[0])
        return RewriteResult(text=best_ok[2], ok=True, attempts=max_attempts, swaps=swaps)

    # Never hand back a rewrite that lost facts: return the original instead.
    return RewriteResult(text=original, ok=False, missing=last_missing,
                         attempts=max_attempts, swaps=swaps)
