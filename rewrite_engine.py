"""
rewrite_engine.py — DeAilize's rewrite core.

One model call sees the WHOLE post (so "it", "this method", etc. keep their
meaning), plus a rhythm plan built from a Lorenz attractor. The plan is a
soft guide, not a hard rule: the model decides the wording, the chaos only
decides the pacing (sentence lengths, where an And/But/So opener is allowed,
where a short aside is allowed).

After the call we verify that every number, link, quote and proper name from
the original is still present, and that no garbled or invented words slipped in
(checked against words_alpha.txt). If not, we retry with feedback, and if facts
are still lost we return the original text untouched rather than a corrupted rewrite.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("wordloom.engine")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------- chaos ----
def lorenz_sequence(n: int, seed: int | None = None, dt: float = 0.01,
                    steps_per_sample: int = 25) -> list[tuple[float, float, float]]:
    """Integrate the Lorenz system (sigma=10, rho=28, beta=8/3) and sample it.
    A tiny seed-dependent nudge to the start point gives a different,
    non-repeating trajectory per request."""
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


@dataclass
class RhythmPlan:
    lengths: list[int] = field(default_factory=list)   # target words per sentence
    openers: set[int] = field(default_factory=set)     # sentence #s that MAY open with And/But/So
    asides: set[int] = field(default_factory=set)      # sentence #s that MAY carry a dash aside


def rhythm_plan(n_sentences: int, seed: int | None = None,
                min_len: int = 4, max_len: int = 30,
                opener_share: float = 0.12, aside_share: float = 0.10) -> RhythmPlan:
    """x -> sentence length, y -> opener positions, z -> aside positions."""
    n = max(1, n_sentences)
    pts = lorenz_sequence(n, seed)

    lengths: list[int] = []
    for x, _, _ in pts:
        length = int(round(min_len + _norm(x, -20, 20) * (max_len - min_len)))
        # break up runs of similar lengths so the rhythm never goes flat
        if len(lengths) >= 2:
            window = lengths[-2:] + [length]
            if max(window) - min(window) < 4:
                length = lengths[-1] + 10 if lengths[-1] < 17 else max(min_len, lengths[-1] - 10)
        lengths.append(max(min_len, min(max_len, length)))

    def top_positions(axis: int, share: float, lo: float, hi: float) -> set[int]:
        k = max(1, round(n * share)) if n >= 4 else 0
        ranked = sorted(range(1, n), key=lambda i: _norm(pts[i][axis], lo, hi), reverse=True)
        chosen: set[int] = set()
        for i in ranked:
            if len(chosen) >= k:
                break
            if i - 1 in chosen or i + 1 in chosen:  # no two in a row
                continue
            chosen.add(i)
        return chosen

    return RhythmPlan(
        lengths=lengths,
        openers=top_positions(1, opener_share, -25, 25),
        asides=top_positions(2, aside_share, 0, 50),
    )


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
    file is missing or too small to trust, which disables the word guard."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            words = frozenset(w.strip().lower() for w in f if w.strip())
    except OSError:
        return None
    return words if len(words) > 50_000 else None


_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*")


def _known(word: str, wordset: "frozenset[str]") -> bool:
    """True if `word` (lowercase letters only) is a real word, allowing the
    regular endings the list may not spell out (plurals, -ed, -ing, -ly)."""
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


def unknown_words(original: str, rewritten: str, wordset: "frozenset[str] | None") -> list[str]:
    """New words in the rewrite that are not in the original and not in the word
    list. Catches garbled or invented words from small / flaky models. Proper
    nouns (capitalised mid-sentence) and anything already in the original are
    ignored."""
    if not wordset:
        return []
    seen_original = {t.lower() for t in _TOKEN.findall(original)}
    bad: list[str] = []
    for sentence in _SENT_SPLIT.split(rewritten):
        for i, tok in enumerate(_TOKEN.findall(sentence)):
            low = tok.lower()
            if low in seen_original:
                continue
            if i > 0 and tok[0].isupper():  # likely a proper noun
                continue
            parts = [p for p in re.split(r"['’\-]", low) if p]
            if all(_known(p, wordset) for p in parts):
                continue
            if low not in bad:
                bad.append(low)
    return bad


# --------------------------------------------------------------- helpers ----
def count_sentences(text: str) -> int:
    return len([s for s in _SENT_SPLIT.split(text.strip()) if s.strip()])


def suggest_swaps(text: str, flip_map: dict[str, list[str]], per_word: int = 3,
                  wordset: "frozenset[str] | None" = None) -> str:
    """Hints for the model, never blind replacements. Only plain 1-2 word
    options are offered, in the dictionary's order (most common first)."""
    lines = []
    lowered = text.lower()
    for word, options in flip_map.items():
        if re.search(rf"\b{re.escape(word)}\w*", lowered):
            plain = [o for o in options if len(o.split()) <= 2
                     and (not wordset or all(_known(w, wordset) for w in re.findall(r"[a-z]+", o.lower())))][:per_word]
            if plain:
                lines.append(f'- "{word}" -> {", ".join(plain)} (only if the meaning stays identical)')
    return "\n".join(lines[:25])


_CLICHES = (
    "delve, tapestry, moreover, furthermore, in conclusion, it's worth noting, "
    "in today's fast-paced world, game-changer, unlock the power of, navigate the "
    "landscape, seamlessly, holistic, paradigm"
)


def _build_prompt(text: str, plan: RhythmPlan, style_line: str, voice_hint: str,
                  location: str, swaps: str) -> str:
    lengths = ", ".join(str(n) for n in plan.lengths)
    openers = ", ".join(f"#{i + 1}" for i in sorted(plan.openers)) or "none"
    asides = ", ".join(f"#{i + 1}" for i in sorted(plan.asides)) or "none"
    parts = [
        "Rewrite the blog post below so it reads like a thoughtful person wrote it: "
        "specific, natural, with a varied rhythm. You can see the WHOLE post, so use "
        "that context to keep every reference clear.",
        "",
        "HARD RULES",
        "- Keep every fact, number, name, link and quote exactly as written. Add no new claims.",
        "- Keep the paragraph breaks. No headers, bullets or lists unless the original has them.",
        "- Meaning and flow come first. If a guide below would hurt clarity, ignore it.",
        "",
        "RHYTHM GUIDE (soft, follow in order, cycle if needed; words per sentence):",
        lengths,
        f"- Sentences {openers} may open with And, But or So, only where it reads naturally.",
        f"- Sentences {asides} may include a short dash aside, only where it adds something.",
        "- Mix long and short sentences. Never let three in a row be a similar length.",
        "",
        "WORD CHOICE",
        "- Prefer plain, concrete words and contractions where the tone allows.",
        f"- Avoid these clichés: {_CLICHES}.",
    ]
    if swaps:
        parts += ["- Optional swaps, use one only when it is a perfect fit:", swaps]
    if style_line:
        parts += ["", f"STYLE: {style_line}"]
    if voice_hint:
        parts += ["", f"WRITER VOICE: {voice_hint}"]
    if location:
        parts += ["", f"LOCATION: mention {location} only if it fits naturally. Never invent facts about it."]
    parts += ["", "Return ONLY the rewritten post.", "", "POST:", text]
    return "\n".join(parts)


# -------------------------------------------------------------- lexicon ----




def audit_flip_map(flip_map: dict[str, list[str]], wordset: "frozenset[str] | None") -> list[str]:
    """Swap options in the dictionary that contain a word the word list doesn't know."""
    if not wordset:
        return []
    problems = []
    for key, options in flip_map.items():
        for opt in options:
            for w in re.findall(r"[A-Za-z]+", opt):
                if not _known(w, wordset):
                    problems.append(f"{key} -> {opt!r} ({w!r} not a known word)")
    return problems


# Meaning-safe swaps only: same part of speech, same meaning in every context.
# Every form is spelled out so inflections can never come out wrong.
SAFE_FLIPS: dict[str, str] = {
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "utilise": "use", "utilises": "uses", "utilised": "used", "utilising": "using",
    "commence": "start", "commences": "starts", "commenced": "started", "commencing": "starting",
    "ascertain": "find out", "ascertains": "finds out", "ascertained": "found out",
    "ascertaining": "finding out",
    "delve into": "dig into", "delves into": "digs into", "delved into": "dug into",
    "delving into": "digging into",
    "in order to": "to", "prior to": "before", "subsequent to": "after",
    "due to the fact that": "because", "in the event that": "if",
    "at this point in time": "now", "a plethora of": "plenty of", "a myriad of": "many",
    "numerous": "many", "approximately": "about", "in addition to": "besides",
    "with regard to": "about", "a large number of": "many",
}
_SAFE_SORTED = sorted(SAFE_FLIPS, key=len, reverse=True)
_SAFE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _SAFE_SORTED) + r")\b", re.IGNORECASE)


def apply_safe_flips(text: str) -> str:
    """Deterministic backstop: swaps the stock AI phrases the model left behind."""
    def sub(m: re.Match) -> str:
        new = SAFE_FLIPS[m.group(1).lower()]
        return new[0].upper() + new[1:] if m.group(1)[0].isupper() else new
    return _SAFE_RE.sub(sub, text)


# ------------------------------------------------------------- main API ----
@dataclass
class RewriteResult:
    text: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    attempts: int = 0


def rewrite_post(text: str, chat: Callable[[str], str], *, style_line: str = "",
                 voice_hint: str = "", location: str = "",
                 flip_map: dict[str, list[str]] | None = None,
                 wordset: "frozenset[str] | None" = None,
                 max_attempts: int = 3, seed: int | None = None) -> RewriteResult:
    """Rewrite `text` in one model call, verify facts, retry on loss."""
    original = text.strip()
    if not original:
        return RewriteResult(text=text, ok=False)

    facts = extract_facts(original)
    plan = rhythm_plan(count_sentences(original), seed)
    swaps = suggest_swaps(original, flip_map or {}, wordset=wordset)
    base_prompt = _build_prompt(original, plan, style_line, voice_hint, location, swaps)
    in_words = len(original.split())

    feedback = ""
    last_missing: list[str] = []
    best_ok: tuple[int, str] | None = None  # (unknown-word count, text): facts intact, wording imperfect
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
        bad_len = not (0.6 * in_words <= out_words <= 1.5 * in_words)
        weird = unknown_words(original, out, wordset) if out else []
        if out and not miss and not bad_len:
            if len(weird) <= 1:  # tolerate one slang / brand word
                return RewriteResult(text=apply_safe_flips(out), ok=True, attempts=attempt)
            if best_ok is None or len(weird) < best_ok[0]:
                best_ok = (len(weird), out)

        last_missing = miss
        problems = []
        if miss:
            problems.append("you dropped or changed these, put them back exactly: " + "; ".join(miss[:15]))
        if bad_len:
            problems.append(f"keep the length close to the original (~{in_words} words)")
        if len(weird) > 1:
            problems.append("these are not real English words, use normal words instead: " + ", ".join(weird[:10]))
        feedback = "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: " + " | ".join(problems)
        logger.info("rewrite attempt %d rejected: %s", attempt, problems)

    if best_ok is not None and best_ok[0] <= 3:  # facts intact, a few odd words: better than the original
        logger.warning("accepting rewrite with %d unrecognised words", best_ok[0])
        return RewriteResult(text=apply_safe_flips(best_ok[1]), ok=True, attempts=max_attempts)

    # Never hand back a rewrite that lost facts: return the original instead.
    return RewriteResult(text=apply_safe_flips(original), ok=False, missing=last_missing, attempts=max_attempts)
