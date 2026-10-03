"""
rewrite_engine.py — DeAilize's rewrite core. One fused pipeline.

  1 STIFFNESS GATE  Code-only, runs before any model call. Scores every
                    sentence on measurable stiffness (lead-in openers, flat
                    runs of same-length sentences, formal-word density,
                    repeated openers, passive/dummy-subject frames). Sentences
                    over the line are RED; the rest are GREEN and never touched.
  2 CHAOS ENGINE    A Lorenz attractor sets a target length for each RED
                    sentence (length only, aware of its green neighbours).
  3 PERSONALITY     The model sees the whole section plus a persona/voice, so
                    it rewrites the red sentences to fit the flow around the
                    green ones. One mini-story can be woven in.
  4 WORD FLIPPER    The model proposes 30 ranked candidates for stiff words;
                    words_alpha.txt keeps only real words; we take rank ~5.
  5 JUDGE           Code checks facts, length, real words per sentence and
                    sends exact feedback back. A final model pass restructures
                    sentences that still read stiff (split, active voice,
                    subject first). Anything that can't be fixed safely keeps
                    its original text and is reported as still stiff.
  6 OUTPUT          Green + rewritten red, plus per-sentence status so the
                    site can show what changed and what still needs an edit.

The stiffness gate is a writing-style heuristic, not a detector, and says
nothing about how any particular AI-detection tool will score a text.
"""

import bisect
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("wordloom.engine")

_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*")


# ------------------------------------------------------------ sentences ----
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                  "e.g", "i.e", "u.s", "a.m", "p.m", "inc", "ltd", "no", "fig"}
_BOUNDARY = re.compile(r'[.!?]["\')\]”’]*\s+(?=["“‘(\[]?[A-Z0-9])')
_ENDS_WITH_ABBR = re.compile(r"(?:^|[\s(])([A-Za-z](?:\.[A-Za-z])*)\.$")


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking on Dr., e.g., U.S., or initials."""
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        candidate = text[start:m.end()].strip()
        stripped = candidate.rstrip("\"')]”’ ")
        hit = _ENDS_WITH_ABBR.search(stripped)
        if stripped.endswith(".") and hit:
            word = hit.group(1)
            if word.lower() in _ABBREVIATIONS or len(word) == 1:
                continue  # an abbreviation or an initial, not a sentence end
        out.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def word_count(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------- chaos engine ----
_SIGMA, _RHO, _BETA, _DT = 10.0, 28.0, 8.0 / 3.0, 0.01


def _lorenz_advance(state: list[float], steps: int) -> None:
    x, y, z = state
    for _ in range(steps):
        dx = _SIGMA * (y - x)
        dy = x * (_RHO - z) - y
        dz = x * y - _BETA * z
        x, y, z = x + dx * _DT, y + dy * _DT, z + dz * _DT
    state[0], state[1], state[2] = x, y, z


def _norm(v: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


_X_CDF: list[float] = []


def _x_cdf() -> list[float]:
    """Sorted x-values of a long Lorenz run. Turning x into its rank in this
    list gives a uniform 0..1 value no matter which lobe the trajectory is in,
    so the mix of short / medium / long sentences is controlled, not luck."""
    if not _X_CDF:
        state = [1.0, 1.0, 1.0]
        _lorenz_advance(state, 500)
        xs = []
        for _ in range(3000):
            _lorenz_advance(state, 25)
            xs.append(state[0])
        _X_CDF.extend(sorted(xs))
    return _X_CDF


MIN_LEN, MAX_LEN = 4, 31
MIN_STEP = 5  # neighbouring sentences always differ by at least this many words


class ChaosStream:
    """Endless, non-repeating stream of target sentence lengths.

    x (via its rank) picks the length band: ~22% short (4-9 words),
    ~40% medium (10-18), ~38% long (19-31). y adds jitter. The step rule keeps
    neighbouring sentences at least MIN_STEP words apart, so the rhythm can
    never go flat. Different seed, different trajectory."""

    def __init__(self, seed: int | None = None):
        import random
        rng = random.Random(seed)
        self._state = [1.0 + rng.uniform(-0.5, 0.5) for _ in range(3)]
        _lorenz_advance(self._state, 500)  # settle onto the attractor
        self._prev: int | None = None

    def next_length(self) -> int:
        _lorenz_advance(self._state, 25)
        x, y, _ = self._state
        cdf = _x_cdf()
        q = bisect.bisect_left(cdf, x) / len(cdf)
        if q < 0.22:
            length = MIN_LEN + int(q / 0.22 * 6)                    # 4..9
        elif q < 0.62:
            length = 10 + int((q - 0.22) / 0.40 * 9)                # 10..18
        else:
            length = 19 + int((q - 0.62) / 0.38 * (MAX_LEN - 18))   # 19..31
        if self._prev is not None and abs(length - self._prev) < MIN_STEP:
            jitter = int(_norm(y, -25, 25) * 5)
            if self._prev >= 16:
                length = max(MIN_LEN, self._prev - MIN_STEP - 1 - jitter)
            else:
                length = min(MAX_LEN, self._prev + MIN_STEP + 1 + jitter)
        self._prev = length
        return length

    def plan_paragraph(self, words: int) -> list[int]:
        """Sentence lengths for a paragraph of `words` words; they add up to
        the original word count, so nothing has to be cut or padded."""
        if words <= 9:
            return [max(1, words)]
        lengths: list[int] = []
        remaining = words
        while remaining >= MIN_LEN:
            n = self.next_length()
            if n > remaining + MIN_LEN:
                n = remaining
            lengths.append(n)
            remaining -= n
        if remaining > 0 and lengths:
            lengths[-1] += remaining
        return lengths


def burstiness(lengths: list[int]) -> float:
    """Spread of sentence lengths (std / mean). Flat prose is ~0.1-0.25."""
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(var) / mean if mean else 0.0


def _tolerance(target: int) -> int:
    return max(2, round(0.15 * target))


def check_lengths(plan: list[list[int]], paragraphs: list[str]) -> tuple[float, list[str]]:
    """Measure a rewrite against the chaos plan. Returns (share of sentences
    on target, exact problems to send back to the model)."""
    total = sum(len(p) for p in plan)
    if not total:
        return 1.0, []
    if len(paragraphs) != len(plan):
        return 0.0, [f"the section has {len(plan)} paragraphs, you wrote {len(paragraphs)}; "
                     "keep the same paragraphs"]
    ok = 0
    issues: list[str] = []
    for i, (targets, para) in enumerate(zip(plan, paragraphs), start=1):
        sentences = split_sentences(para)
        if len(sentences) != len(targets):
            issues.append(f"paragraph {i} must have exactly {len(targets)} sentences "
                          f"(word counts {', '.join(map(str, targets))}), you wrote {len(sentences)}")
            continue
        for j, (t, s) in enumerate(zip(targets, sentences), start=1):
            w = word_count(s)
            if abs(w - t) <= _tolerance(t):
                ok += 1
            else:
                issues.append(f"paragraph {i}, sentence {j} has {w} words, needs about {t}")
    return ok / total, issues


# ---------------------------------------------------------- fact guard ----
_URL = re.compile(r"https?://[^\s)\]]+")
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")
_QUOTE = re.compile(r"[\"“]([^\"”]{4,})[\"”]")
_NAME = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_NAME_STOP = {"The", "This", "That", "These", "Those", "And", "But", "So", "When",
              "While", "What", "Why", "How", "Here", "There", "They", "Then", "Also"}


def extract_facts(text: str, wordset: "frozenset[str] | None" = None) -> list[str]:
    """Things that must survive a rewrite. Capitalised words that are also used
    in lowercase in the text are not names, and a capitalised run of ordinary
    words ("Artificial Intelligence") may be abbreviated ("AI") but not dropped."""
    facts: list[str] = []
    facts += _URL.findall(text)
    facts += [m.strip().rstrip(".,") for m in _NUM.findall(text)]
    facts += _QUOTE.findall(text)
    term_words: set[str] = set()
    for sentence in split_sentences(text):
        words = sentence.split()
        tail = " ".join(words[1:])  # skip the first word: capitalised anyway
        if wordset:
            for m in _PHRASE.finditer(tail):
                parts = m.group().split()
                if all(w.lower() in wordset for w in parts):
                    facts.append("ANY:" + m.group() + "|" + "".join(w[0] for w in parts))
                    term_words.update(parts)
        for name in _NAME.findall(tail):
            if name in term_words or name in _NAME_STOP or name == "I":
                continue
            if wordset and name.lower() in wordset and re.search(rf"\b{name.lower()}\b", text):
                continue
            facts.append(name)
    seen: set[str] = set()
    return [f for f in facts if f and not (f in seen or seen.add(f))]


def missing_facts(facts: list[str], rewritten: str) -> list[str]:
    missing = []
    for f in facts:
        if f.startswith("ANY:"):  # any one of these spellings is fine
            options = f[4:].split("|")
            if not any(o in rewritten for o in options):
                missing.append(options[0])
        elif f not in rewritten:
            missing.append(f)
    return missing


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
    for sentence in split_sentences(rewritten):
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
    sentences = split_sentences(text)
    if len(sentences) < 5:
        return False
    hits = sum(1 for x in sentences if _OPENER.match(x.strip()))
    return hits >= min_count and hits / len(sentences) > max_share


def lead_ins(text: str) -> list[str]:
    """The one-word lead-ins ("Moreover", "However", ...) that open sentences."""
    found = []
    for x in split_sentences(text):
        m = _OPENER.match(x.strip())
        if m:
            found.append(m.group().strip().strip("\"'“(").rstrip(",").strip())
    return found


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
    n_targets = max(3, min(12, n_words // 35))
    prompt = (
        f"Read this whole section. Pick about {n_targets} words or short phrases that are the "
        "most formal, abstract or over-polished wording, where a more natural word would "
        "keep the exact meaning in THIS text. Skip names, numbers, quotes and technical "
        "terms the author clearly needs.\n"
        f"For each, give {n_candidates} replacement candidates that fit that exact sentence "
        "(same part of speech and same inflection as the original, meaning unchanged), "
        "ranked from 1 = plain everyday wording to "
        f"{n_candidates} = the wording an AI model would most likely write.\n"
        'Return ONLY JSON: {"words":[{"word":"<exact text>","candidates":["...", ...]}]}\n\n'
        f"TEXT:\n{text}"
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


# ------------------------------------------------------------- stiffness ----
RED_AT = 0.30
_DUMMY_SUBJECT = re.compile(r"^(?:It|There)(?:'s| is| are| was| were)\b")
_NOT_ONLY = re.compile(r"\bnot only\b.+\bbut\b", re.I)
_PASSIVE_BY = re.compile(r"\b(?:is|are|was|were|been|being)\s+\w+(?:ed|en)\s+by\b", re.I)


_TELLS: list[str] = []
_LIST_OF_THREE = re.compile(r"\b[\w'-]+, [\w'-]+,? (?:and|or) [\w'-]+\b", re.I)


def load_tells(path: str) -> list[str]:
    """Stock words/phrases the gate should flag, one per line in ai_tells.txt
    (lines starting with # are comments). Edit the file, no code needed."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")]
    except OSError:
        return []


def set_tells(tells: list[str]) -> None:
    _TELLS[:] = tells


def stiffness(sentences: list[str]) -> list[tuple[float, list[str]]]:
    """Score each sentence 0..1 for how stiff it reads, from measurable signals
    only (no model, no word list). Returns (score, reasons) per sentence."""
    n = len(sentences)
    lens = [word_count(s) for s in sentences]
    firsts = []
    for s in sentences:
        toks = _TOKEN.findall(s)
        firsts.append(toks[0].lower() if toks else "")
    out: list[tuple[float, list[str]]] = []
    for i, s in enumerate(sentences):
        score, why = 0.0, []
        text = s.strip()
        if _OPENER.match(text):
            score += 0.35
            why.append("opens with a lead-in word")
        near = [lens[j] for j in (i - 1, i + 1) if 0 <= j < n]
        if lens[i] >= 10 and near and all(abs(lens[i] - x) <= 3 for x in near):
            score += 0.25
            why.append("same length as its neighbours")
        if lens[i] > 30:
            score += 0.15
            why.append("very long")
        elif lens[i] >= 24:
            score += 0.10
            why.append("long")
        toks = _TOKEN.findall(s)
        if len(toks) >= 6 and sum(len(t) >= 10 for t in toks) / len(toks) >= 0.18:
            score += 0.20
            why.append("formal, heavy wording")
        if i > 0 and firsts[i] and firsts[i] == firsts[i - 1]:
            score += 0.20
            why.append("same opener as the previous sentence")
        if _DUMMY_SUBJECT.match(text):
            score += 0.15
            why.append("empty 'it is / there are' opener")
        if _NOT_ONLY.search(s):
            score += 0.15
            why.append("'not only... but' frame")
        if _PASSIVE_BY.search(s):
            score += 0.10
            why.append("passive voice")
        if s.count(",") >= 4:
            score += 0.10
            why.append("comma-heavy")
        if _LIST_OF_THREE.search(s):
            score += 0.10
            why.append("list of three")
        low = s.lower()
        hits = [t for t in _TELLS if re.search(rf"\b{re.escape(t)}", low)]
        if hits:
            score += min(0.5, 0.25 * len(hits))
            why.append("stock wording: " + ", ".join(hits[:3]))
        out.append((min(1.0, round(score, 2)), why))
    return out


# ------------------------------------------------------------- sentences ----
@dataclass
class _Sent:
    text: str                 # original sentence
    para: int                 # paragraph index (prose paragraphs only)
    score: float = 0.0
    why: list[str] = field(default_factory=list)
    red: bool = False
    target: int = 0           # chaos length for red sentences
    rid: int = 0              # id inside its section
    new: str | None = None    # accepted rewrite
    judged: bool = False

    @property
    def cur(self) -> str:
        return self.new if self.new is not None else self.text


_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SECTION_WORDS = 350


def _is_locked(block: str) -> bool:
    """Headings and lists are copied through untouched, never rewritten."""
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if not lines:
        return True
    if all(_LIST_LINE.match(ln) for ln in lines):
        return True
    if len(lines) == 1:
        line = lines[0].strip()
        if line.startswith("#"):
            return True
        if len(line.split()) <= 14 and not re.search(r"[.!?][\"')\]”]*$", line):
            return True
    return False


def _assign_targets(sents: list[_Sent], stream: ChaosStream) -> None:
    """Chaos lengths for the RED sentences only, nudged away from the length of
    the sentence just before them so the rhythm stays uneven around green text."""
    prev_len: int | None = None
    for s in sents:
        if not s.red:
            prev_len = word_count(s.text)
            continue
        src = word_count(s.text)
        t = stream.next_length()
        lo = max(MIN_LEN, math.ceil(0.6 * src))
        hi = max(lo, min(MAX_LEN + 6, round(1.6 * src)))
        if prev_len is not None and abs(t - prev_len) < MIN_STEP:
            t = prev_len + MIN_STEP + 1 if prev_len < 16 else prev_len - MIN_STEP - 1
        s.target = max(lo, min(hi, t))
        prev_len = s.target


def _sections(sents: list[_Sent]) -> list[list[_Sent]]:
    out: list[list[_Sent]] = []
    cur: list[_Sent] = []
    count = 0
    for s in sents:
        n = word_count(s.text)
        if cur and count + n > _SECTION_WORDS:
            out.append(cur)
            cur, count = [], 0
        cur.append(s)
        count += n
    if cur:
        out.append(cur)
    return out


def _display(section: list[_Sent], tagged: set[int]) -> str:
    """The section as running text. Sentences still to rewrite are tagged [R#]."""
    parts: list[str] = []
    prev_para = None
    for s in section:
        piece = f"[R{s.rid}] {s.cur}" if (s.red and s.rid in tagged) else s.cur
        if prev_para is not None:
            parts.append("\n\n" if s.para != prev_para else " ")
        parts.append(piece)
        prev_para = s.para
    return "".join(parts)


# ---------------------------------------------------------------- prompts ----
def _rewrite_prompt(section: list[_Sent], tasks: dict[int, str], swaps: list[Swap],
                    style_line: str, voice_hint: str, persona: str, location: str,
                    casual_words: bool, context: str, previous: str) -> str:
    by_rid = {s.rid: s for s in section if s.red}
    lines = []
    for rid in sorted(tasks):
        s = by_rid[rid]
        why = "; ".join(s.why[:3])
        fb = f"  FIX: {tasks[rid]}" if tasks[rid] else ""
        lines.append(f"R{rid}: about {s.target} words (it has {word_count(s.text)}). Stiff because: {why}.{fb}")
    parts = [
        "You are editing part of a blog post. Rewrite ONLY the sentences tagged [R#]. Leave every "
        "untagged sentence exactly as it is; it already reads naturally, so your rewrites must "
        "flow with the text before and after them.",
        "",
        "HARD RULES",
        "- Keep every fact, number, name, link and quote of the tagged sentence exactly. Add no claims.",
        "- Each rewrite is ONE sentence, about the target word count (within 2 words). Count your words.",
        "- Start the sentence with its subject where you can; avoid lead-in words like 'Moreover,' or 'However,'.",
        "- Use plain, concrete wording and contractions where the tone allows. No stock phrases.",
    ]
    if persona:
        parts += ["", f"WRITE AS: {persona}"]
    if style_line:
        parts += ["", f"STYLE: {style_line}"]
    if voice_hint:
        parts += ["", f"WRITER VOICE: {voice_hint}"]
    if location:
        parts += ["", f"LOCATION: mention {location} only if it fits naturally. Never invent facts about it."]
    if casual_words:
        parts += ["", "SPOKEN TOUCHES: in 1-3 of your rewrites, where the tone allows, use a casual, "
                      "spoken-style word you would not find in a formal dictionary (a y'all-style "
                      "contraction or a playful coinage). Vary them, and only where natural."]
    if swaps:
        parts += ["", "WORD SWAPS (use one only where the meaning stays exactly the same, "
                      "otherwise keep the original word):"]
        parts += [f'- "{w.word}" -> "{w.replacement}"' for w in swaps]
    if context:
        parts += ["", "THE POST OPENS WITH (context only):", context]
    if previous:
        parts += ["", "THE PREVIOUS SECTION ENDED WITH (context only):", previous]
    parts += ["", "TASKS", *lines, "",
              "Reply with ONLY one line per task, exactly like:", "R1: <the rewritten sentence>", "",
              "SECTION:", _display(section, set(tasks))]
    return "\n".join(parts)


def _judge_prompt(items: list[tuple[int, _Sent]], persona: str) -> str:
    lines = []
    for jid, s in items:
        lines.append(f"J{jid}: ORIGINAL: {s.text}\n    CURRENT: {s.cur}\n    STILL STIFF BECAUSE: "
                     f"{'; '.join(s.why[:3]) or 'rhythm'}\n    TOTAL LENGTH: about {s.target} words")
    return "\n".join([
        "These rewritten sentences still read stiff. Restructure each one completely: split it into "
        "two sentences, switch passive to active, or start with its subject. Keep every fact, number, "
        "name and link. Use plain wording. The total length of your version should be about the "
        "target. If the CURRENT version already reads naturally, repeat it unchanged."
        + (f" Write as: {persona}." if persona else ""),
        "", *lines, "",
        "Reply with ONLY one line per item, exactly like:", "J1: <the restructured text>",
    ])


_REPLY = re.compile(r"^\s*[*_`>-]*\s*([RJ])(\d+)\s*[*_`]*\s*[:.)\-]\s*(.+?)\s*$", re.M)


def _parse_replies(raw: str, kind: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for m in _REPLY.finditer(raw or ""):
        if m.group(1) == kind:
            out[int(m.group(2))] = m.group(3).strip().strip('"“”').strip()
    return out


def _tidy(reply: str, original: str) -> str:
    """Small models forget capitals and full stops: restore them from the original."""
    t = re.sub(r"^\[?[RJ]\d+\]?\s*[:.)\-]?\s*", "", " ".join(reply.split()))
    if t and original.lstrip()[:1].isupper() and t[:1].isalpha():
        t = t[0].upper() + t[1:]
    end = original.rstrip()[-1:]
    if t and end in ".!?" and t[-1] not in ".!?\"'”)":
        t += end
    return t


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _check_sentence(s: _Sent, new: str | None, wordset: "frozenset[str] | None",
                    tolerate: int, loose_len: bool = False) -> tuple[bool, bool, list[str], int]:
    """Judge one rewrite. Returns (safe, on_length, problems, odd_words)."""
    if not new:
        return False, False, ["no reply for this sentence"], 0
    problems: list[str] = []
    miss = missing_facts(extract_facts(s.text, wordset), new)
    if miss:
        problems.append("put these back exactly: " + "; ".join(miss[:8]))
    if _norm_text(new) == _norm_text(s.text):
        problems.append("it is unchanged, rewrite it")
    weird = unknown_words(s.text, new, wordset)
    if len(weird) > tolerate:
        problems.append("not real English words, use normal words: " + ", ".join(weird[:6]))
    w = word_count(new)
    tol = max(3, round(0.3 * s.target)) if loose_len else _tolerance(s.target)
    on_len = abs(w - s.target) <= tol
    if not on_len:
        problems.append(f"it has {w} words, it needs about {s.target}")
    safe = not miss and _norm_text(new) != _norm_text(s.text) and len(weird) <= tolerate + 1
    return safe, on_len, problems, len(weird)


# ------------------------------------------------------------- main API ----
@dataclass
class RewriteResult:
    text: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    attempts: int = 0
    swaps: list[Swap] = field(default_factory=list)
    parts_kept_original: int = 0   # red sentences that could not be rewritten safely
    on_target: float = 0.0         # share of rewritten sentences on their chaos length
    burstiness: float = 0.0        # spread of sentence lengths in the final text
    n_sentences: int = 0
    n_red: int = 0                 # flagged stiff before rewriting
    n_rewritten: int = 0
    n_still_stiff: int = 0         # still stiff after everything
    story_added: bool = False
    segments: list[dict] = field(default_factory=list)  # [{"text","status"}], status: kept|rewritten|stiff|locked|story|break


def _process_section(section: list[_Sent], chat: Callable[[str], str], *, wordset, target_rank,
                     style_line, voice_hint, persona, location, casual_words, context, previous,
                     max_attempts: int, word_pass: bool = True) -> tuple[int, list[Swap], list[str]]:
    """Personality + word flipper + sentence judge for one section."""
    reds = [s for s in section if s.red]
    for i, s in enumerate(reds, start=1):
        s.rid = i
    red_text = " ".join(s.text for s in reds)
    swaps = (plan_swaps(red_text, chat, wordset, extract_facts(red_text, wordset), target_rank)
             if word_pass else [])
    tolerate = 2 if casual_words else 1
    pending = {s.rid: s for s in reds}
    feedback: dict[int, str] = {}
    best: dict[int, tuple[tuple[int, int], str]] = {}
    used = 0
    failures: list[str] = []

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        prompt = _rewrite_prompt(section, {rid: feedback.get(rid, "") for rid in pending}, swaps,
                                 style_line, voice_hint, persona, location if attempt == 1 else "",
                                 casual_words, context, previous)
        used += 1
        try:
            replies = _parse_replies(chat(prompt) or "", "R")
        except Exception as exc:
            logger.error("rewrite attempt %d failed: %s", attempt, exc)
            if attempt == max_attempts:
                raise
            continue
        for rid, s in list(pending.items()):
            reply = _tidy(replies[rid], s.text) if rid in replies else None
            safe, on_len, problems, odd = _check_sentence(s, reply, wordset, tolerate)
            if safe:
                rank = (0 if on_len else 1, odd)
                if rid not in best or rank < best[rid][0]:
                    best[rid] = (rank, reply)
                if on_len and odd <= tolerate:
                    s.new = reply
                    del pending[rid]
                    continue
            feedback[rid] = "; ".join(problems)
        if pending:
            logger.info("section attempt %d: %d sentence(s) still pending", attempt, len(pending))

    for rid, s in pending.items():  # not perfect, but safe: better than the stiff original
        if rid in best:
            s.new = best[rid][1]
        else:
            failures.append(f"R{rid}: {feedback.get(rid, 'no usable rewrite')}")
    return used, swaps, failures


def _judge_pass(section: list[_Sent], all_sents: list[_Sent], chat: Callable[[str], str],
                wordset, persona: str, tolerate: int) -> int:
    """Final judge: restructure rewritten sentences that still read stiff."""
    scores = stiffness([s.cur for s in all_sents])
    index = {id(s): i for i, s in enumerate(all_sents)}
    stubborn = [s for s in section if s.new is not None and scores[index[id(s)]][0] >= RED_AT]
    if not stubborn:
        return 0
    for s in stubborn:
        s.why = scores[index[id(s)]][1]
    items = list(enumerate(stubborn, start=1))
    try:
        replies = _parse_replies(chat(_judge_prompt(items, persona)) or "", "J")
    except Exception as exc:
        logger.warning("judge pass skipped: %s", exc)
        return 1
    before = {id(s): scores[index[id(s)]][0] for s in stubborn}
    old = {id(s): s.new for s in stubborn}
    for jid, s in items:
        new = _tidy(replies[jid], s.text) if jid in replies else None
        safe, _, _, _ = _check_sentence(s, new, wordset, tolerate, loose_len=True)
        if safe and new:
            s.new = new
    after = stiffness([s.cur for s in all_sents])
    for s in stubborn:
        if after[index[id(s)]][0] >= before[id(s)]:   # no better: keep the earlier rewrite
            s.new = old[id(s)]
    return 1


def _make_story(chat: Callable[[str], str], opening: str, middle: str, story_note: str,
                wordset, persona: str, casual_words: bool) -> str | None:
    """One short story for the middle of the post. A real anecdote from the
    writer is used if given; otherwise an illustrative example that doesn't
    claim to be real. Returns None if no clean version comes back."""
    if story_note.strip():
        task = ("Write 2-4 sentences that work this real anecdote from the writer into the post, "
                f"using only the details given and adding no facts: {story_note.strip()}")
    else:
        task = ("Write a short illustrative mini-story (2-4 sentences, about 60-90 words) that brings "
                "a point from the post to life and gives it a personal feel. Frame it as an example "
                "(\"Picture someone who...\", \"Say you're...\"), NOT as the writer's real experience. "
                "Do not invent real people, places, numbers or statistics.")
    prompt = "\n".join([
        task + (f" Write as: {persona}." if persona else ""),
        "It must follow naturally from the paragraph below and read in the same voice.",
        "", "THE POST OPENS WITH:", opening, "", "THE PARAGRAPH IT FOLLOWS:", middle, "",
        "Reply with ONLY the story paragraph.",
    ])
    tolerate = 3 if casual_words else 1
    for _ in range(2):
        try:
            text = (chat(prompt) or "").strip().strip('"“”')
        except Exception as exc:
            logger.warning("story skipped: %s", exc)
            return None
        words = len(text.split())
        no_numbers = story_note.strip() or not (re.search(r"\d", text) or _URL.search(text))
        if 25 <= words <= 140 and no_numbers and "\n\n" not in text \
                and len(unknown_words(opening + " " + middle, text, wordset)) <= tolerate:
            return text
    return None


def rewrite_post(text: str, chat: Callable[[str], str], *, style_line: str = "",
                 voice_hint: str = "", persona: str = "", location: str = "",
                 wordset: "frozenset[str] | None" = None, target_rank: int = 5,
                 story: bool = False, story_note: str = "", casual_words: bool = False,
                 max_attempts: int = 3, seed: int | None = None, word_pass: bool = True,
                 progress: Callable[[str], None] | None = None) -> RewriteResult:
    """Run the full flow: gate -> chaos -> personality -> word flipper -> judge."""
    original = text.strip()
    if not original:
        return RewriteResult(text=text, ok=False)

    # ---- 1. split into locked blocks and prose sentences, score stiffness
    units: list[tuple[str, object]] = []
    all_sents: list[_Sent] = []
    para_no = 0
    for block in re.split(r"\n\s*\n", original):
        if not block.strip():
            continue
        if _is_locked(block):
            units.append(("locked", block.strip()))
            continue
        sents = [_Sent(t, para_no) for t in split_sentences(block)]
        para_no += 1
        units.append(("para", sents))
        all_sents += sents
    for s, (score, why) in zip(all_sents, stiffness([s.text for s in all_sents])):
        s.score, s.why, s.red = score, why, score >= RED_AT
    n_red = sum(s.red for s in all_sents)

    attempts = 0
    swaps_all: list[Swap] = []
    failures: list[str] = []
    story_added = False
    story_text: str | None = None
    story_after_unit = -1

    if n_red:
        # ---- 2. chaos lengths for the red sentences
        _assign_targets(all_sents, ChaosStream(seed))
        sections = [sec for sec in _sections(all_sents) if any(s.red for s in sec)]
        opening = " ".join(original.split()[:80])
        previous = ""
        for k, section in enumerate(sections, start=1):
            if progress:
                progress(f"Rewriting part {k} of {len(sections)}…")
            # ---- 3-5. personality, word flipper, sentence judge
            used, swaps, fails = _process_section(
                section, chat, wordset=wordset, target_rank=target_rank, style_line=style_line,
                voice_hint=voice_hint, persona=persona, location=location if k == 1 else "",
                casual_words=casual_words, context=opening if k > 1 else "", previous=previous,
                max_attempts=max_attempts, word_pass=word_pass)
            attempts += used
            swaps_all += swaps
            failures += fails
            attempts += _judge_pass(section, all_sents, chat, wordset, persona,
                                    2 if casual_words else 1)
            previous = " ".join(" ".join(s.cur for s in section).split()[-40:])

        # ---- the story, in the middle of the post
        para_units = [i for i, (kind, _) in enumerate(units) if kind == "para"]
        if (story or story_note.strip()) and para_units:
            mid = para_units[max(0, len(para_units) // 2 - 1)] if len(para_units) > 1 else para_units[0]
            mid_text = " ".join(s.cur for s in units[mid][1])  # type: ignore[union-attr]
            if progress:
                progress("Writing the story…")
            story_text = _make_story(chat, opening, mid_text, story_note, wordset, persona, casual_words)
            story_after_unit = mid
            story_added = bool(story_text)

    # ---- 6. output with per-sentence status
    final_scores = stiffness([s.cur for s in all_sents])
    status: dict[int, str] = {}
    for s, (sc, _) in zip(all_sents, final_scores):
        if s.new is not None:
            status[id(s)] = "stiff" if sc >= RED_AT else "rewritten"
        else:
            status[id(s)] = "stiff" if s.red else "kept"

    segments: list[dict] = []
    out_blocks: list[str] = []
    for i, (kind, payload) in enumerate(units):
        if segments:
            segments.append({"text": "\n\n", "status": "break"})
        if kind == "locked":
            out_blocks.append(str(payload))
            segments.append({"text": str(payload), "status": "locked"})
        else:
            sents: list[_Sent] = payload  # type: ignore[assignment]
            out_blocks.append(" ".join(s.cur for s in sents))
            for j, s in enumerate(sents):
                segments.append({"text": s.cur + (" " if j < len(sents) - 1 else ""),
                                 "status": status[id(s)]})
        if story_text and i == story_after_unit:
            out_blocks.append(story_text)
            segments.append({"text": "\n\n", "status": "break"})
            segments.append({"text": story_text, "status": "story"})

    final = "\n\n".join(out_blocks)
    rewritten = [s for s in all_sents if s.new is not None]
    hit = [abs(word_count(s.new or "") - s.target) <= _tolerance(s.target) for s in rewritten]
    return RewriteResult(
        text=final, ok=not failures, missing=failures, attempts=attempts, swaps=swaps_all,
        parts_kept_original=len(failures),
        on_target=(sum(hit) / len(hit)) if hit else 0.0,
        burstiness=burstiness([word_count(s) for s in split_sentences(final)]),
        n_sentences=len(all_sents), n_red=n_red, n_rewritten=len(rewritten),
        n_still_stiff=sum(1 for v in status.values() if v == "stiff"),
        story_added=story_added, segments=segments)
