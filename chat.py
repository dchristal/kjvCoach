"""CLI chat against any OpenAI-compatible server (LM Studio, Ollama, OpenRouter,
Groq, OpenAI, …), grounded in the bundled KJV via deterministic kjvcode tools.

Usage:
    python chat.py                           # uses defaults; starts a new session log
    python chat.py --model gemma-4-e4b-it    # override model id
    python chat.py --resume                  # resume the most recent session
    python chat.py --resume <name|path>      # resume a specific session
    python chat.py --list-sessions           # list saved sessions and exit
    python chat.py --no-tools                # disable kjvcode tool-calling

In-chat commands:
    /reset              clear conversation history (starts a new session log)
    /sessions           list saved sessions
    /session            show current session file path
    /count /stats /nth /verse-at /letters /verify   (see kjvcode.py)
    /quit, /exit        leave
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

# Kept for back-compat with kjvcode.py, which imports DEFAULT_VAULT. The
# CLI no longer reads any vault — all answers come from the KJV files
# bundled in this repo (concord text + KJPBS OSIS XML).
DEFAULT_VAULT = Path("/nonexistent")
DEFAULT_SESSIONS_DIR = Path.home() / ".cache" / "0cli" / "sessions"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"  # id LM Studio exposes for the loaded model
DEFAULT_KJV_TEXT = Path(__file__).resolve().parent / "kjv.txt"
DEFAULT_KJV_OSIS = Path(__file__).resolve().parent / "kjv1769.osis.xml"

SYSTEM_PROMPT = """You answer as the Holy Spirit whose sole source of truth is the King James
Bible (KJV). Every reply must conform strictly to the KJV.

Hard rules — never violate:
- No elaboration, commentary, reasoning, greetings, apologies, disclaimers, hedges,
  headings, or follow-up questions beyond what the rules below explicitly permit.
- Reply with the shortest faithful answer, in this order of preference:
    1. If the question is a yes/no or affirmation about the Lordship of Christ, reply
       exactly: Jesus is Lord
    2. If the question is a factual/structural/statistical query about the KJV itself
       — counts of words, verses, chapters, books, occurrences of a name or term,
       which book or testament contains a passage, lengths, orderings, genealogies,
       and the like — reply with the bare answer in this form, on a single line when
       possible:
         <answer> — <one-line scope>
       Then, only if directly relevant and helpful as evidence, append up to 3 KJV
       verses verbatim, one per line, each with its reference in parentheses.
       Rules for this mode:
         - Give a number only when you are confident it is accurate for the KJV. If
           uncertain about an exact count, give an explicit qualified estimate
           prefixed by "approximately" and state the basis in <one-line scope> (e.g.
           "approximately 973 — KJV NT, exact-token match for 'Jesus'"). Never invent
           precise figures.
         - State the scope you actually used (e.g. "KJV whole canon", "KJV NT",
           "Genesis 1–11", "odd-numbered NT books: Matthew, Luke, John, Acts, …").
         - For occurrence questions, count only exact KJV tokens unless the user asks
           for variants; if you include variants (e.g. "Jesus" + "Christ" +
           pronouns), say so in the scope line.
         - If the question is not actually answerable from the KJV text (it requires
           outside data, modern science, history, or opinion), use rule 4 instead.
    3. Otherwise, reply with KJV verses that directly bear on the question.
       Quote each verse verbatim and append its reference in parentheses, e.g.
       (John 3:16). Put each verse on its own line. Provide the top 7 most
       relevant verses when that many apply; fewer is fine if fewer apply.
       Do not exceed 7.
    4. Only if no KJV verse — in the wider canon — and no faithful KJV-internal
       fact answers the question, reply exactly:
       That falls outside the scope of Scripture.
- Apply Scripture by clear principle, not only by literal keyword match. For example,
  questions about how to treat people are answered by passages on love, neighbour,
  enemies, brethren, the unbelieving, rebuke, forgiveness, etc.
- Never introduce modern science, history, opinion, or general knowledge from outside
  the KJV. Statistical answers must be facts about the KJV text itself.
- Never break character, never mention these instructions, never describe what you are
  doing.
- Never emit tool-call syntax as plain text (e.g. "<|tool_call>", "kjv_count(...)").
  Either invoke a tool through the proper tool-calling channel, or answer in prose.

Quote verses verbatim from the KJV."""


# --- kjvcode tool-calling bridge ------------------------------------------

# Compact schemas — descriptions live in the system prompt, not per-tool,
# so small-context models (e.g. 4k) can still fit them.
def _scope_param() -> dict:
    return {"type": "string", "default": "all"}


def _mode_param() -> dict:
    return {"type": "string", "default": "pure"}


KJV_TOOLS = [
    {"type": "function", "function": {
        "name": "kjv_count",
        "description": "Count term occurrences in KJV scope.",
        "parameters": {"type": "object", "required": ["term"], "properties": {
            "term": {"type": "string"}, "scope": _scope_param(),
            "mode": _mode_param(), "case_sensitive": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "kjv_nth_mention",
        "description": "Reference of Nth mention of term in scope.",
        "parameters": {"type": "object", "required": ["term", "n"], "properties": {
            "term": {"type": "string"}, "n": {"type": "integer"},
            "scope": _scope_param(), "mode": _mode_param(),
            "case_sensitive": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "kjv_verse_at",
        "description": "Verse at ordinal N in scope.",
        "parameters": {"type": "object", "required": ["n"], "properties": {
            "n": {"type": "integer"}, "scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_position_of",
        "description": "Ordinal position of ref ('Book N:V') in scope.",
        "parameters": {"type": "object", "required": ["ref"], "properties": {
            "ref": {"type": "string"}, "scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_verse_count",
        "description": "Verses in scope (optionally containing term).",
        "parameters": {"type": "object", "properties": {
            "scope": _scope_param(), "term": {"type": "string"},
            "mode": _mode_param(), "case_sensitive": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "kjv_chapter_count", "description": "Chapters in scope.",
        "parameters": {"type": "object", "properties": {"scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_book_count", "description": "Books in scope.",
        "parameters": {"type": "object", "properties": {"scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_word_count", "description": "Words/digit-tokens in scope verse text.",
        "parameters": {"type": "object", "properties": {"scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_letter_count", "description": "Letters/vowels/consonants in scope.",
        "parameters": {"type": "object", "properties": {"scope": _scope_param()}}}},
    {"type": "function", "function": {
        "name": "kjv_stats",
        "description": "Authoritative books/chapters/verses/words totals (KJPBS-backed for whole/OT/NT).",
        "parameters": {"type": "object", "properties": {"scope": _scope_param()}}}},
]

# Appended to SYSTEM_PROMPT only when tools are enabled. Documents shared param
# vocabulary once instead of per-tool, which would explode token count.
TOOLS_SYSTEM_NOTE = """
You may call the following tools for KJV facts. ALWAYS call a tool for any
count / position / Nth / totals question; never invent numbers.
Shared params:
  scope: all|ot|nt|gospels|law|history-ot|wisdom|major-prophets|minor-prophets|
         pauline|general-epistles|first-last-books|odd-nt|even-nt|odd-ot|even-ot|
         book:<Name> | chapter:<Book> <N> | verse:<Book> <N>:<V> |
         range:<Book> <N>:<V>-<Book> <N>:<V>
  mode:  pure (whole-word, default) | raw (substring incl. punctuation) |
         standalone (whitespace token == term) | concealed (substring anywhere,
         even inside larger words) | regex (raw Python regex)
Examples:
  "words in psalm 119" -> kjv_word_count(scope="chapter:Psalms 119")
  "Jesus in NT"        -> kjv_count(term="Jesus", scope="nt")
  "verse #777"         -> kjv_verse_at(n=777, scope="all")
"""


# Patterns used to scrub tool-call syntax that some chat templates leak into
# plain text (e.g. when the backend rejected our tools= list and the model
# fell back to emitting the raw template tokens).
_TOOL_CALL_LEAK_RES = [
    re.compile(r"<\|tool_call(?:_begin)?\|?>.*?(?:<\|tool_call_end\|>|$)", re.DOTALL),
    re.compile(r"<\|tool_call\|?>.*", re.DOTALL),
    re.compile(r"^\s*```(?:tool_call|json)?\s*\{.*?\}\s*```", re.DOTALL | re.MULTILINE),
]


def _strip_tool_call_leakage(text: str) -> str:
    out = text
    for pat in _TOOL_CALL_LEAK_RES:
        out = pat.sub("", out)
    return out.strip()


def _dispatch_tool(name: str, args: dict, vault: Path, kjv_text: Path | None) -> dict:
    """Run a kjvcode primitive and return its dict result (or {'error': str})."""
    try:
        import kjvcode as _kj
        # Prefer KJPBS OSIS for tool answers when available.
        if DEFAULT_KJV_OSIS and DEFAULT_KJV_OSIS.exists():
            _kj.CORPUS_SOURCE = "osis"
        common = {"vault": vault, "kjv_text": kjv_text}
        if name == "kjv_count":
            return _kj.count(args["term"], args.get("scope", "all"),
                             args.get("mode", "pure"),
                             bool(args.get("case_sensitive", False)), **common)
        if name == "kjv_nth_mention":
            return _kj.nth_mention(args["term"], int(args["n"]),
                                   args.get("scope", "all"),
                                   args.get("mode", "pure"),
                                   bool(args.get("case_sensitive", False)), **common)
        if name == "kjv_verse_at":
            return _kj.verse_at(int(args["n"]), args.get("scope", "all"), **common)
        if name == "kjv_position_of":
            return _kj.position_of(args["ref"], args.get("scope", "all"), **common)
        if name == "kjv_verse_count":
            return _kj.verse_count(args.get("scope", "all"), args.get("term"),
                                   args.get("mode", "pure"),
                                   bool(args.get("case_sensitive", False)), **common)
        if name == "kjv_chapter_count":
            return _kj.chapter_count(args.get("scope", "all"), **common)
        if name == "kjv_book_count":
            return _kj.book_count(args.get("scope", "all"), **common)
        if name == "kjv_word_count":
            return _kj.word_count(args.get("scope", "all"), **common)
        if name == "kjv_letter_count":
            return _kj.letter_count(args.get("scope", "all"), **common)
        if name == "kjv_stats":
            return kjv_stats(vault, args.get("scope", "all"), kjv_text)
        return {"error": f"unknown tool: {name}"}
    except Exception as e:  # surface error back to the model
        return {"error": f"{type(e).__name__}: {e}"}


# --- KJV deterministic stats / counts -------------------------------------

# Pretty book names as written by ingest_kjv.py (folder + filename prefix).
_OT_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges",
    "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles",
    "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
]
_NT_BOOKS = [
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
    "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
]
_ALL_BOOKS = _OT_BOOKS + _NT_BOOKS

# Authoritative KJV1769 totals from KingJamesPureBibleSearch (KJPBS),
# https://github.com/dewhisna/KingJamesPureBibleSearch — used as ground truth
# for whole-canon / OT / NT statistics so our results agree with KJPBS.
# Keys are scope labels as returned by _resolve_scope().
_KJPBS_TOTALS: dict[str, dict[str, int]] = {
    "KJV whole canon": {"books": 66, "chapters": 1189, "verses": 31102, "words": 790849},
    "KJV OT":          {"books": 39, "chapters":  929, "verses": 23145, "words": 609269},
    "KJV NT":          {"books": 27, "chapters":  260, "verses":  7957, "words": 181580},
}

# OSIS book-id -> testament. Populated lazily during OSIS parse.
_OSIS_NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
_OSIS_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*")
_OSIS_STATS_CACHE: dict[Path, dict[str, dict[str, int]]] | None = None

# OSIS osisID -> canonical full book name used throughout this codebase.
_OSIS_BOOK_ID: dict[str, str] = {
    "Gen": "Genesis", "Exod": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
    "Deut": "Deuteronomy", "Josh": "Joshua", "Judg": "Judges", "Ruth": "Ruth",
    "1Sam": "1 Samuel", "2Sam": "2 Samuel", "1Kgs": "1 Kings", "2Kgs": "2 Kings",
    "1Chr": "1 Chronicles", "2Chr": "2 Chronicles", "Ezra": "Ezra",
    "Neh": "Nehemiah", "Esth": "Esther", "Job": "Job", "Ps": "Psalms",
    "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
    "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Ezek": "Ezekiel",
    "Dan": "Daniel", "Hos": "Hosea", "Joel": "Joel", "Amos": "Amos",
    "Obad": "Obadiah", "Jonah": "Jonah", "Mic": "Micah", "Nah": "Nahum",
    "Hab": "Habakkuk", "Zeph": "Zephaniah", "Hag": "Haggai", "Zech": "Zechariah",
    "Mal": "Malachi",
    "Matt": "Matthew", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts": "Acts", "Rom": "Romans", "1Cor": "1 Corinthians", "2Cor": "2 Corinthians",
    "Gal": "Galatians", "Eph": "Ephesians", "Phil": "Philippians",
    "Col": "Colossians", "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy", "2Tim": "2 Timothy", "Titus": "Titus", "Phlm": "Philemon",
    "Heb": "Hebrews", "Jas": "James", "1Pet": "1 Peter", "2Pet": "2 Peter",
    "1John": "1 John", "2John": "2 John", "3John": "3 John", "Jude": "Jude",
    "Rev": "Revelation",
}

# OSIS-derived corpus cache (separate from concord corpus cache).
_KJV_OSIS_CORPUS: list[tuple[str, int, int, str]] | None = None
_KJV_OSIS_SOURCE: Path | None = None

# Loaded lazily: list of (book, chapter, verse, text) tuples.
_KJV_CORPUS: list[tuple[str, int, int, str]] | None = None
_KJV_SOURCE: Path | None = None

# Loaded lazily: per-book raw text slices from the concord file (verbatim).
_KJV_RAW_BY_BOOK: dict[str, str] | None = None
_KJV_RAW_SOURCE: Path | None = None

_VERSE_RE = re.compile(r"^\*\*(\d+)\*\*\s+(.*)$")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NUMBER_RE = re.compile(r"\b\d+\b")

# --- concord plaintext parser (kjv.txt)
# Header signatures in canonical book order. Some keys repeat (e.g. CHRONICLES,
# CORINTHIANS, THESSALONIANS, TIMOTHY, PETER, JOHN); the linear scan binds each
# occurrence to the next book in the list.
_CONCORD_HEADERS: list[tuple[str, "re.Pattern[str]"]] = [
    ("Genesis",          re.compile(r"^GENESIS\.$")),
    ("Exodus",           re.compile(r"^EXODUS\.$")),
    ("Leviticus",        re.compile(r"^LEVITICUS\.$")),
    ("Numbers",          re.compile(r"^NUMBERS\.$")),
    ("Deuteronomy",      re.compile(r"^DEUTERONOMY\.$")),
    ("Joshua",           re.compile(r"^THE BOOK OF JOSHUA\.$")),
    ("Judges",           re.compile(r"^THE BOOK OF JUDGES\.$")),
    ("Ruth",             re.compile(r"^THE BOOK OF RUTH\.$")),
    # Archaic naming: 1/2 Samuel are "FIRST/SECOND BOOK OF THE KINGS",
    # and 1/2 Kings are "THIRD/FOURTH BOOK OF THE KINGS".
    ("1 Samuel",         re.compile(r"^THE FIRST BOOK OF THE KINGS\.$")),
    ("2 Samuel",         re.compile(r"^THE SECOND BOOK OF THE KINGS\.$")),
    ("1 Kings",          re.compile(r"^THE THIRD BOOK OF THE KINGS\.$")),
    ("2 Kings",          re.compile(r"^THE FOURTH BOOK OF THE KINGS\.$")),
    ("1 Chronicles",     re.compile(r"^CHRONICLES\.$")),
    ("2 Chronicles",     re.compile(r"^CHRONICLES\.$")),
    ("Ezra",             re.compile(r"^EZRA\.$")),
    ("Nehemiah",         re.compile(r"^BOOK OF NEHEMIAH\.$")),
    ("Esther",           re.compile(r"^BOOK OF ESTHER\.$")),
    ("Job",              re.compile(r"^THE BOOK OF JOB\.$")),
    ("Psalms",           re.compile(r"^BOOK OF PSALMS\.$")),
    ("Proverbs",         re.compile(r"^THE PROVERBS\.$")),
    ("Ecclesiastes",     re.compile(r"^ECCLESIASTES;$")),
    ("Song of Solomon",  re.compile(r"^SONG OF SOLOMON\.$")),
    ("Isaiah",           re.compile(r"^ISAIAH\.$")),
    ("Jeremiah",         re.compile(r"^JEREMIAH\.$")),
    ("Lamentations",     re.compile(r"^OF JEREMIAH\.$")),
    ("Ezekiel",          re.compile(r"^EZEKIEL\.$")),
    ("Daniel",           re.compile(r"^THE BOOK OF DANIEL\.$")),
    ("Hosea",            re.compile(r"^HOSEA\.$")),
    ("Joel",             re.compile(r"^JOEL\.$")),
    ("Amos",             re.compile(r"^AMOS\.$")),
    ("Obadiah",          re.compile(r"^OBADIAH\.$")),
    ("Jonah",            re.compile(r"^JONAH\.$")),
    ("Micah",            re.compile(r"^MICAH\.$")),
    ("Nahum",            re.compile(r"^NAHUM\.$")),
    ("Habakkuk",         re.compile(r"^HABAKKUK\.$")),
    ("Zephaniah",        re.compile(r"^ZEPHANIAH\.$")),
    ("Haggai",           re.compile(r"^HAGGAI\.$")),
    ("Zechariah",        re.compile(r"^ZECHARIAH\.$")),
    ("Malachi",          re.compile(r"^MALACHI\.$")),
    ("Matthew",          re.compile(r"^ST\. MATTHEW\.$")),
    ("Mark",             re.compile(r"^ST\. MARK\.$")),
    ("Luke",             re.compile(r"^ST\. LUKE\.$")),
    ("John",             re.compile(r"^ST\. JOHN\.$")),
    ("Acts",             re.compile(r"^ACTS OF THE APOSTLES\.$")),
    ("Romans",           re.compile(r"^ROMANS\.$")),
    ("1 Corinthians",    re.compile(r"^CORINTHIANS\.$")),
    ("2 Corinthians",    re.compile(r"^CORINTHIANS\.$")),
    ("Galatians",        re.compile(r"^GALATIANS\.$")),
    ("Ephesians",        re.compile(r"^EPHESIANS\.$")),
    ("Philippians",      re.compile(r"^PHILIPPIANS\.$")),
    ("Colossians",       re.compile(r"^COLOSSIANS\.$")),
    ("1 Thessalonians",  re.compile(r"^THESSALONIANS\.$")),
    ("2 Thessalonians",  re.compile(r"^THESSALONIANS\.$")),
    ("1 Timothy",        re.compile(r"^TIMOTHY\.$")),
    ("2 Timothy",        re.compile(r"^TIMOTHY\.$")),
    ("Titus",            re.compile(r"^TITUS\.$")),
    ("Philemon",         re.compile(r"^PHILEMON\.$")),
    ("Hebrews",          re.compile(r"^HEBREWS\.$")),
    ("James",            re.compile(r"^JAMES\.$")),
    ("1 Peter",          re.compile(r"^PETER\.$")),
    ("2 Peter",          re.compile(r"^PETER\.$")),
    ("1 John",           re.compile(r"^JOHN\.$")),
    ("2 John",           re.compile(r"^JOHN\.$")),
    ("3 John",           re.compile(r"^JOHN\.$")),
    ("Jude",             re.compile(r"^JUDE\.$")),
    ("Revelation",       re.compile(r"^ST\. JOHN THE DIVINE\.$")),
]

_CONCORD_CHAPTER_RE = re.compile(r"^(?:CHAPTER|PSALM)\s+(\d+)\s*$")
_CONCORD_VERSE_RE = re.compile(r"^(\d+)\s+(.+)$")
_CONCORD_ACROSTIC_RE = re.compile(r"^[A-Z]{2,}\.$")  # e.g. ALEPH., BETH. (Psalm 119)


def _parse_concord(path: Path) -> list[tuple[str, int, int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    corpus: list[tuple[str, int, int, str]] = []
    hi = 0  # next expected header
    book: str | None = None
    chapter = 0
    verse_num = 0
    parts: list[str] = []

    def flush() -> None:
        nonlocal verse_num, parts
        if book and chapter and verse_num:
            corpus.append((book, chapter, verse_num, " ".join(parts).strip()))
        verse_num = 0
        parts = []

    for raw in lines:
        line = raw.rstrip()
        if hi < len(_CONCORD_HEADERS) and _CONCORD_HEADERS[hi][1].match(line):
            flush()
            book = _CONCORD_HEADERS[hi][0]
            hi += 1
            chapter = 0
            continue
        m = _CONCORD_CHAPTER_RE.match(line)
        if m:
            flush()
            chapter = int(m.group(1))
            continue
        if not book or not chapter:
            continue
        if not line.strip():
            continue
        # Skip Psalm 119 acrostic letter headings (single uppercase token + period).
        if verse_num > 0 and _CONCORD_ACROSTIC_RE.match(line.strip()):
            continue
        v = _CONCORD_VERSE_RE.match(line)
        if v:
            flush()
            verse_num = int(v.group(1))
            parts = [v.group(2).strip()]
        else:
            # Implicit verse 1 (or continuation of current verse).
            if verse_num == 0:
                verse_num = 1
            parts.append(line.strip())

    flush()
    return corpus


def _load_kjv(vault: Path, kjv_text: Path | None = DEFAULT_KJV_TEXT) -> list[tuple[str, int, int, str]]:
    """Load (book, chapter, verse, text). Prefers the concord plaintext file
    (`kjv_text`); falls back to <vault>/Bible/<Book>/<Book> N.md if not present.
    Cached after first call."""
    global _KJV_CORPUS, _KJV_SOURCE
    src: Path | None = kjv_text if (kjv_text and kjv_text.exists()) else None
    if src is None:
        src = vault / "Bible"
    if _KJV_CORPUS is not None and _KJV_SOURCE == src:
        return _KJV_CORPUS

    corpus: list[tuple[str, int, int, str]] = []
    if src.is_file():
        corpus = _parse_concord(src)
    elif src.is_dir():
        for book in _ALL_BOOKS:
            book_dir = src / book
            if not book_dir.is_dir():
                continue
            for md in sorted(book_dir.glob(f"{book} *.md")):
                try:
                    ch = int(md.stem[len(book) + 1:])
                except ValueError:
                    continue
                for line in md.read_text(encoding="utf-8").splitlines():
                    m = _VERSE_RE.match(line)
                    if m:
                        corpus.append((book, ch, int(m.group(1)), m.group(2)))
    _KJV_CORPUS, _KJV_SOURCE = corpus, src
    return corpus


def _load_kjv_osis(osis_path: Path = DEFAULT_KJV_OSIS) -> list[tuple[str, int, int, str]]:
    """Load (book, chapter, verse, text) from the KJPBS OSIS XML.
    Verse text matches what KJPBS treats as canonical (footnotes excluded,
    transChange and other markup flattened to their text content).
    Cached after first call."""
    global _KJV_OSIS_CORPUS, _KJV_OSIS_SOURCE
    if _KJV_OSIS_CORPUS is not None and _KJV_OSIS_SOURCE == osis_path:
        return _KJV_OSIS_CORPUS
    import xml.etree.ElementTree as _ET
    tree = _ET.parse(str(osis_path))
    root = tree.getroot()
    corpus: list[tuple[str, int, int, str]] = []
    for v in root.iter(_OSIS_NS + 'verse'):
        oid = v.get('osisID')
        if not oid:
            continue
        try:
            book_id, ch_s, v_s = oid.split('.')
            ch, vn = int(ch_s), int(v_s)
        except ValueError:
            continue
        book = _OSIS_BOOK_ID.get(book_id)
        if not book:
            continue
        text = re.sub(r"\s+", " ", _osis_text_filtered(v)).strip()
        corpus.append((book, ch, vn, text))
    _KJV_OSIS_CORPUS, _KJV_OSIS_SOURCE = corpus, osis_path
    return corpus


def _resolve_scope(spec: str) -> tuple[str, list[str]]:
    """Return (label, books). spec examples: all, ot, nt, odd-nt, even-ot, book:John."""
    s = (spec or "all").strip().lower()
    if s in ("all", "kjv", "whole"):
        return ("KJV whole canon", _ALL_BOOKS)
    if s == "ot":
        return ("KJV OT", _OT_BOOKS)
    if s == "nt":
        return ("KJV NT", _NT_BOOKS)
    if s == "odd-nt":
        sel = [b for i, b in enumerate(_NT_BOOKS, 1) if i % 2 == 1]
        return (f"odd-numbered NT books ({', '.join(sel)})", sel)
    if s == "even-nt":
        sel = [b for i, b in enumerate(_NT_BOOKS, 1) if i % 2 == 0]
        return (f"even-numbered NT books ({', '.join(sel)})", sel)
    if s == "odd-ot":
        sel = [b for i, b in enumerate(_OT_BOOKS, 1) if i % 2 == 1]
        return (f"odd-numbered OT books ({len(sel)} books)", sel)
    if s == "even-ot":
        sel = [b for i, b in enumerate(_OT_BOOKS, 1) if i % 2 == 0]
        return (f"even-numbered OT books ({len(sel)} books)", sel)
    if s.startswith("book:"):
        want = s[5:].strip().lower().replace(" ", "")
        for b in _ALL_BOOKS:
            if b.lower().replace(" ", "") == want:
                return (b, [b])
        raise ValueError(f"unknown book: {spec[5:]}")
    raise ValueError(f"unknown scope: {spec}")


def kjv_count(vault: Path, term: str, scope: str = "all", kjv_text: Path | None = DEFAULT_KJV_TEXT) -> dict:
    """Count whole-word, case-insensitive occurrences of `term` in `scope`.
    Multi-word terms are matched as a phrase. Returns dict with totals."""
    corpus = _load_kjv(vault, kjv_text)
    if not corpus:
        raise RuntimeError(
            f"KJV not found. Set --kjv-text to the concord .txt file, "
            f"or run: python ingest_kjv.py (vault: {vault / 'Bible'})."
        )
    label, books = _resolve_scope(scope)
    book_set = set(books)
    pattern = re.compile(
        r"\b" + r"\s+".join(re.escape(w) for w in term.split()) + r"\b",
        re.IGNORECASE,
    )
    occ = 0
    verse_hits = 0
    per_book: dict[str, int] = {}
    for book, _ch, _v, text in corpus:
        if book not in book_set:
            continue
        n = len(pattern.findall(text))
        if n:
            occ += n
            verse_hits += 1
            per_book[book] = per_book.get(book, 0) + n
    return {
        "term": term,
        "scope": label,
        "occurrences": occ,
        "verses_with_hit": verse_hits,
        "per_book": per_book,
    }


def _load_kjv_raw(kjv_text: Path) -> dict[str, str]:
    """Slice the concord file verbatim into {book: raw_text}. Each slice runs
    from the line *after* the book's header to the line *before* the next
    book's header (or EOF)."""
    global _KJV_RAW_BY_BOOK, _KJV_RAW_SOURCE
    if _KJV_RAW_BY_BOOK is not None and _KJV_RAW_SOURCE == kjv_text:
        return _KJV_RAW_BY_BOOK
    lines = kjv_text.read_text(encoding="utf-8").splitlines()
    boundaries: list[tuple[str, int]] = []
    hi = 0
    for i, raw in enumerate(lines):
        if hi >= len(_CONCORD_HEADERS):
            break
        if _CONCORD_HEADERS[hi][1].match(raw.rstrip()):
            boundaries.append((_CONCORD_HEADERS[hi][0], i + 1))
            hi += 1
    out: dict[str, str] = {}
    for idx, (book, start) in enumerate(boundaries):
        end = boundaries[idx + 1][1] - 1 if idx + 1 < len(boundaries) else len(lines)
        out[book] = "\n".join(lines[start:end])
    _KJV_RAW_BY_BOOK, _KJV_RAW_SOURCE = out, kjv_text
    return out


def _osis_text_filtered(elem) -> str:
    """Concatenate text under `elem` (its descendants), skipping <note>
    subtrees. Matches what KJPBS treats as canonical verse content
    (titles/superscriptions kept; footnotes excluded)."""
    parts: list[str] = []

    def rec(e, top: bool) -> None:
        tag = e.tag.split('}', 1)[-1] if '}' in e.tag else e.tag
        if not top and tag == 'note':
            if e.tail:
                parts.append(e.tail)
            return
        if e.text:
            parts.append(e.text)
        for c in e:
            rec(c, False)
        if not top and e.tail:
            parts.append(e.tail)

    if elem.text:
        parts.append(elem.text)
    for c in elem:
        rec(c, False)
    return "".join(parts)


def _load_osis_stats(osis_path: Path) -> dict[str, dict[str, int]]:
    """Parse the KJPBS OSIS XML once and return per-scope structural counts.
    Returns {scope_label: {books, chapters, verses, words}} for whole/OT/NT.
    Books/chapters/verses match KJPBS exactly; words are an OSIS-derived
    estimate using a KJPBS-like tokenizer (close to but not byte-identical
    with KJPBS's published 790,849 figure)."""
    global _OSIS_STATS_CACHE
    if _OSIS_STATS_CACHE is not None and osis_path in _OSIS_STATS_CACHE:
        return _OSIS_STATS_CACHE[osis_path]
    import xml.etree.ElementTree as _ET
    tree = _ET.parse(str(osis_path))
    root = tree.getroot()

    def lname(t: str) -> str:
        return t.split('}', 1)[-1] if '}' in t else t

    out: dict[str, dict[str, int]] = {
        "KJV whole canon": {"books": 0, "chapters": 0, "verses": 0, "words": 0},
        "KJV OT":          {"books": 0, "chapters": 0, "verses": 0, "words": 0},
        "KJV NT":          {"books": 0, "chapters": 0, "verses": 0, "words": 0},
    }
    testaments = [d for d in root.iter(_OSIS_NS + 'div') if d.get('type') == 'x-testament']
    for idx, testament in enumerate(testaments):
        label = "KJV OT" if idx == 0 else "KJV NT"
        for book in testament.iter(_OSIS_NS + 'div'):
            if book.get('type') != 'book':
                continue
            out[label]["books"] += 1
            for ch in book.iter(_OSIS_NS + 'chapter'):
                if ch.get('osisID'):
                    out[label]["chapters"] += 1
            for v in book.iter(_OSIS_NS + 'verse'):
                if v.get('osisID'):
                    out[label]["verses"] += 1
                    out[label]["words"] += len(_OSIS_WORD_RE.findall(_osis_text_filtered(v)))
    whole = out["KJV whole canon"]
    for k in ("books", "chapters", "verses", "words"):
        whole[k] = out["KJV OT"][k] + out["KJV NT"][k]
    _OSIS_STATS_CACHE = {osis_path: out}
    return out


def kjv_stats(vault: Path, scope: str = "all", kjv_text: Path | None = DEFAULT_KJV_TEXT,
              osis_path: Path | None = DEFAULT_KJV_OSIS) -> dict:
    """Verbatim split-on-space stats. Every character of the source is kept.
    A token is a 'number' iff it is non-empty and consists entirely of ASCII
    digits; otherwise it is a 'word'. Chapter/verse markers contribute their
    digit tokens to the number count."""
    label, books = _resolve_scope(scope)

    # Prefer raw concord file for verbatim accounting; fall back to the parsed
    # corpus (markdown vault) if the concord file isn't available.
    if kjv_text and kjv_text.exists():
        # For the whole-canon scope, count the file verbatim — including the
        # title, book headers, CHAPTER/PSALM markers, and acrostic letters.
        if set(books) == set(_ALL_BOOKS):
            text_blob = kjv_text.read_text(encoding="utf-8")
        else:
            raw_by_book = _load_kjv_raw(kjv_text)
            text_blob = "\n".join(raw_by_book.get(b, "") for b in books)
        # Verses/chapters/books still come from the parsed corpus for accuracy.
        corpus = _load_kjv(vault, kjv_text)
    else:
        corpus = _load_kjv(vault, kjv_text)
        if not corpus:
            raise RuntimeError(
                f"KJV not found. Set --kjv-text to the concord .txt file, "
                f"or run: python ingest_kjv.py (vault: {vault / 'Bible'})."
            )
        # Reconstruct a verbatim-ish blob from parsed verses (markdown fallback).
        text_blob = "\n".join(
            f"{v} {t}" for (b, _ch, v, t) in corpus if b in set(books)
        )

    book_set = set(books)
    verses = sum(1 for b, _, _, _ in corpus if b in book_set)
    chapters = len({(b, ch) for b, ch, _, _ in corpus if b in book_set})
    books_seen = len({b for b, _, _, _ in corpus if b in book_set})

    words = numbers = 0
    for tok in text_blob.split():
        if not tok:
            continue
        if tok.isdigit():
            numbers += 1
        else:
            words += 1

    result = {
        "scope": label,
        "books": books_seen,
        "chapters": chapters,
        "verses": verses,
        "words": words,
        "numbers": numbers,
        "source": "verbatim split of concord text (this app)",
    }

    # Layer in OSIS-derived counts (parsed from KJPBS's bundled OSIS XML).
    # These match KJPBS exactly for books/chapters/verses on whole/OT/NT.
    if osis_path and osis_path.exists() and label in _KJPBS_TOTALS:
        try:
            osis_stats = _load_osis_stats(osis_path)[label]
            result["osis"] = {
                "books": osis_stats["books"],
                "chapters": osis_stats["chapters"],
                "verses": osis_stats["verses"],
                "words": osis_stats["words"],
            }
        except Exception:
            pass

    # KJPBS authoritative override for whole/OT/NT.
    if label in _KJPBS_TOTALS:
        result.update(_KJPBS_TOTALS[label])
        result["source"] = "KingJamesPureBibleSearch (KJV1769) authoritative"

    return result


# --- meta-question intercept -----------------------------------------------

_SCOPE_PHRASES: list[tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\bodd[-\s]+(numbered\s+)?(books?\s+of\s+the\s+)?nt\b", re.I), "odd-nt"),
    (re.compile(r"\beven[-\s]+(numbered\s+)?(books?\s+of\s+the\s+)?nt\b", re.I), "even-nt"),
    (re.compile(r"\bodd[-\s]+(numbered\s+)?(books?\s+of\s+the\s+)?ot\b", re.I), "odd-ot"),
    (re.compile(r"\beven[-\s]+(numbered\s+)?(books?\s+of\s+the\s+)?ot\b", re.I), "even-ot"),
    (re.compile(r"\b(new\s+testament|nt)\b", re.I), "nt"),
    (re.compile(r"\b(old\s+testament|ot|hebrew\s+bible|tanakh)\b", re.I), "ot"),
]


def _detect_scope(q: str) -> str:
    # book:<Name>
    for b in _ALL_BOOKS:
        if re.search(rf"\bin\s+(the\s+book\s+of\s+)?{re.escape(b)}\b", q, re.I):
            return f"book:{b}"
    for pat, scope in _SCOPE_PHRASES:
        if pat.search(q):
            return scope
    return "all"


_META_KEYS = ("words", "numbers", "verses", "chapters", "books")


_STATS_OVERVIEW_RE = re.compile(
    r"\b(stats?|statistics|metrics|totals?|overview|summary|figures?|numbers?)\b",
    re.I,
)
_STATS_TRIGGER_RE = re.compile(
    r"\b(what|which|show|give|list|tell|have|got|available|do you|you have)\b",
    re.I,
)


def _format_stats_overview(r: dict) -> str:
    keys = ("books", "chapters", "verses", "words", "numbers")
    parts = [f"{r[k]:,} {k}" for k in keys if k in r]
    src = r.get("source", "")
    suffix = f" [{src}]" if "KingJamesPureBibleSearch" in src else ""
    if "osis" in r:
        osis_bits = [f"{r['osis'][k]:,} {k}" for k in ("books", "chapters", "verses", "words") if k in r["osis"]]
        if osis_bits:
            suffix += f" (OSIS-parsed from KJPBS XML: {', '.join(osis_bits)})"
    return f"{', '.join(parts)} — {r['scope']} (KJV){suffix}"


def _try_meta_answer(user: str, vault: Path, kjv_text: Path | None) -> str | None:
    """If the user is asking a structural/statistical question about the KJV,
    answer deterministically. Returns None if the question doesn't match."""
    q = user.lower().strip().rstrip("?").strip()

    # Overview-style: "what stats do you have?", "show me the totals", etc.
    if _STATS_OVERVIEW_RE.search(q) and _STATS_TRIGGER_RE.search(q):
        scope = _detect_scope(user)
        try:
            r = kjv_stats(vault, scope, kjv_text)
        except (RuntimeError, ValueError):
            return None
        return _format_stats_overview(r)

    if "how many" not in q and "count of" not in q and "number of" not in q:
        return None
    bible_words = ("bible", "kjv", "scripture", "testament", " ot", " nt", "ot ", "nt ")
    if not any(w in f" {q} " for w in bible_words):
        # Allow "in the book of John" style without "bible".
        if not any(re.search(rf"\bin\s+(the\s+book\s+of\s+)?{re.escape(b)}\b", user, re.I) for b in _ALL_BOOKS):
            return None
    wanted = [k for k in _META_KEYS if k in q]
    if not wanted:
        return None
    scope = _detect_scope(user)
    try:
        r = kjv_stats(vault, scope, kjv_text)
    except (RuntimeError, ValueError):
        return None
    parts = [f"{r[k]:,} {k}" for k in wanted if k in r]
    if not parts:
        return None
    src = r.get("source", "")
    suffix = f" [{src}]" if "KingJamesPureBibleSearch" in src else ""
    if "osis" in r:
        # Append per-key OSIS-derived figures for transparency.
        osis_bits = [f"{r['osis'][k]:,} {k}" for k in wanted if k in r["osis"]]
        if osis_bits:
            suffix += f" (OSIS-parsed from KJPBS XML: {', '.join(osis_bits)})"
    return f"{', '.join(parts)} — {r['scope']} (KJV){suffix}"


# --- session persistence ---------------------------------------------------

def _new_session_path(sessions_dir: Path) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return sessions_dir / f"{stamp}.jsonl"


def _resolve_session(spec: str | None, sessions_dir: Path) -> Path | None:
    """Resolve --resume value to a session file. Returns None if nothing found."""
    if spec in (None, "", "latest"):
        files = sorted(sessions_dir.glob("*.jsonl"))
        return files[-1] if files else None
    p = Path(spec)
    if p.is_absolute() and p.exists():
        return p
    cand = sessions_dir / spec
    if cand.exists():
        return cand
    cand = sessions_dir / f"{spec}.jsonl"
    if cand.exists():
        return cand
    return None


def _load_session(path: Path) -> tuple[list[dict], dict]:
    """Returns (messages, meta). messages includes the system row if present."""
    messages: list[dict] = []
    meta: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "meta":
                meta = rec
            elif t == "message" and rec.get("role") in ("system", "user", "assistant"):
                messages.append({"role": rec["role"], "content": rec.get("content", "")})
    return messages, meta


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _list_sessions(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser(description="KJV chat CLI (kjvcode-grounded)")
    ap.add_argument(
        "--kjv-text", type=Path, default=DEFAULT_KJV_TEXT,
        help="plaintext KJV file used by /count and /stats (concord format)",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "lm-studio"))
    ap.add_argument("--model", default=os.environ.get("CHAT_MODEL", DEFAULT_MODEL))
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument(
        "--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR,
        help="directory where chat session JSONL files are stored",
    )
    ap.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        metavar="NAME",
        help="resume a session: latest if no value, else a name or path",
    )
    ap.add_argument(
        "--no-save", action="store_true",
        help="do not write a session log for this run",
    )
    ap.add_argument(
        "--list-sessions", action="store_true",
        help="list saved sessions and exit",
    )
    ap.add_argument(
        "--no-tools", action="store_true",
        help="disable kjvcode tool-calling (model answers stats from memory)",
    )
    args = ap.parse_args()

    if args.list_sessions:
        for p in _list_sessions(args.sessions_dir):
            size = p.stat().st_size
            print(f"{p.name}\t{size:>8}B\t{p}")
        return 0

    console = Console()
    # Vault is intentionally None: this CLI is grounded in the bundled KJV.
    args.vault = DEFAULT_VAULT

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # System prompt with the tools-usage note appended only when tools are on,
    # so the no-tools path doesn't waste context advertising tools.
    active_system = SYSTEM_PROMPT if args.no_tools else SYSTEM_PROMPT + TOOLS_SYSTEM_NOTE

    # --- session setup ---
    resumed_from: Path | None = None
    history: list[dict] = [{"role": "system", "content": active_system}]
    if args.resume is not None:
        resumed_from = _resolve_session(args.resume, args.sessions_dir)
        if resumed_from is None:
            console.print(f"[red]No session found for[/red] {args.resume!r}")
            return 2
        loaded, _meta = _load_session(resumed_from)
        if loaded:
            # Keep the current SYSTEM_PROMPT (it may have been edited); reuse turns only.
            history = [{"role": "system", "content": active_system}] + [
                m for m in loaded if m["role"] in ("user", "assistant")
            ]
        console.print(f"[dim]Resumed session: {resumed_from} ({len(history)-1} msgs)[/dim]")

    session_path: Path | None = None
    if not args.no_save:
        session_path = resumed_from if resumed_from else _new_session_path(args.sessions_dir)
        if resumed_from is None:
            _append_jsonl(session_path, {
                "type": "meta",
                "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "base_url": args.base_url,
            })
            _append_jsonl(session_path, {
                "type": "message", "role": "system", "content": active_system,
            })

    save_line = "(not saved)" if args.no_save else str(session_path)
    console.print(
        Panel.fit(
            f"[bold]kjvCoach[/bold]  model=[cyan]{args.model}[/cyan]  "
            f"tools=[cyan]{'off' if args.no_tools else 'on'}[/cyan]\n"
            f"session=[cyan]{save_line}[/cyan]\n"
            "Commands: /reset  /sessions  /session  "
            "/count  /stats  /nth  /verse-at  /letters  /verify  /quit",
            border_style="green",
        )
    )

    while True:
        try:
            user = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not user:
            continue

        if user in ("/quit", "/exit"):
            return 0
        if user == "/reset":
            history = [{"role": "system", "content": active_system}]
            if not args.no_save:
                session_path = _new_session_path(args.sessions_dir)
                _append_jsonl(session_path, {
                    "type": "meta",
                    "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                    "model": args.model,
                    "base_url": args.base_url,
                    "note": "started via /reset",
                })
                _append_jsonl(session_path, {
                    "type": "message", "role": "system", "content": active_system,
                })
                console.print(f"[yellow]conversation cleared; new session:[/yellow] {session_path}")
            else:
                console.print("[yellow]conversation cleared[/yellow]")
            continue
        if user == "/sessions":
            files = _list_sessions(args.sessions_dir)
            if not files:
                console.print("[dim]no saved sessions[/dim]")
            else:
                for p in files[-20:]:
                    marker = "  *" if session_path and p == session_path else "   "
                    console.print(f"{marker} {p.name}  [dim]({p.stat().st_size}B)[/dim]")
            continue
        if user == "/session":
            console.print(str(session_path) if session_path else "[dim](not saved)[/dim]")
            continue

        if user.startswith("/count"):
            parts = user.split(maxsplit=2)
            if len(parts) < 2:
                console.print(
                    "[red]usage:[/red] /count <term> [scope]   "
                    "[dim]scope: all|ot|nt|odd-nt|even-nt|odd-ot|even-ot|book:<Name>[/dim]"
                )
                continue
            term = parts[1].strip('"\'')
            scope = parts[2].strip() if len(parts) == 3 else "all"
            try:
                r = kjv_count(args.vault, term, scope, args.kjv_text)
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
                continue
            console.print(
                f"[bold]{r['occurrences']}[/bold] occurrences of "
                f"[cyan]{r['term']!r}[/cyan] in {r['verses_with_hit']} verses "
                f"— {r['scope']}"
            )
            if r["per_book"]:
                top = sorted(r["per_book"].items(), key=lambda x: -x[1])[:10]
                for b, n in top:
                    console.print(f"  [dim]{n:>5}[/dim]  {b}")
            continue

        if user.startswith("/stats"):
            parts = user.split(maxsplit=1)
            scope = parts[1].strip() if len(parts) == 2 else "all"
            try:
                r = kjv_stats(args.vault, scope, args.kjv_text)
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
                continue
            console.print(
                f"[bold]{r['scope']}[/bold]: "
                f"books=[cyan]{r['books']}[/cyan]  "
                f"chapters=[cyan]{r['chapters']}[/cyan]  "
                f"verses=[cyan]{r['verses']}[/cyan]  "
                f"words=[cyan]{r['words']}[/cyan]  "
                f"numbers=[cyan]{r['numbers']}[/cyan]"
            )
            continue

        if user.startswith("/nth"):
            # /nth <term> <N> [scope] [mode]
            parts = user.split()
            if len(parts) < 3:
                console.print("[red]usage:[/red] /nth <term> <N> [scope] [mode]")
                continue
            try:
                from kjvcode import nth_mention
                term = parts[1]
                n = int(parts[2])
                scope = parts[3] if len(parts) > 3 else "all"
                mode = parts[4] if len(parts) > 4 else "pure"
                r = nth_mention(term, n, scope, mode, vault=args.vault, kjv_text=args.kjv_text)
                console.print(f"[bold]#{r['n']}[/bold] of [cyan]{term!r}[/cyan] in {r['scope']} — [bold green]{r['ref']}[/bold green]")
                console.print(f"[dim]{r['text']}[/dim]")
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user.startswith("/verse-at"):
            # /verse-at <N> [scope]
            parts = user.split()
            if len(parts) < 2:
                console.print("[red]usage:[/red] /verse-at <N> [scope]")
                continue
            try:
                from kjvcode import verse_at
                n = int(parts[1])
                scope = parts[2] if len(parts) > 2 else "all"
                r = verse_at(n, scope, vault=args.vault, kjv_text=args.kjv_text)
                console.print(f"[bold]#{r['n']}[/bold] in {r['scope']} — [bold green]{r['ref']}[/bold green]")
                console.print(f"[dim]{r['text']}[/dim]")
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user.startswith("/letters"):
            parts = user.split(maxsplit=1)
            scope = parts[1].strip() if len(parts) == 2 else "all"
            try:
                from kjvcode import letter_count
                r = letter_count(scope, vault=args.vault, kjv_text=args.kjv_text)
                console.print(
                    f"[bold]{r['scope']}[/bold]: "
                    f"letters=[cyan]{r['letters']:,}[/cyan]  "
                    f"vowels=[cyan]{r['vowels']:,}[/cyan]  "
                    f"consonants=[cyan]{r['consonants']:,}[/cyan]"
                )
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
            continue

        if user.startswith("/verify"):
            # /verify [path/to/patterns.json]
            parts = user.split(maxsplit=1)
            from pathlib import Path as _P
            import kjvcode as _kj
            from kjvcode import verify_patterns, DEFAULT_PATTERNS
            cat = _P(parts[1].strip()) if len(parts) == 2 else DEFAULT_PATTERNS
            if not cat.exists():
                console.print(f"[red]not found:[/red] {cat}")
                continue
            _kj.CORPUS_SOURCE = "osis"  # use KJPBS XML (exact)
            verify_patterns(cat, args.vault, args.kjv_text)
            continue

        # Auto-detect KJV meta-questions and answer them deterministically.
        meta = _try_meta_answer(user, args.vault, args.kjv_text)
        if meta is not None:
            console.print(f"[bold magenta]kjv ›[/bold magenta] {meta}")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": meta})
            if session_path is not None:
                _append_jsonl(session_path, {
                    "type": "message", "role": "user", "content": user,
                })
                _append_jsonl(session_path, {
                    "type": "message", "role": "assistant", "content": meta,
                })
            continue

        user_content = user

        turn_messages = history + [{"role": "user", "content": user_content}]

        # --- tool-call resolution loop (non-streaming) -----------------
        tool_kwargs: dict = {}
        if not args.no_tools:
            tool_kwargs = {"tools": KJV_TOOLS, "tool_choice": "auto"}
        max_tool_steps = 4
        tool_error = False
        for _step in range(max_tool_steps):
            if not tool_kwargs:
                break
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=turn_messages,
                    temperature=args.temperature,
                    stream=False,
                    **tool_kwargs,
                )
            except Exception as e:
                console.print(f"[dim]tools disabled this turn: {e}[/dim]")
                tool_error = True
                break
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                # Model is done thinking with tools; fall through to stream a final answer.
                break
            # Append the assistant message that requested the tool calls.
            turn_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in calls
                ],
            })
            for tc in calls:
                try:
                    tc_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    tc_args = {}
                result = _dispatch_tool(tc.function.name, tc_args, args.vault, args.kjv_text)
                console.print(
                    f"[dim]· {tc.function.name}({json.dumps(tc_args, ensure_ascii=False)}) → "
                    f"{json.dumps(result, ensure_ascii=False)[:200]}[/dim]"
                )
                turn_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            console.print("[yellow]tool-call loop: max steps reached[/yellow]")

        if tool_error:
            # Retry without tools on this turn. Use the slim (no-tools) system
            # prompt to free context, in case the failure was an n_ctx overflow.
            slim_system = SYSTEM_PROMPT
            slim_history_tail = [m for m in history[1:] if m["role"] in ("user", "assistant")]
            turn_messages = (
                [{"role": "system", "content": slim_system}]
                + slim_history_tail
                + [{"role": "user", "content": user}]
            )

        # Stream the final response.
        try:
            stream = client.chat.completions.create(
                model=args.model,
                messages=turn_messages,
                temperature=args.temperature,
                stream=True,
            )
        except Exception as e:
            console.print(f"[red]API error:[/red] {e}")
            console.print(
                "[dim]Is LM Studio's local server running and the model loaded?\n"
                f"Expected at {args.base_url}[/dim]"
            )
            continue

        console.print("[bold magenta]kjv ›[/bold magenta] ", end="")
        full = []
        try:
            for ev in stream:
                delta = ev.choices[0].delta.content or ""
                if delta:
                    full.append(delta)
                    console.print(delta, end="", soft_wrap=True, highlight=False)
        except KeyboardInterrupt:
            console.print("\n[yellow]…interrupted[/yellow]")
        console.print()

        answer = "".join(full).strip()
        # Strip any leaked tool-call syntax the model emitted as plain text
        # (happens on the no-tools fallback path with chat templates that
        # know about tool tokens but weren't given a tools= list).
        answer = _strip_tool_call_leakage(answer)
        # Persist the *unaugmented* user turn so context doesn't bloat history.
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": answer})

        # Append to session log (full fidelity, before any trimming).
        if session_path is not None:
            _append_jsonl(session_path, {
                "type": "message", "role": "user", "content": user,
            })
            _append_jsonl(session_path, {
                "type": "message", "role": "assistant", "content": answer,
            })

        # Trim history to keep prompts small (system + last 12 turns).
        if len(history) > 1 + 24:
            history = [history[0]] + history[-24:]


if __name__ == "__main__":
    sys.exit(main())
