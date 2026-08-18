"""KJV deterministic search and statistics library for kjvCoach.

Public API:
    count(term, scope, mode, case_sensitive, *, vault, kjv_text) -> dict
    nth_mention(term, n, scope, mode, case_sensitive, *, vault, kjv_text) -> dict
    verse_at(n, scope, *, vault, kjv_text) -> dict
    position_of(ref, scope, *, vault, kjv_text) -> dict
    verse_count(scope, term, mode, case_sensitive, *, vault, kjv_text) -> dict
    chapter_count(scope, *, vault, kjv_text) -> dict
    book_count(scope, *, vault, kjv_text) -> dict
    word_count(scope, *, vault, kjv_text) -> dict
    letter_count(scope, *, vault, kjv_text) -> dict
    verify_patterns(cat, vault, kjv_text) -> None

Module-level variable:
    CORPUS_SOURCE: "concord" | "osis"  — set to "osis" to use KJPBS XML corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent
DEFAULT_KJV_TEXT = _DIR / "Holy-Bible-King-James-Version-Entire-Bible-Concord.txt"
DEFAULT_KJV_OSIS = _DIR / "kjv1769.osis.xml"
DEFAULT_PATTERNS = _DIR / "patterns.json"

CORPUS_SOURCE: str = "concord"

# ---------------------------------------------------------------------------
# Book lists and canonical order
# ---------------------------------------------------------------------------

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

_SCOPE_GROUPS: dict[str, tuple[list[str], str]] = {
    "gospels":          (["Matthew", "Mark", "Luke", "John"], "Gospels"),
    "law":              (["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"], "Pentateuch (Law)"),
    "history-ot":       (["Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                          "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                          "Ezra", "Nehemiah", "Esther"], "OT History"),
    "wisdom":           (["Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon"],
                         "Wisdom Literature"),
    "major-prophets":   (["Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel"],
                         "Major Prophets"),
    "minor-prophets":   (["Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
                          "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"],
                         "Minor Prophets"),
    "pauline":          (["Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                          "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
                          "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon"],
                         "Pauline Epistles"),
    "general-epistles": (["Hebrews", "James", "1 Peter", "2 Peter", "1 John",
                          "2 John", "3 John", "Jude"], "General Epistles"),
    "first-last-books": (["Genesis", "Revelation"], "Genesis and Revelation"),
    "apocalyptic":      (["Daniel", "Revelation"], "Apocalyptic Books"),
}

# ---------------------------------------------------------------------------
# OSIS book-id -> canonical name
# ---------------------------------------------------------------------------

_OSIS_NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
_OSIS_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
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

# ---------------------------------------------------------------------------
# Corpus caches
# ---------------------------------------------------------------------------

_KJV_CORPUS: list[tuple[str, int, int, str]] | None = None
_KJV_SOURCE: Path | None = None
_KJV_OSIS_CORPUS: list[tuple[str, int, int, str]] | None = None
_KJV_OSIS_SOURCE: Path | None = None

_VERSE_RE = re.compile(r"^\*\*(\d+)\*\*\s+(.*)$")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ---------------------------------------------------------------------------
# Concord plaintext parser
# ---------------------------------------------------------------------------

_CONCORD_HEADERS: list[tuple[str, re.Pattern[str]]] = [
    ("Genesis",          re.compile(r"^GENESIS\.$")),
    ("Exodus",           re.compile(r"^EXODUS\.$")),
    ("Leviticus",        re.compile(r"^LEVITICUS\.$")),
    ("Numbers",          re.compile(r"^NUMBERS\.$")),
    ("Deuteronomy",      re.compile(r"^DEUTERONOMY\.$")),
    ("Joshua",           re.compile(r"^THE BOOK OF JOSHUA\.$")),
    ("Judges",           re.compile(r"^THE BOOK OF JUDGES\.$")),
    ("Ruth",             re.compile(r"^THE BOOK OF RUTH\.$")),
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
_CONCORD_ACROSTIC_RE = re.compile(r"^[A-Z]{2,}\.$")


def _parse_concord(path: Path) -> list[tuple[str, int, int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    corpus: list[tuple[str, int, int, str]] = []
    hi = 0
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
        if verse_num > 0 and _CONCORD_ACROSTIC_RE.match(line.strip()):
            continue
        v = _CONCORD_VERSE_RE.match(line)
        if v:
            flush()
            verse_num = int(v.group(1))
            parts = [v.group(2).strip()]
        else:
            if verse_num == 0:
                verse_num = 1
            parts.append(line.strip())

    flush()
    return corpus


def _load_kjv(vault: Path, kjv_text: Path | None = DEFAULT_KJV_TEXT) -> list[tuple[str, int, int, str]]:
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


def _osis_text_filtered(elem) -> str:
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


def _load_kjv_osis(osis_path: Path = DEFAULT_KJV_OSIS) -> list[tuple[str, int, int, str]]:
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


def _active_corpus(vault: Path, kjv_text: Path | None) -> list[tuple[str, int, int, str]]:
    if CORPUS_SOURCE == "osis" and DEFAULT_KJV_OSIS.exists():
        return _load_kjv_osis(DEFAULT_KJV_OSIS)
    return _load_kjv(vault, kjv_text or DEFAULT_KJV_TEXT)


# ---------------------------------------------------------------------------
# Scope resolution — works on the full ordered corpus
# ---------------------------------------------------------------------------

def _normalize_book(name: str) -> str:
    want = name.strip().lower().replace(" ", "")
    for b in _ALL_BOOKS:
        if b.lower().replace(" ", "") == want:
            return b
    raise ValueError(f"unknown book: {name!r}")


def _scoped(
    corpus: list[tuple[str, int, int, str]], spec: str
) -> tuple[str, list[tuple[str, int, int, str]]]:
    """Return (label, filtered_corpus) for the given scope spec."""
    s = (spec or "all").strip().lower()

    if s in ("all", "kjv", "whole"):
        return "KJV whole canon", corpus

    if s == "ot":
        book_set = set(_OT_BOOKS)
        return "KJV OT", [r for r in corpus if r[0] in book_set]

    if s == "nt":
        book_set = set(_NT_BOOKS)
        return "KJV NT", [r for r in corpus if r[0] in book_set]

    if s == "odd-nt":
        sel = {b for i, b in enumerate(_NT_BOOKS, 1) if i % 2 == 1}
        sel_ordered = [b for b in _NT_BOOKS if b in sel]
        label = f"odd-numbered NT books ({', '.join(sel_ordered)})"
        return label, [r for r in corpus if r[0] in sel]

    if s == "even-nt":
        sel = {b for i, b in enumerate(_NT_BOOKS, 1) if i % 2 == 0}
        sel_ordered = [b for b in _NT_BOOKS if b in sel]
        label = f"even-numbered NT books ({', '.join(sel_ordered)})"
        return label, [r for r in corpus if r[0] in sel]

    if s == "odd-ot":
        sel = {b for i, b in enumerate(_OT_BOOKS, 1) if i % 2 == 1}
        label = f"odd-numbered OT books ({len(sel)} books)"
        return label, [r for r in corpus if r[0] in sel]

    if s == "even-ot":
        sel = {b for i, b in enumerate(_OT_BOOKS, 1) if i % 2 == 0}
        label = f"even-numbered OT books ({len(sel)} books)"
        return label, [r for r in corpus if r[0] in sel]

    if s in _SCOPE_GROUPS:
        books, label = _SCOPE_GROUPS[s]
        book_set = set(books)
        return label, [r for r in corpus if r[0] in book_set]

    if s.startswith("book:"):
        book = _normalize_book(spec[5:])
        return book, [r for r in corpus if r[0] == book]

    if s.startswith("chapter:"):
        rest = spec[8:].strip()
        m = re.match(r"^(.+?)\s+(\d+)$", rest)
        if not m:
            raise ValueError(f"chapter scope format: chapter:<Book> <N>, got: {spec!r}")
        book = _normalize_book(m.group(1))
        ch_n = int(m.group(2))
        filtered = [r for r in corpus if r[0] == book and r[1] == ch_n]
        return f"{book} {ch_n}", filtered

    if s.startswith("verse:"):
        rest = spec[6:].strip()
        m = re.match(r"^(.+?)\s+(\d+):(\d+)$", rest)
        if not m:
            raise ValueError(f"verse scope format: verse:<Book> <N>:<V>, got: {spec!r}")
        book = _normalize_book(m.group(1))
        ch_n, v_n = int(m.group(2)), int(m.group(3))
        filtered = [r for r in corpus if r[0] == book and r[1] == ch_n and r[2] == v_n]
        return f"{book} {ch_n}:{v_n}", filtered

    if s.startswith("range:"):
        rest = spec[6:].strip()
        m = re.match(r"^(.+?)\s+(\d+):(\d+)\s*-\s*(.+?)\s+(\d+):(\d+)$", rest)
        if not m:
            raise ValueError(
                f"range scope format: range:<Book> <N>:<V>-<Book> <N>:<V>, got: {spec!r}"
            )
        b1 = _normalize_book(m.group(1))
        ch1, v1 = int(m.group(2)), int(m.group(3))
        b2 = _normalize_book(m.group(4))
        ch2, v2 = int(m.group(5)), int(m.group(6))
        start = next(
            (i for i, r in enumerate(corpus) if r[0] == b1 and r[1] == ch1 and r[2] == v1),
            None,
        )
        end = next(
            (i for i, r in enumerate(corpus) if r[0] == b2 and r[1] == ch2 and r[2] == v2),
            None,
        )
        if start is None:
            raise ValueError(f"start verse not found: {b1} {ch1}:{v1}")
        if end is None:
            raise ValueError(f"end verse not found: {b2} {ch2}:{v2}")
        filtered = corpus[start : end + 1]
        return f"{b1} {ch1}:{v1}–{b2} {ch2}:{v2}", filtered

    raise ValueError(f"unknown scope: {spec!r}")


# ---------------------------------------------------------------------------
# Pattern builder
# ---------------------------------------------------------------------------

def _make_pattern(term: str, mode: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    if mode in ("pure", "whole-word", ""):
        words = term.split()
        pat = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
    elif mode == "standalone":
        pat = r"(?<!\S)" + re.escape(term) + r"(?!\S)"
    elif mode in ("concealed", "raw"):
        pat = re.escape(term)
    elif mode == "regex":
        pat = term
    else:
        pat = r"\b" + re.escape(term) + r"\b"
    return re.compile(pat, flags)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count(
    term: str,
    scope: str = "all",
    mode: str = "pure",
    case_sensitive: bool = False,
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Count occurrences of `term` in `scope`.

    Returns {"term", "scope", "occurrences", "verses_with_hit", "per_book"}.
    """
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError(
            "KJV corpus not found. Pass --kjv-text to the concord .txt file."
        )
    label, filtered = _scoped(corpus, scope)
    pat = _make_pattern(term, mode, case_sensitive)
    occ = 0
    verse_hits = 0
    per_book: dict[str, int] = {}
    for book, _ch, _v, text in filtered:
        n = len(pat.findall(text))
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


def nth_mention(
    term: str,
    n: int,
    scope: str = "all",
    mode: str = "pure",
    case_sensitive: bool = False,
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return reference of the Nth occurrence of `term` in `scope`.

    Returns {"n", "scope", "ref", "text"}.
    """
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    pat = _make_pattern(term, mode, case_sensitive)
    count_so_far = 0
    for book, ch, v, text in filtered:
        hits = pat.findall(text)
        if count_so_far + len(hits) >= n:
            return {"n": n, "scope": label, "ref": f"{book} {ch}:{v}", "text": text}
        count_so_far += len(hits)
    raise ValueError(
        f"'{term}' has fewer than {n} occurrences in {label} "
        f"(found {count_so_far})"
    )


def verse_at(
    n: int,
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return the Nth verse in `scope` (1-indexed).

    Returns {"n", "scope", "ref", "text"}.
    """
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    if n < 1 or n > len(filtered):
        raise ValueError(f"verse #{n} out of range 1–{len(filtered)} in {label}")
    book, ch, v, text = filtered[n - 1]
    return {"n": n, "scope": label, "ref": f"{book} {ch}:{v}", "text": text}


def position_of(
    ref: str,
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return the ordinal position of verse `ref` in `scope`.

    `ref` format: "Book N:V" e.g. "John 3:16".
    Returns {"ref", "scope", "position", "total_verses"}.
    """
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    m = re.match(r"^(.+?)\s+(\d+):(\d+)\s*$", ref.strip())
    if not m:
        raise ValueError(f"invalid ref format: {ref!r}  (expected 'Book N:V')")
    book = _normalize_book(m.group(1))
    ch_n, v_n = int(m.group(2)), int(m.group(3))
    for i, (b, ch, v, _) in enumerate(filtered, 1):
        if b == book and ch == ch_n and v == v_n:
            return {
                "ref": ref,
                "scope": label,
                "position": i,
                "total_verses": len(filtered),
            }
    raise ValueError(f"{ref} not found in {label}")


def verse_count(
    scope: str = "all",
    term: str | None = None,
    mode: str = "pure",
    case_sensitive: bool = False,
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Count verses in `scope`, optionally filtered to those containing `term`.

    Returns {"scope", "verses"} (plus "term" when term is given).
    """
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    if term:
        pat = _make_pattern(term, mode, case_sensitive)
        n = sum(1 for _, _, _, text in filtered if pat.search(text))
        return {"scope": label, "verses": n, "term": term}
    return {"scope": label, "verses": len(filtered)}


def chapter_count(
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return {"scope", "chapters"}."""
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    chapters = len({(b, ch) for b, ch, _, _ in filtered})
    return {"scope": label, "chapters": chapters}


def book_count(
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return {"scope", "books"}."""
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    books_seen = len({b for b, _, _, _ in filtered})
    return {"scope": label, "books": books_seen}


def word_count(
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return {"scope", "words"} — alpha tokens in verse text only."""
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    words = sum(len(_WORD_RE.findall(text)) for _, _, _, text in filtered)
    return {"scope": label, "words": words}


_VOWELS = frozenset("aeiouAEIOU")


def letter_count(
    scope: str = "all",
    *,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> dict:
    """Return {"scope", "letters", "vowels", "consonants"}."""
    corpus = _active_corpus(vault, kjv_text)
    if not corpus:
        raise RuntimeError("KJV corpus not found.")
    label, filtered = _scoped(corpus, scope)
    letters = vowels = consonants = 0
    for _, _, _, text in filtered:
        for ch in text:
            if ch.isalpha():
                letters += 1
                if ch in _VOWELS:
                    vowels += 1
                else:
                    consonants += 1
    return {
        "scope": label,
        "letters": letters,
        "vowels": vowels,
        "consonants": consonants,
    }


def verify_patterns(
    cat: Path,
    vault: Path = Path("/nonexistent"),
    kjv_text: Path | None = None,
) -> None:
    """Load a patterns.json catalog and verify each entry against the corpus.

    Catalog formats accepted:
      {"term": expected_count, ...}
      {"patterns": [{"term": ..., "expected": ..., "scope": ..., "mode": ...}, ...]}
      [{"term": ..., "expected": ..., "scope": ..., "mode": ...}, ...]
    """
    from rich.console import Console
    console = Console()
    data = json.loads(cat.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "patterns" in data:
        entries = data["patterns"]
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = [{"term": k, "expected": v} for k, v in data.items()]
    else:
        console.print(f"[red]Unrecognised patterns format in {cat}[/red]")
        return

    pass_n = fail_n = 0
    for entry in entries:
        term = entry.get("term") or entry.get("word", "")
        expected = int(entry.get("expected", entry.get("count", 0)))
        scope = entry.get("scope", "all")
        mode = entry.get("mode", "pure")
        case_sensitive = bool(entry.get("case_sensitive", False))
        try:
            r = count(term, scope, mode, case_sensitive, vault=vault, kjv_text=kjv_text)
            actual = r["occurrences"]
        except Exception as e:
            console.print(f"[red]ERROR[/red] {term!r}: {e}")
            fail_n += 1
            continue
        if actual == expected:
            console.print(f"[green]PASS[/green]  {term!r} ({scope}): {actual:,}")
            pass_n += 1
        else:
            console.print(
                f"[red]FAIL[/red]  {term!r} ({scope}): "
                f"got [bold]{actual:,}[/bold], expected [bold]{expected:,}[/bold]"
            )
            fail_n += 1

    total = pass_n + fail_n
    colour = "green" if fail_n == 0 else "red"
    console.print(f"\n[{colour}]{pass_n}/{total} passed[/{colour}]")
