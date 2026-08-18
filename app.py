"""kjvCoach — scripture-only KJV lookup and deterministic Bible facts."""

import json
import os
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccdb as _ccdb
import kjvcode as _kjv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, "kjv.db")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
PATTERNS_PATH = Path(BASE_DIR) / "patterns.json"

app = FastAPI(title="kjvCoach", description="KJV scripture lookup and Bible facts")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_verify_cache: dict | None = None

@app.on_event("startup")
async def _precompute():
    """Pre-compute all pattern verifications at startup so /verify is instant."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _build_verify_cache)

def _build_verify_cache():
    global _verify_cache
    db       = _ccdb.load()
    patterns = json.loads(PATTERNS_PATH.read_text())
    results  = []
    for p in patterns:
        ptype = p.get("type", "word_sum")
        if ptype == "four_names_980x3":
            r = _verify_four_names(p)
        elif ptype == "jesus_god_7777":
            r = _verify_jesus_god_7777(p, db)
        elif ptype == "staircase_verses":
            r = _verify_staircase_verses(p)
        elif ptype == "jesus_christ_777":
            r = _verify_jesus_christ(p)
        elif ptype == "father_son":
            r = _verify_father_son(p, db)
        elif ptype == "alternating_books":
            r = _verify_alternating_books(p)
        else:
            r = _verify_word_sum(p, db)
        expected = p["expected"]
        results.append({
            "id":       p["id"],
            "label":    p["label"],
            "url":      p.get("url", ""),
            "note":     p.get("note", ""),
            "type":     ptype,
            "expected": expected,
            "actual":   r["actual"],
            "pass":     r["actual"] == expected,
            **{k: v for k, v in r.items() if k != "actual"},
        })
    _verify_cache = {
        "source":   "KJV 1769 Blayney concordance text + KJPBS bbl-kjv1769.ccdb",
        "patterns": results,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _require_db():
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, "Database not ready — run load.py first.")


def _verse_dict(row) -> Dict[str, Any]:
    return {
        "book":    row["name"],
        "book_id": row["book_id"],
        "chapter": row["chapter"],
        "verse":   row["verse"],
        "text":    row["text"],
        "ref":     f"{row['name']} {row['chapter']}:{row['verse']}",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    _require_db()
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    exists = os.path.exists(DB_PATH)
    return {"status": "ok", "db": exists}


@app.get("/books")
def books(testament: Optional[str] = None):
    _require_db()
    conn = _db()
    sql    = "SELECT id, name, testament, chapters, verses, words FROM books"
    params: list = []
    if testament:
        t = testament.upper()
        if t not in ("OT", "NT"):
            raise HTTPException(400, "testament must be OT or NT")
        sql += " WHERE testament = ?"
        params.append(t)
    sql += " ORDER BY id"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return {"books": [dict(r) for r in rows]}


@app.get("/verse/{book_name}/{chapter}/{verse}")
def get_verse(book_name: str, chapter: int, verse: int):
    _require_db()
    conn = _db()
    row = conn.execute("""
        SELECT b.name, v.book_id, v.chapter, v.verse, v.text
        FROM verses v JOIN books b ON b.id = v.book_id
        WHERE lower(b.name) = lower(?) AND v.chapter = ? AND v.verse = ?
    """, (book_name, chapter, verse)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"{book_name} {chapter}:{verse} not found")
    return _verse_dict(row)


@app.get("/chapter/{book_name}/{chapter}")
def get_chapter(book_name: str, chapter: int):
    _require_db()
    conn = _db()
    rows = conn.execute("""
        SELECT b.name, v.book_id, v.chapter, v.verse, v.text
        FROM verses v JOIN books b ON b.id = v.book_id
        WHERE lower(b.name) = lower(?) AND v.chapter = ?
        ORDER BY v.verse
    """, (book_name, chapter)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, f"{book_name} chapter {chapter} not found")
    return {
        "book":        rows[0]["name"],
        "chapter":     chapter,
        "verse_count": len(rows),
        "verses":      [_verse_dict(r) for r in rows],
    }


@app.get("/random")
def random_verse(testament: Optional[str] = None):
    _require_db()
    conn = _db()
    sql    = """
        SELECT b.name, v.book_id, v.chapter, v.verse, v.text
        FROM verses v JOIN books b ON b.id = v.book_id
    """
    params: list = []
    if testament:
        sql += " WHERE b.testament = ?"
        params.append(testament.upper())
    sql += " ORDER BY RANDOM() LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return _verse_dict(row)


@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    testament: Optional[str] = None,
    book:      Optional[str] = None,
    limit: int = Query(default=25, le=200),
):
    _require_db()
    conn  = _db()
    fts_q = _sanitize_fts(q)

    sql = """
        SELECT b.name, v.book_id, v.chapter, v.verse, v.text
        FROM verses_fts f
        JOIN verses v ON v.id = f.rowid
        JOIN books  b ON b.id = v.book_id
        WHERE verses_fts MATCH ?
    """
    params: list = [fts_q]

    if testament:
        t = testament.upper()
        if t not in ("OT", "NT"):
            raise HTTPException(400, "testament must be OT or NT")
        sql += " AND b.testament = ?"
        params.append(t)

    if book:
        sql += " AND lower(b.name) = lower(?)"
        params.append(book)

    sql += " ORDER BY bm25(verses_fts) LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        raise HTTPException(400, "Invalid search query")

    conn.close()
    return {
        "query":  q,
        "count":  len(rows),
        "verses": [_verse_dict(r) for r in rows],
    }


def _sanitize_fts(q: str) -> str:
    cleaned = re.sub(r'["\*\(\)\:\^]', ' ', q).strip()
    if ' ' in cleaned:
        return f'"{cleaned}"'
    return f'{cleaned}*'


# ---------------------------------------------------------------------------
# Deterministic Bible facts — all computed directly from the DB
# ---------------------------------------------------------------------------

_STOP = {
    "the","and","of","to","that","in","he","his","a","for","i","with",
    "shall","they","them","it","unto","is","be","was","not","said","all",
    "are","have","from","this","which","thou","thee","but","when","their",
    "were","as","my","by","so","we","her","him","out","up","at","an","had",
    "then","your","thy","upon","will","on","who","or","me","no","o","ye",
    "us","did","may","do","our","one","into","if","what","also","now",
    "came","went","there","even","than","these","after","any","yet","come",
    "let","how","set","more","down","put","its","hath","against","again",
    "before","over","those","would","made","make","through","you","has",
    "been","him","man","men","son","sons","children","people","land","day",
    "days","time","hand","name","house","lord","god","king","israel",
}


@lru_cache(maxsize=1)
def _compute_stats() -> Dict[str, Any]:
    conn = _db()

    def one(sql, *p):
        return conn.execute(sql, p).fetchone()

    def val(sql, *p):
        return conn.execute(sql, p).fetchone()[0]

    # --- totals ---
    total_books    = val("SELECT COUNT(*) FROM books")
    total_chapters = val("SELECT COUNT(DISTINCT book_id||'-'||chapter) FROM verses")
    total_verses   = val("SELECT COUNT(*) FROM verses")
    total_words    = val("SELECT SUM(word_count) FROM verses")

    ot_verses = val("SELECT SUM(b.verses) FROM books b WHERE b.testament='OT'")
    ot_words  = val("SELECT SUM(b.words)  FROM books b WHERE b.testament='OT'")
    nt_verses = val("SELECT SUM(b.verses) FROM books b WHERE b.testament='NT'")
    nt_words  = val("SELECT SUM(b.words)  FROM books b WHERE b.testament='NT'")

    # --- book records ---
    long_book  = one("SELECT name, verses FROM books ORDER BY verses DESC LIMIT 1")
    short_book = one("SELECT name, verses FROM books ORDER BY verses ASC  LIMIT 1")
    most_ch    = one("SELECT name, chapters FROM books ORDER BY chapters DESC LIMIT 1")

    # --- verse records ---
    long_v = one("""
        SELECT b.name, v.chapter, v.verse, v.text, v.word_count
        FROM verses v JOIN books b ON b.id=v.book_id
        ORDER BY v.word_count DESC LIMIT 1
    """)
    short_v = one("""
        SELECT b.name, v.chapter, v.verse, v.text, v.word_count
        FROM verses v JOIN books b ON b.id=v.book_id
        ORDER BY v.word_count ASC LIMIT 1
    """)

    # Middle verse
    mid_offset = total_verses // 2
    mid_v = one("""
        SELECT b.name, v.chapter, v.verse, v.text
        FROM verses v JOIN books b ON b.id=v.book_id
        ORDER BY v.book_id, v.chapter, v.verse
        LIMIT 1 OFFSET ?
    """, mid_offset)

    # --- kjvcode.com patterns (deterministic) ---

    # Exact word counts (whole-word, case-insensitive)
    def count_exact(word):
        # Match word bounded by spaces or start/end of text
        pattern = f"% {word} %"
        # Count occurrences using both patterns for start/end of verse
        n = val(
            "SELECT COUNT(*) FROM verses WHERE ' '||text||' ' LIKE ?",
            f"% {word} %",
        )
        return n

    jesus_count    = val("SELECT COUNT(*) FROM verses WHERE ' '||lower(text)||' ' LIKE '% jesus %'")
    lord_upper     = val("SELECT COUNT(*) FROM verses WHERE text LIKE '%LORD%'")
    lord_lower     = val("SELECT COUNT(*) FROM verses WHERE text GLOB '*[Ll]ord*' AND text NOT LIKE '%LORD%'")
    god_count      = val("SELECT COUNT(*) FROM verses WHERE ' '||lower(text)||' ' LIKE '% god %'")
    christ_count   = val("SELECT COUNT(*) FROM verses WHERE ' '||lower(text)||' ' LIKE '% christ %'")
    combined       = val("""
        SELECT COUNT(*) FROM verses
        WHERE lower(text) LIKE '%lord%'
           OR lower(text) LIKE '%god%'
           OR lower(text) LIKE '%jesus%'
           OR lower(text) LIKE '%christ%'
    """)

    # Verses ending in punctuation vs not (kjvcode.com claims 77%)
    punct_end = val("SELECT COUNT(*) FROM verses WHERE text GLOB '*[.!?;:]'")
    no_punct  = val("SELECT COUNT(*) FROM verses WHERE text NOT GLOB '*[.!?;:]'")
    punct_pct = round(punct_end / total_verses * 100, 1)

    # Verses with no ending punctuation — kjvcode.com claims exactly 7
    no_punct_verses = conn.execute("""
        SELECT b.name, v.chapter, v.verse, v.text
        FROM verses v JOIN books b ON b.id=v.book_id
        WHERE v.text NOT GLOB '*[.!?;:,]'
        ORDER BY v.book_id, v.chapter, v.verse
        LIMIT 20
    """).fetchall()

    # OT/NT word ratio
    ot_nt_ratio = round(ot_words / nt_words, 3) if nt_words else None

    # --- keyword counts ---
    keywords = {}
    for word in ["love", "fear", "faith", "grace", "pray", "heart",
                 "covenant", "righteous", "mercy", "truth", "peace", "holy"]:
        keywords[word] = val(
            "SELECT COUNT(*) FROM verses WHERE lower(text) LIKE ?",
            f"%{word}%",
        )

    # --- top meaningful words ---
    all_text_rows = conn.execute("SELECT text FROM verses").fetchall()
    counter: Counter = Counter()
    for (text,) in all_text_rows:
        for w in re.findall(r"[a-zA-Z']+", text):
            w_lower = w.lower().rstrip("'s")
            if w_lower not in _STOP and len(w_lower) > 3:
                counter[w_lower] += 1
    top_words = counter.most_common(20)

    # --- top books by verse count ---
    top_books = conn.execute(
        "SELECT name, verses FROM books ORDER BY verses DESC LIMIT 10"
    ).fetchall()

    conn.close()

    return {
        "totals": {
            "books":    total_books,
            "chapters": total_chapters,
            "verses":   total_verses,
            "words":    total_words,
            "OT": {"verses": ot_verses, "words": ot_words},
            "NT": {"verses": nt_verses, "words": nt_words},
            "OT_NT_word_ratio": ot_nt_ratio,
        },
        "longest_book":  {"name": long_book["name"],  "verses": long_book["verses"]},
        "shortest_book": {"name": short_book["name"], "verses": short_book["verses"]},
        "most_chapters": {"name": most_ch["name"],    "chapters": most_ch["chapters"]},
        "longest_verse": {
            "ref":   f"{long_v['name']} {long_v['chapter']}:{long_v['verse']}",
            "words": long_v["word_count"],
            "text":  long_v["text"],
        },
        "shortest_verse": {
            "ref":   f"{short_v['name']} {short_v['chapter']}:{short_v['verse']}",
            "words": short_v["word_count"],
            "text":  short_v["text"],
        },
        "middle_verse": {
            "ref":  f"{mid_v['name']} {mid_v['chapter']}:{mid_v['verse']}",
            "text": mid_v["text"],
            "note": f"verse {mid_offset + 1} of {total_verses}",
        },
        "kjvcode_patterns": {
            "jesus_in_verses":       jesus_count,
            "jesus_70x7":            jesus_count == 490,
            "LORD_all_caps":         lord_upper,
            "god_in_verses":         god_count,
            "christ_in_verses":      christ_count,
            "lord_god_jesus_christ_combined": combined,
            "verses_ending_punct":   punct_end,
            "verses_no_ending_punct": no_punct,
            "punct_end_pct":         punct_pct,
            "verses_no_punct_list": [
                f"{r['name']} {r['chapter']}:{r['verse']}" for r in no_punct_verses
            ],
            "OT_NT_word_ratio":      ot_nt_ratio,
        },
        "keyword_counts":         keywords,
        "top_words":              [{"word": w, "count": c} for w, c in top_words],
        "top_books_by_verses":    [{"name": r["name"], "verses": r["verses"]} for r in top_books],
    }


@app.get("/stats")
def stats():
    _require_db()
    return _compute_stats()


_KJV_TXT = Path(BASE_DIR) / "Holy-Bible-King-James-Version-Entire-Bible-Concord.txt"
_NT_BOOKS = [
    "Matthew","Mark","Luke","John","Acts","Romans",
    "1 Corinthians","2 Corinthians","Galatians","Ephesians",
    "Philippians","Colossians","1 Thessalonians","2 Thessalonians",
    "1 Timothy","2 Timothy","Titus","Philemon","Hebrews",
    "James","1 Peter","2 Peter","1 John","2 John","3 John",
    "Jude","Revelation",
]

@lru_cache(maxsize=1)
def _kjv_corpus():
    return _kjv._load_kjv(Path("/nonexistent"), _KJV_TXT)


def _verify_word_sum(p: dict, db) -> dict:
    breakdown, total = [], 0
    for term in p["terms"]:
        word, label, exact = term["word"], term["label"], term.get("exact_form")
        if word == "__raw__":
            count = db.raw_token_count()
            forms = {"(all whitespace-split tokens)": count}
        else:
            forms = db.word_forms(word)
            count = forms.get(exact, 0) if exact is not None else sum(forms.values())
        total += count
        breakdown.append({"label": label, "word": word, "exact": exact,
                          "forms": forms, "count": count})
    return {"breakdown": breakdown, "actual": total}


def _verify_four_names(p: dict) -> dict:
    corpus = _kjv_corpus()

    # Jesus pure (same method as pattern 3)
    pat_Jmix  = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    j_anti    = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    jesus = sum(
        len(pat_Jmix.findall(t)) + len(pat_JESUS.findall(t))
        for b, ch, v, t in corpus if (b, ch, v) not in j_anti
    )

    # Christ('s), David('s), Abraham('s) — possessives use curly apostrophes,
    # so \bName\b already captures Name and Name's (curly quote is non-word char)
    def count_name(name):
        pat = re.compile(r"\b" + name + r"\b")
        return sum(len(pat.findall(t)) for _, _, _, t in corpus)

    christ   = count_name("Christ")
    david    = count_name("David")
    abraham  = count_name("Abraham")

    components = [
        ("Jesus / JESUS (pure, 3 antimentions excluded)", jesus,   980),
        ("Christ('s)",                                    christ,  571),
        ("David('s)",                                     david,   1139),
        ("Abraham('s)",                                   abraham, 250),
    ]

    return {
        "breakdown": [
            {"label": lbl, "count": cnt, "expected": exp}
            for lbl, cnt, exp in components
        ],
        "actual": sum(cnt for _, cnt, _ in components),
    }


def _verify_jesus_god_7777(p: dict, db) -> dict:
    corpus = _kjv_corpus()

    # God titles from CCDB (exact all-caps forms)
    LORD    = db.word_count("lord",    exact_form="LORD")
    GOD     = db.word_count("god",     exact_form="GOD")
    JEHOVAH = db.word_count("jehovah", exact_form="JEHOVAH")
    JAH     = db.word_count("jah",     exact_form="JAH")
    BRANCH  = db.word_count("branch",  exact_form="BRANCH")
    KING    = db.word_count("king",    exact_form="KING")

    # I AM: count occurrences in Exodus 3:14 only (God's self-declared name)
    pat_iam = re.compile(r"\bI AM\b")
    I_AM = sum(len(pat_iam.findall(t))
               for b, ch, v, t in corpus if b == "Exodus" and ch == 3 and v == 14)

    # Jesus pure: all forms minus antimentions (same as pattern 3)
    pat_J     = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    j_anti    = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    jesus_pure = sum(
        len(pat_J.findall(t)) + len(pat_JESUS.findall(t))
        for b, ch, v, t in corpus if (b, ch, v) not in j_anti
    )

    components = [
        ("LORD (all caps)",     LORD),
        ("GOD (all caps)",      GOD),
        ("JEHOVAH",             JEHOVAH),
        ("I AM — Exodus 3:14 only", I_AM),
        ("JAH",                 JAH),
        ("BRANCH (all caps)",   BRANCH),
        ("KING (all caps)",     KING),
        ("Jesus/JESUS (pure, 3 antimentions excluded)", jesus_pure),
    ]

    return {
        "breakdown": [{"label": lbl, "count": cnt} for lbl, cnt in components],
        "actual":    sum(cnt for _, cnt in components),
    }


def _verify_staircase_verses(p: dict) -> dict:
    corpus  = _kjv_corpus()
    total   = len(corpus)
    stairs  = p["staircase"]
    return {
        "breakdown": [{
            "total_verses": total,
            "staircase":    stairs,
            "staircase_sum": sum(stairs),
        }],
        "actual": total,
    }


def _verify_jesus_christ(p: dict) -> dict:
    corpus = _kjv_corpus()
    pat_Jmix  = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    pat_C     = re.compile(r"\bChrist(?:s|'s|ian(?:s)?)?\b")

    j_anti = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    c_anti = {(a["book"], a["chapter"], a["verse"]) for a in p["christ_antimentions"]}

    j_raw = j_pure = c_raw = c_pure = 0
    for b, ch, v, text in corpus:
        jh  = len(pat_Jmix.findall(text)) + len(pat_JESUS.findall(text))
        ch_ = len(pat_C.findall(text))
        j_raw += jh;  c_raw += ch_
        if (b, ch, v) not in j_anti: j_pure += jh
        if (b, ch, v) not in c_anti: c_pure += ch_

    return {
        "breakdown": [
            {
                "label":       "Jesus(*) — Jesus, Jesus', JESUS",
                "raw_count":   j_raw,
                "antimentions": [{"ref": f"{a['book']} {a['chapter']}:{a['verse']}", "note": a["note"]}
                                 for a in p["jesus_antimentions"]],
                "count":       j_pure,
            },
            {
                "label":       "Christ(*) — Christ, Christs, Christ's, Christian, Christians",
                "raw_count":   c_raw,
                "antimentions": [{"ref": f"{a['book']} {a['chapter']}:{a['verse']}", "note": a["note"]}
                                 for a in p["christ_antimentions"]],
                "count":       c_pure,
            },
        ],
        "actual": j_pure + c_pure,
    }


def _verify_father_son(p: dict, db) -> dict:
    corpus = _kjv_corpus()

    # Father: use CCDB capital-F count, subtract specific antimentions
    father_ccdb  = db.word_count("father", exact_form="Father")
    father_anti  = p.get("father_antimentions", [])
    father_count = father_ccdb - len(father_anti)

    # Son: use corpus capital-S count, subtract Ezekiel + specific verse antimentions
    pat_S = re.compile(r"\bSon\b")
    son_verse_anti = {
        (a["book"], a["chapter"], a["verse"])
        for a in p.get("son_antimentions", [])
        if a["chapter"] is not None
    }
    son_count = 0
    for b, ch, v, text in corpus:
        hits = len(pat_S.findall(text))
        if hits == 0:
            continue
        if b == "Ezekiel":
            continue  # all antimentions
        if (b, ch, v) in son_verse_anti:
            continue
        son_count += hits

    actual = father_count + son_count
    return {
        "breakdown": [
            {
                "label":      "Father (capital F, entire Bible)",
                "word":       "Father",
                "ccdb_count": father_ccdb,
                "antimentions": [{"ref": f"{a['book']} {a['chapter']}:{a['verse']}", "note": a["note"]}
                                 for a in father_anti],
                "count":      father_count,
            },
            {
                "label":      "Son (capital S, entire Bible)",
                "word":       "Son",
                "ccdb_count": 297,
                "antimentions": [
                    {"ref": "Ezekiel (all 61)", "note": "Son of man — God addressing Ezekiel"},
                    *[{"ref": f"{a['book']} {a['chapter']}:{a['verse']}", "note": a["note"]}
                      for a in p.get("son_antimentions", []) if a["chapter"] is not None],
                ],
                "count": son_count,
            },
        ],
        "actual": actual,
    }


def _verify_alternating_books(p: dict) -> dict:
    corpus = _kjv_corpus()
    term = p["term"]
    cs   = p.get("case_sensitive", False)
    pat  = re.compile(r"\b" + re.escape(term) + r"\b", 0 if cs else re.IGNORECASE)

    # Build antimentation set: {(book, chapter, verse)}
    anti = {(a["book"], a["chapter"], a["verse"]) for a in p.get("antimentions", [])}

    books = _NT_BOOKS if p.get("testament") == "NT" else _NT_BOOKS
    book_counts = {}
    for book in books:
        total = 0
        for b, ch, v, text in corpus:
            if b != book:
                continue
            hits = len(pat.findall(text))
            if (book, ch, v) in anti:
                hits = 0
            total += hits
        book_counts[book] = total

    odd_total  = sum(c for i, b in enumerate(books, 1) if i % 2 == 1 for c in [book_counts[b]])
    even_total = sum(c for i, b in enumerate(books, 1) if i % 2 == 0 for c in [book_counts[b]])
    actual     = odd_total + even_total

    book_rows = [
        {
            "book":   b,
            "number": i,
            "parity": "odd" if i % 2 == 1 else "even",
            "count":  book_counts[b],
        }
        for i, b in enumerate(books, 1)
    ]

    return {
        "breakdown": book_rows,
        "actual":    actual,
        "odd_total": odd_total,
        "even_total": even_total,
        "antimentions": [
            {"ref": f"{a['book']} {a['chapter']}:{a['verse']}"}
            for a in p.get("antimentions", [])
        ],
    }


@app.get("/verify")
def verify():
    """Return pre-computed pattern verification results (computed at startup)."""
    _require_db()
    if _verify_cache is None:
        _build_verify_cache()
    return _verify_cache


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
