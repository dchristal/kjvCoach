"""kjvCoach — scripture-only KJV lookup and deterministic Bible facts."""

import json
import os
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio

import ccdb as _ccdb
import kjvcode as _kjv
from chat import SYSTEM_PROMPT, KJV_TOOLS, _dispatch_tool
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI as _OpenAI

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, "kjv.db")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

# ---------------------------------------------------------------------------
# LLM client (daveLLM — connects to LM Studio or any OpenAI-compatible server)
# ---------------------------------------------------------------------------
# LLM client: Groq free tier when GROQ_API_KEY is set, else LM Studio local.
# Set LLM_BASE_URL / LLM_MODEL to override either path.
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if _GROQ_API_KEY:
    _LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    _LLM_MODEL    = os.environ.get("LLM_MODEL",    "llama-3.3-70b-versatile")
    _llm          = _OpenAI(base_url=_LLM_BASE_URL, api_key=_GROQ_API_KEY)
else:
    _LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
    _LLM_MODEL    = os.environ.get("LLM_MODEL",    "google/gemma-4-e4b")
    _llm          = _OpenAI(base_url=_LLM_BASE_URL,
                            api_key=os.environ.get("LLM_API_KEY", "lm-studio"))
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
        if ptype == "jesus_inner_circle_1200":
            r = _verify_inner_circle_1200(p, db)
        elif ptype == "moses_aaron_1200":
            r = _verify_moses_aaron_1200(p)
        elif ptype == "fishermen_153":
            r = _verify_fishermen_153(p)
        elif ptype == "jesus_elijah":
            r = _verify_jesus_elijah(p)
        elif ptype == "god_moses_jesus":
            r = _verify_god_moses_jesus(p, db)
        elif ptype == "four_names_980x3":
            r = _verify_four_names(p, db)
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


@app.get("/lab")
def lab():
    """The verification workbench — kept for counting, not for strangers."""
    _require_db()
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/153")
def public_153():
    """Public campaign door — one pattern, starting from John 21 and the math."""
    return FileResponse(os.path.join(STATIC_DIR, "153.html"))


@app.get("/api/153")
def api_153():
    """Live fishermen-153 payload for the public page. Computed on its own so
    the door does not wait for the full /verify cache."""
    _require_db()
    patterns = json.loads(PATTERNS_PATH.read_text())
    spec = next((p for p in patterns if p.get("id") == "fishermen-153"), None)
    if spec is None:
        raise HTTPException(404, "fishermen-153 pattern not found")
    r = _verify_fishermen_153(spec)
    expected = spec["expected"]
    actual = r["actual"]
    return {
        "id":       spec["id"],
        "label":    spec["label"],
        "url":      spec.get("url", ""),
        "note":     spec.get("note", ""),
        "expected": expected,
        "actual":   actual,
        "pass":     actual == expected,
        "source":   "KJV 1769 Blayney concordance text (Cambridge Concord)",
        **{k: v for k, v in r.items() if k != "actual"},
    }


@app.get("/api/chat/info")
def api_chat_info():
    """Which LLM backend is active."""
    return {
        "model":   _LLM_MODEL,
        "backend": "groq" if _GROQ_API_KEY else "local",
        "enabled": bool(_GROQ_API_KEY or True),
    }


@app.post("/api/chat")
async def api_chat(payload: dict):
    """Stream a KJV-grounded LLM response as SSE.

    Body: {"messages": [{"role": "user|assistant", "content": "..."}]}
    Events: {"token": str} | {"tool": str, "result": dict} | {"error": str} | [DONE]
    """
    user_messages = payload.get("messages", [])
    _KJV_PATH = Path(BASE_DIR) / "kjv.txt"

    async def _stream():
        turn = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]}
            for m in user_messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        try:
            # Tool-calling loop (non-streaming)
            for _ in range(4):
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=turn: _llm.chat.completions.create(
                        model=_LLM_MODEL, messages=t,
                        tools=KJV_TOOLS, tool_choice="auto", stream=False,
                    ),
                )
                msg = resp.choices[0].message
                calls = getattr(msg, "tool_calls", None) or []
                if not calls:
                    break
                turn.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in calls
                    ],
                })
                for tc in calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = _dispatch_tool(tc.function.name, args,
                                            Path("/nonexistent"), _KJV_PATH)
                    yield f"data: {json.dumps({'tool': tc.function.name, 'result': result})}\n\n"
                    turn.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })

            # Stream the final text response
            stream = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _llm.chat.completions.create(
                    model=_LLM_MODEL, messages=turn, stream=True,
                ),
            )
            for ev in stream:
                delta = ev.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"

        except Exception as exc:
            msg = str(exc)
            if "Connection" in msg or "connect" in msg.lower() or "refused" in msg.lower():
                if _GROQ_API_KEY:
                    msg = "Could not reach Groq. Check your GROQ_API_KEY and internet connection."
                else:
                    msg = f"Cannot reach LLM server at {_LLM_BASE_URL}. Is LM Studio running with a model loaded?"
            yield f"data: {json.dumps({'error': msg})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


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


_KJV_TXT = Path(BASE_DIR) / "kjv.txt"
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


def _verify_four_names(p: dict, db) -> dict:
    corpus = _kjv_corpus()

    # Jesus pure (corpus + antimentions logic)
    pat_Jmix  = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    j_anti    = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    jesus = sum(
        len(pat_Jmix.findall(t)) + len(pat_JESUS.findall(t))
        for b, ch, v, t in corpus if (b, ch, v) not in j_anti
    )

    # Christ('s): corpus regex matches base + curly-apostrophe possessives
    pat_christ = re.compile(r"\bChrist\b")
    christ = sum(len(pat_christ.findall(t)) for _, _, _, t in corpus)

    # David('s) and Abraham('s): use CCDB exact counts (base + possessive).
    # The concordance file is 1 David short of the 1769 CCDB text.
    # CCDB stores possessives with curly apostrophe (U+2019) as the key.
    curly = "’"
    david   = (db.word_count("david",            exact_form="David")
             + sum(db.word_forms("david"  + curly + "s").values()))
    abraham = (db.word_count("abraham",          exact_form="Abraham")
             + sum(db.word_forms("abraham" + curly + "s").values()))

    components = [
        ("Jesus / JESUS (pure, 3 antimentions excluded)", jesus,   980),
        ("Christ('s)  — corpus",                          christ,  571),
        ("David('s)   — CCDB 1769 text",                  david,   1139),
        ("Abraham('s) — CCDB 1769 text",                  abraham, 250),
    ]

    return {
        "breakdown": [
            {"label": lbl, "count": cnt, "expected": exp}
            for lbl, cnt, exp in components
        ],
        "actual": sum(cnt for _, cnt, _ in components),
    }


def _verify_inner_circle_1200(p: dict, db=None) -> dict:
    corpus = _kjv_corpus()
    if db is None:
        db = _ccdb.load()

    # Jesus pure (established 3 antimentions)
    j_anti = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    pat_J     = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    jesus = sum(
        len(pat_J.findall(t)) + len(pat_JESUS.findall(t))
        for b, ch, v, t in corpus if (b, ch, v) not in j_anti
    )

    # Peter: case-insensitive (catches PETER at 1 Pet 1:1), no antimentions
    pat_peter = re.compile(r"\bPeter\b", re.IGNORECASE)
    peter = sum(len(pat_peter.findall(t)) for _, _, _, t in corpus)

    # James: case-insensitive (catches JAMES at Jas 1:1)
    # Per-verse antimentation subtraction (some verses have multiple James)
    james_anti_count = {
        # Gospels: James the Less (son of Alphaeus, 8×) + half-brother (2×)
        ("Matthew", 10, 3): 1, ("Matthew", 13, 55): 1, ("Matthew", 27, 56): 1,
        ("Mark", 3, 18): 1,   ("Mark", 6, 3): 1,      ("Mark", 15, 40): 1,
        ("Mark", 16, 1): 1,   ("Luke", 6, 15): 1,      ("Luke", 6, 16): 1,
        ("Luke", 24, 10): 1,
        # Outside Gospels: Acts 1:13 has 1 apostle + 2 antimentions (James the Less + Judas's James)
        ("Acts", 1, 13): 2,
        ("Acts", 12, 17): 1, ("Acts", 15, 13): 1, ("Acts", 21, 18): 1,
        ("1 Corinthians", 15, 7): 1,
        ("Galatians", 1, 19): 1, ("Galatians", 2, 9): 1, ("Galatians", 2, 12): 1,
        ("Jude", 1, 1): 1,
        # JAMES all-caps at James 1:1 (Lord's brother, not apostle Zebedee)
        ("James", 1, 1): 1,
    }
    pat_james = re.compile(r"\bJames\b", re.IGNORECASE)
    james = 0
    for b, ch, v, text in corpus:
        hits = len(pat_james.findall(text))
        if hits:
            james += hits - james_anti_count.get((b, ch, v), 0)

    # John: case-insensitive (catches JOHN at Rev 1:4)
    # Whitelist of apostle John verses (all others = John the Baptist or John Mark)
    john_apostle = {
        # Gospels (20 verses)
        ("Matthew", 4, 21), ("Matthew", 10, 2), ("Matthew", 17, 1),
        ("Mark", 1, 19), ("Mark", 1, 29), ("Mark", 3, 17), ("Mark", 5, 37),
        ("Mark", 9, 2), ("Mark", 9, 38), ("Mark", 10, 35), ("Mark", 10, 41),
        ("Mark", 13, 3), ("Mark", 14, 33),
        ("Luke", 5, 10), ("Luke", 6, 14), ("Luke", 8, 51),
        ("Luke", 9, 28), ("Luke", 9, 49), ("Luke", 9, 54), ("Luke", 22, 8),
        # Acts (11 verses)
        # 4:6: "John" listed at the gathering for Peter & John's trial (apostle present as defendant)
        # 13:5: "had also John to their minister" — the verse names only "John" without "Mark"
        ("Acts", 1, 13), ("Acts", 3, 1), ("Acts", 3, 3), ("Acts", 3, 4),
        ("Acts", 3, 11), ("Acts", 4, 6), ("Acts", 4, 13), ("Acts", 4, 19),
        ("Acts", 8, 14), ("Acts", 12, 2), ("Acts", 13, 5),
        # Galatians (1 verse — pillars of the church)
        ("Galatians", 2, 9),
        # Revelation (5 verses — author self-identifies)
        ("Revelation", 1, 1), ("Revelation", 1, 4), ("Revelation", 1, 9),
        ("Revelation", 21, 2), ("Revelation", 22, 8),
    }
    pat_john = re.compile(r"\bJohn\b", re.IGNORECASE)
    john = sum(
        len(pat_john.findall(t))
        for b, ch, v, t in corpus if (b, ch, v) in john_apostle
    )

    components = [
        ("Jesus (pure) — 3 antimentions excluded",                   jesus, 980),
        ("Peter (incl. PETER at 1 Pet 1:1) — no antimentions",       peter, 162),
        ("James (son of Zebedee) — 21 antimentions excl. incl. JAMES 1:1", james, 21),
        ("John (son of Zebedee) — 96 Baptist/Mark antimentions excl.", john,  37),
    ]
    return {
        "breakdown": [{"label": lbl, "count": cnt, "expected": exp}
                      for lbl, cnt, exp in components],
        "actual": jesus + peter + james + john,
    }


def _verify_moses_aaron_1200(p: dict) -> dict:
    corpus = _kjv_corpus()
    # Moses: \bMoses\b also captures Moses' (curly apostrophe is non-word boundary)
    pat_moses = re.compile(r"\bMoses\b")
    # Aaron*: base + Aaron's (possessive, curly) + Aaronites (1 Chr 12:27, 27:17)
    pat_aaron = re.compile(r"\bAaron(?:ites?)?\b")

    moses = sum(len(pat_moses.findall(t)) for _, _, _, t in corpus)
    aaron = sum(len(pat_aaron.findall(t)) for _, _, _, t in corpus)

    components = [
        ("Moses(') — entire Bible incl. possessives",              moses, 848),
        ("Aaron(*) — entire Bible incl. Aaron's + Aaronites",      aaron, 352),
    ]
    return {
        "breakdown": [{"label": lbl, "count": cnt, "expected": exp}
                      for lbl, cnt, exp in components],
        "actual": moses + aaron,
    }


def _verse_from_corpus(corpus, book: str, chapter: int, verse: int) -> str:
    for b, ch, v, text in corpus:
        if b == book and ch == chapter and v == verse:
            return text
    return ""


def _verify_fishermen_153(p: dict) -> dict:
    corpus  = _kjv_corpus()
    GOSPELS = {"Matthew", "Mark", "Luke", "John"}

    # James antimentions: James the Less (son of Alphaeus, 8×) + half-brother of Jesus (2×)
    james_anti  = {(a["book"], a["chapter"], a["verse"]) for a in p.get("james_antimentions", [])}
    # John whitelist: only these verses have John the Apostle (all others = John the Baptist)
    john_apostle = {(a["book"], a["chapter"], a["verse"]) for a in p.get("john_apostle_verses", [])}

    pat_peter = re.compile(r"\bPeter\b")    # also matches Peter's (curly apos is non-word)
    pat_thom  = re.compile(r"\bThomas\b")
    pat_nath  = re.compile(r"\bNathanael\b")
    pat_james = re.compile(r"\bJames\b")
    pat_john  = re.compile(r"\bJohn\b")

    peter = thomas = nathanael = james = john = 0
    for b, ch, v, text in corpus:
        if b not in GOSPELS:
            continue
        key = (b, ch, v)
        peter     += len(pat_peter.findall(text))
        thomas    += len(pat_thom.findall(text))
        nathanael += len(pat_nath.findall(text))
        if key not in james_anti:
            james += len(pat_james.findall(text))
        if key in john_apostle:
            john += len(pat_john.findall(text))

    components = [
        ("Peter (incl. Peter's) — Gospels only, no antimentions",            peter,     97),
        ("Thomas — Gospels only, no antimentions",                            thomas,    11),
        ("Nathanael — Gospels only, no antimentions",                         nathanael,  6),
        ("James (son of Zebedee) — 10 antimentions excluded",                 james,     19),
        ("John (son of Zebedee) — 83 John the Baptist verses excluded",        john,      20),
    ]
    triangle_n = int(p.get("triangle_n", 17))
    series = list(range(1, triangle_n + 1))
    return {
        "breakdown": [{"label": lbl, "count": cnt, "expected": exp}
                      for lbl, cnt, exp in components],
        "actual": peter + thomas + nathanael + james + john,
        "crew": [
            {"name": "Peter",     "in_boat_as": "Simon Peter",          "count": peter,     "expected": 97},
            {"name": "Thomas",    "in_boat_as": "Thomas called Didymus","count": thomas,    "expected": 11},
            {"name": "Nathanael", "in_boat_as": "Nathanael of Cana",    "count": nathanael, "expected": 6},
            {"name": "James",     "in_boat_as": "the sons of Zebedee",  "count": james,     "expected": 19},
            {"name": "John",      "in_boat_as": "the sons of Zebedee",  "count": john,      "expected": 20},
        ],
        "verses": {
            "crew": {
                "ref":  "John 21:2",
                "text": _verse_from_corpus(corpus, "John", 21, 2),
            },
            "catch": {
                "ref":  "John 21:11",
                "text": _verse_from_corpus(corpus, "John", 21, 11),
            },
        },
        "triangle": {
            "n":      triangle_n,
            "series": series,
            "sum":    triangle_n * (triangle_n + 1) // 2,
            "formula": "n(n+1)/2",
        },
        "digit_cubes": {
            "digits": [1, 5, 3],
            "cubes":  [1, 125, 27],
            "sum":    1 + 125 + 27,
        },
        "scope": "Gospels only (Matthew, Mark, Luke, John)",
        "rules": [
            "Count only the four Gospels — the catch is a Gospel story.",
            "The five names are the men John identifies in the boat: Simon Peter, Thomas, Nathanael, and the two sons of Zebedee.",
            "John 21:2 says “the sons of Zebedee,” not “James and John.” Matthew 4:21 and Mark 1:19 name those sons.",
            "Two other disciples are in the boat and unnamed. They are not counted — there is no name to count.",
            "Peter includes the possessive Peter’s. Thomas and Nathanael have no exclusions.",
            "James drops 10 Gospel verses that name James the Less (son of Alphaeus) or James the Lord’s brother.",
            "John counts only 20 verses where the name is the apostle, not John the Baptist.",
        ],
    }


def _verify_jesus_elijah(p: dict) -> dict:
    corpus = _kjv_corpus()

    j_anti = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}
    e_anti = {(a["book"], a["chapter"], a["verse"]) for a in p.get("elijah_antimentions", [])}

    pat_J     = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")
    pat_elijah = re.compile(r"\bElijah\b")
    pat_elias  = re.compile(r"\bElias\b")

    jesus = elijah = elias = 0
    for b, ch, v, text in corpus:
        key = (b, ch, v)
        if key not in j_anti:
            jesus += len(pat_J.findall(text)) + len(pat_JESUS.findall(text))
        if key not in e_anti:
            elijah += len(pat_elijah.findall(text))
        elias += len(pat_elias.findall(text))

    components = [
        ("Jesus (pure) — 3 antimentions excluded",          jesus,  980),
        ("Elijah — Ezra 10:21 antimentation excluded",       elijah,  68),
        ("Elias (NT form of Elijah) — no antimentions",      elias,   30),
    ]
    return {
        "breakdown": [
            {"label": lbl, "count": cnt, "expected": exp}
            for lbl, cnt, exp in components
        ],
        "actual": jesus + elijah + elias,
    }


def _verify_god_moses_jesus(p: dict, db) -> dict:
    corpus = _kjv_corpus()
    curly  = chr(0x2019)  # U+2019 RIGHT SINGLE QUOTATION MARK (used for possessives in KJV)

    # CCDB authoritative totals (base + possessive forms)
    god_ccdb   = (db.word_count("god",   exact_form="God")
                + sum(db.word_forms("god"   + curly + "s").values()))
    moses_ccdb = (db.word_count("moses", exact_form="Moses")
                + sum(db.word_forms("moses" + curly).values()))

    # Verse-level antimentions: same 3 verses as Jesus antimentions.
    # Those verses also contain "God" — exclude all name occurrences in them.
    anti_set = {(a["book"], a["chapter"], a["verse"]) for a in p["jesus_antimentions"]}

    pat_God   = re.compile(r"\bGod\b")
    pat_Moses = re.compile(r"\bMoses\b")
    pat_J     = re.compile(r"\bJesus'?")
    pat_JESUS = re.compile(r"\bJESUS\b")

    god_anti = moses_anti = jesus = 0
    for b, ch, v, text in corpus:
        if (b, ch, v) in anti_set:
            god_anti   += len(pat_God.findall(text))
            moses_anti += len(pat_Moses.findall(text))
        else:
            jesus += len(pat_J.findall(text)) + len(pat_JESUS.findall(text))

    god   = god_ccdb   - god_anti
    moses = moses_ccdb - moses_anti

    components = [
        ("God('s) — CCDB 1769, antimetion verses excluded", god,   4102),
        ("Moses(') — CCDB 1769, antimetion verses excluded", moses, 847),
        ("Jesus (pure) — 3 antimentions excluded",            jesus, 980),
    ]
    return {
        "breakdown": [
            {"label": lbl, "count": cnt, "expected": exp}
            for lbl, cnt, exp in components
        ],
        "actual": god + moses + jesus,
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
