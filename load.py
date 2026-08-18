#!/usr/bin/env python3
"""
Download the kjvcode.com KJV concordance text, parse it, and build kjv.db.

Parsing strategy:
  - "CHAPTER 1"  → new book boundary (exactly 66 in the whole file)
  - "CHAPTER N"  → new chapter within current book
  - ALL-CAPS lines → book/section headers; skip entirely
  - Lines starting with a digit → verse N, text is the rest
  - Other non-blank lines after a chapter marker → verse 1 (no prefix in this edition)
"""

import os
import re
import sqlite3
import urllib.request

SOURCE_URL = (
    "https://kjvcode.com/wp-content/uploads/2024/05/"
    "Holy-Bible-King-James-Version-Entire-Bible-Concord.txt"
)

# Canonical order: (book_num, testament, display_name)
BOOKS = [
    (1,  "OT", "Genesis"),
    (2,  "OT", "Exodus"),
    (3,  "OT", "Leviticus"),
    (4,  "OT", "Numbers"),
    (5,  "OT", "Deuteronomy"),
    (6,  "OT", "Joshua"),
    (7,  "OT", "Judges"),
    (8,  "OT", "Ruth"),
    (9,  "OT", "1 Samuel"),
    (10, "OT", "2 Samuel"),
    (11, "OT", "1 Kings"),
    (12, "OT", "2 Kings"),
    (13, "OT", "1 Chronicles"),
    (14, "OT", "2 Chronicles"),
    (15, "OT", "Ezra"),
    (16, "OT", "Nehemiah"),
    (17, "OT", "Esther"),
    (18, "OT", "Job"),
    (19, "OT", "Psalms"),
    (20, "OT", "Proverbs"),
    (21, "OT", "Ecclesiastes"),
    (22, "OT", "Song of Solomon"),
    (23, "OT", "Isaiah"),
    (24, "OT", "Jeremiah"),
    (25, "OT", "Lamentations"),
    (26, "OT", "Ezekiel"),
    (27, "OT", "Daniel"),
    (28, "OT", "Hosea"),
    (29, "OT", "Joel"),
    (30, "OT", "Amos"),
    (31, "OT", "Obadiah"),
    (32, "OT", "Jonah"),
    (33, "OT", "Micah"),
    (34, "OT", "Nahum"),
    (35, "OT", "Habakkuk"),
    (36, "OT", "Zephaniah"),
    (37, "OT", "Haggai"),
    (38, "OT", "Zechariah"),
    (39, "OT", "Malachi"),
    (40, "NT", "Matthew"),
    (41, "NT", "Mark"),
    (42, "NT", "Luke"),
    (43, "NT", "John"),
    (44, "NT", "Acts"),
    (45, "NT", "Romans"),
    (46, "NT", "1 Corinthians"),
    (47, "NT", "2 Corinthians"),
    (48, "NT", "Galatians"),
    (49, "NT", "Ephesians"),
    (50, "NT", "Philippians"),
    (51, "NT", "Colossians"),
    (52, "NT", "1 Thessalonians"),
    (53, "NT", "2 Thessalonians"),
    (54, "NT", "1 Timothy"),
    (55, "NT", "2 Timothy"),
    (56, "NT", "Titus"),
    (57, "NT", "Philemon"),
    (58, "NT", "Hebrews"),
    (59, "NT", "James"),
    (60, "NT", "1 Peter"),
    (61, "NT", "2 Peter"),
    (62, "NT", "1 John"),
    (63, "NT", "2 John"),
    (64, "NT", "3 John"),
    (65, "NT", "Jude"),
    (66, "NT", "Revelation"),
]

_CHAPTER_RE = re.compile(r'^(?:CHAPTER|PSALM)\s+(\d+)\s*$')
_VERSE_RE   = re.compile(r'^(\d+)\s+(.+)$')


def _is_all_caps(line: str) -> bool:
    """True if line has no lowercase letters — titles, headers, section markers."""
    alpha = [c for c in line if c.isalpha()]
    return bool(alpha) and all(c.isupper() for c in alpha)


def parse(txt: str):
    """
    Yield (book_idx, chapter, verse_num, text) tuples.

    Key insight: every book starts with 'CHAPTER 1', and there are exactly 66
    such lines in the KJV.  We use that as the book boundary instead of trying
    to detect all-caps header blocks (which is fragile).
    """
    book_idx   = -1   # -1 = before first book
    chapter    = 0
    next_verse = 1    # expected verse number for the next un-numbered line

    for raw in txt.splitlines():
        line = raw.rstrip()

        if not line:
            continue

        # --- chapter/book boundary ---
        m = _CHAPTER_RE.match(line)
        if m:
            ch_num = int(m.group(1))
            if ch_num == 1:
                book_idx += 1          # new book
            chapter    = ch_num
            next_verse = 1
            continue

        # --- skip all-caps header/title lines ---
        if _is_all_caps(line):
            continue

        # --- skip everything before the first book ---
        if book_idx < 0 or chapter == 0:
            continue

        # --- too far (safety) ---
        if book_idx >= len(BOOKS):
            continue

        # --- verse line ---
        m_v = _VERSE_RE.match(line)
        if m_v:
            v_num = int(m_v.group(1))
            text  = m_v.group(2).strip()
            next_verse = v_num + 1
        else:
            # verse 1 — no leading number in this edition
            v_num      = next_verse
            text       = line.strip()
            next_verse += 1

        if text:
            yield (book_idx, chapter, v_num, text)


def build(db_path: str, txt_path: str):
    # Download only if not already cached
    if not os.path.exists(txt_path):
        print(f"Downloading {SOURCE_URL} ...")
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as r:
            raw = r.read()
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            txt = raw.decode("latin-1")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Saved {len(txt):,} chars to {txt_path}")
    else:
        print(f"Using cached {txt_path}")
        with open(txt_path, encoding="utf-8") as f:
            txt = f.read()

    print("Building database …")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS verses_fts;
        DROP TABLE IF EXISTS verses;
        DROP TABLE IF EXISTS books;

        CREATE TABLE books (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            testament TEXT NOT NULL,
            chapters  INTEGER DEFAULT 0,
            verses    INTEGER DEFAULT 0,
            words     INTEGER DEFAULT 0
        );

        CREATE TABLE verses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id    INTEGER NOT NULL REFERENCES books(id),
            chapter    INTEGER NOT NULL,
            verse      INTEGER NOT NULL,
            text       TEXT NOT NULL,
            word_count INTEGER NOT NULL
        );
        CREATE INDEX idx_bcv ON verses(book_id, chapter, verse);

        CREATE VIRTUAL TABLE verses_fts USING fts5(
            text,
            book_id UNINDEXED,
            chapter UNINDEXED,
            verse   UNINDEXED,
            content=verses,
            content_rowid=id
        );
    """)

    for num, testament, name in BOOKS:
        conn.execute(
            "INSERT INTO books (id, name, testament) VALUES (?, ?, ?)",
            (num, name, testament),
        )

    verse_rows = []
    book_stats: dict = {}

    for book_idx, chapter, verse_num, text in parse(txt):
        book_id = BOOKS[book_idx][0]
        wc      = len(text.split())
        verse_rows.append((book_id, chapter, verse_num, text, wc))

        bs = book_stats.setdefault(book_id, {"chapters": set(), "verses": 0, "words": 0})
        bs["chapters"].add(chapter)
        bs["verses"] += 1
        bs["words"]  += wc

    conn.executemany(
        "INSERT INTO verses (book_id, chapter, verse, text, word_count) VALUES (?,?,?,?,?)",
        verse_rows,
    )

    for book_id, bs in book_stats.items():
        conn.execute(
            "UPDATE books SET chapters=?, verses=?, words=? WHERE id=?",
            (len(bs["chapters"]), bs["verses"], bs["words"], book_id),
        )

    print("Building FTS index …")
    conn.execute("""
        INSERT INTO verses_fts(rowid, text, book_id, chapter, verse)
        SELECT id, text, book_id, chapter, verse FROM verses
    """)

    conn.commit()
    conn.close()

    total      = len(verse_rows)
    n_books    = len(book_stats)
    expected   = 31102
    delta      = total - expected
    delta_str  = f"+{delta}" if delta >= 0 else str(delta)
    print(f"Done. {total:,} verses ({delta_str} vs expected {expected:,}) "
          f"across {n_books} books → {db_path}")
    if n_books != 66:
        print(f"WARNING: expected 66 books, got {n_books}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    build(
        db_path  = os.path.join(base, "kjv.db"),
        txt_path = os.path.join(base, "kjv.txt"),
    )
