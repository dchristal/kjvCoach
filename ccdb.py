"""Parse the KJPBS .ccdb Bible database (zlib-compressed custom format).

Public API:
    load()  -> CCDBData
    CCDBData.word_count(word, case_sensitive=False) -> int
    CCDBData.word_forms(word) -> dict[form, count]
    CCDBData.raw_token_count() -> int
"""
from __future__ import annotations
import csv
import io
import zlib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT = Path(__file__).parent / "kjv1769.ccdb"


@dataclass
class CCDBData:
    # lowercase word -> {form: count, ...}
    _words: dict[str, dict[str, int]] = field(default_factory=dict)
    _raw_tokens: int = 0

    def word_forms(self, word: str) -> dict[str, int]:
        """Return {form: count} for all case variants of word."""
        return dict(self._words.get(word.lower(), {}))

    def word_count(self, word: str, case_sensitive: bool = False,
                   exact_form: str | None = None) -> int:
        """Total occurrences of word.

        exact_form: if given, count only that exact case (e.g. 'Amen', 'LORD').
        case_sensitive: ignored when exact_form is set.
        """
        entry = self._words.get(word.lower(), {})
        if not entry:
            return 0
        if exact_form is not None:
            return entry.get(exact_form, 0)
        if not case_sensitive:
            return sum(entry.values())
        # case_sensitive without exact_form: count forms that match word exactly
        return entry.get(word, 0)

    def raw_token_count(self) -> int:
        return self._raw_tokens


_cache: CCDBData | None = None


def load(path: Path = _DEFAULT) -> CCDBData:
    global _cache
    if _cache is not None:
        return _cache

    raw = path.read_bytes()
    text = zlib.decompress(raw).decode("utf-8", errors="replace")

    # Raw token count from source text (for 7^7 verification)
    # We split the raw text on whitespace — same as the kjvcode.com concordance file
    # The ccdb and the concordance file share the same token count
    concord = path.parent / "kjv.txt"
    raw_tokens = len(concord.read_text(encoding="utf-8").split()) if concord.exists() else 0

    # Parse WORDS section
    # Format per line: id, canonical, flags, total, "variants_csv", "per_variant_counts_csv", "positions_csv"
    words_idx = text.find("\nWORDS,")
    if words_idx < 0:
        raise ValueError("WORDS section not found in ccdb")

    word_dict: dict[str, dict[str, int]] = {}

    words_block = text[words_idx + 1:]
    end = words_block.find("\nWORDS,")  # shouldn't exist, but be safe
    lines = words_block.split("\n")[1:]  # skip "WORDS,N" header

    csv.field_size_limit(10_000_000)
    reader = csv.reader(io.StringIO("\n".join(lines)), quotechar='"')
    for row in reader:
        if len(row) < 6:
            continue
        try:
            int(row[0])  # word id
        except ValueError:
            break

        canonical = row[1].strip()
        try:
            total = int(row[3])
        except ValueError:
            continue

        variants = [v.strip() for v in row[4].split(",") if v.strip()]
        try:
            counts = [int(c) for c in row[5].split(",") if c.strip()]
        except ValueError:
            counts = []

        forms: dict[str, int] = {}
        if variants and counts and len(variants) == len(counts):
            for v, c in zip(variants, counts):
                forms[v] = c
        else:
            forms[canonical] = total

        word_dict[canonical.lower()] = forms

    _cache = CCDBData(_words=word_dict, _raw_tokens=raw_tokens)
    return _cache
