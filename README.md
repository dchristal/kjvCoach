# kjvCoach

KJV Bible lookup and search — scripture only, no AI generation.

Built with FastAPI + SQLite full-text search. All results come directly from the King James Bible text; nothing is generated or paraphrased.

## Features

- **Full-text search** across all 31,000+ verses with relevance ranking
- **Book/chapter/verse lookup** by reference (e.g. `John 3:16`, `Psalms 23`)
- **Testament filter** — search Old Testament or New Testament only
- **Random verse**
- **Bible facts** — stats, records, and kjvcode.com numerical patterns

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 load.py       # downloads KJV text and builds kjv.db (~30 seconds)
./start.sh            # starts server on http://localhost:8002
```

`load.py` downloads the source text from kjvcode.com on first run and caches it as `kjv.txt`. Subsequent runs reuse the cached file.

## API

| Endpoint | Description |
|---|---|
| `GET /search?q=...` | Full-text search. Optional: `&testament=OT\|NT`, `&book=...`, `&limit=N` |
| `GET /verse/{book}/{chapter}/{verse}` | Single verse lookup |
| `GET /chapter/{book}/{chapter}` | All verses in a chapter |
| `GET /random` | Random verse. Optional: `&testament=OT\|NT` |
| `GET /books` | List all 66 books with metadata |
| `GET /stats` | Bible facts and statistics |

## Notes

- Database and source text (`kjv.db`, `kjv.txt`) are not committed — regenerate with `load.py`
- The KJV source uses `PSALM N` instead of `CHAPTER N` for the Psalter; the parser handles this correctly
- Single-word searches use FTS5 prefix matching, so e.g. `laodicea` also matches `Laodiceans`
