import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path("AI/data/news.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT,
  channel TEXT,
  message_id TEXT,
  date_utc TEXT,
  text TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_unique ON news(source, channel, message_id);
"""


@dataclass
class NewsItem:
    source: str
    channel: str
    message_id: str
    date_utc: str
    text: str


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


def insert_news(items: List[NewsItem]) -> int:
    if not items:
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        inserted = 0
        for it in items:
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO news(source, channel, message_id, date_utc, text) VALUES (?,?,?,?,?)",
                    (it.source, it.channel, it.message_id, it.date_utc, it.text),
                )
                inserted += cur.rowcount > 0
            except Exception:
                pass
        conn.commit()
    return inserted


def fetch_news(limit: int = 500) -> List[Tuple[str, str, str, str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT source, channel, message_id, date_utc, text FROM news ORDER BY date_utc DESC LIMIT ?", (limit,))
        return cur.fetchall() 