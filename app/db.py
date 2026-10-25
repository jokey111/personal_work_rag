"""SQLite 初始化与连接（WAL 模式），建表语句集中在 SCHEMA。"""
import sqlite3
import threading
from contextlib import contextmanager

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL UNIQUE,
  description TEXT DEFAULT '',
  is_default  INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS intent_l1 (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  kb_id       INTEGER NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  description TEXT DEFAULT '',
  sort_order  INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now', 'localtime')),
  UNIQUE(kb_id, name)
);

CREATE TABLE IF NOT EXISTS intent_l2 (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  l1_id       INTEGER NOT NULL REFERENCES intent_l1(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  description TEXT DEFAULT '',
  sort_order  INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now', 'localtime')),
  UNIQUE(l1_id, name)
);

CREATE TABLE IF NOT EXISTS qa (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  l2_id            INTEGER NOT NULL REFERENCES intent_l2(id) ON DELETE CASCADE,
  question         TEXT NOT NULL,
  question_aliases TEXT DEFAULT '',
  answer           TEXT NOT NULL,
  follow_ups       TEXT DEFAULT '',
  star_hint        TEXT DEFAULT '',
  status           TEXT DEFAULT 'active',
  hit_count        INTEGER DEFAULT 0,
  last_asked_at    TEXT,
  quality          INTEGER DEFAULT 3,
  created_at       TEXT DEFAULT (datetime('now', 'localtime')),
  updated_at       TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS query_log (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  kb_id            INTEGER NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
  question         TEXT NOT NULL,
  l1_id            INTEGER,
  l2_id            INTEGER,
  route_confidence REAL,
  matched_qa_ids   TEXT DEFAULT '[]',
  feedback         TEXT DEFAULT 'pending',  -- pending/useful/useless
  source           TEXT DEFAULT 'match',    -- match/new
  created_at       TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_l2       ON qa(l2_id);
CREATE INDEX IF NOT EXISTS idx_l1_kb       ON intent_l1(kb_id);
CREATE INDEX IF NOT EXISTS idx_l2_l1       ON intent_l2(l1_id);
CREATE INDEX IF NOT EXISTS idx_log_kb_time ON query_log(kb_id, created_at);
CREATE INDEX IF NOT EXISTS idx_qa_question ON qa(question);
"""


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return conn


@contextmanager
def tx():
    """写事务：提交或回滚。"""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    # 默认知识库：面试FAQ（保持向后兼容：仅当表空时创建）
    count = conn.execute("SELECT COUNT(*) AS c FROM kb").fetchone()["c"]
    if count == 0:
        conn.execute(
            "INSERT INTO kb(name, description, is_default) VALUES(?,?,1)",
            ("面试FAQ", "个人面试问答知识库（默认）"),
        )
        conn.commit()
