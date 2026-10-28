"""SQLite 初始化与连接（WAL 模式），建表语句集中在 SCHEMA。

v4 数据模型（分类/标签作为 QA 元数据 + 意图路由层）：
  category   二级分类树（parent_id 自引用，最多两级），QA 的组织元数据
  tag        扁平标签，QA 的横向元数据
  qa_tag     QA ↔ 标签 多对多
  intent     全局扁平意图列表（路由层，独立于分类）
  intent_kb  意图 ↔ 知识库 多对多挂载
  qa.kb_id   QA 直接归属知识库；qa.category_id 挂二级分类
老库（intent_l1/intent_l2/qa.l2_id）启动时自动迁移：L1→一级分类、L2→二级分类。
"""
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

CREATE TABLE IF NOT EXISTS intent (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  description TEXT DEFAULT '',
  sort_order  INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS intent_kb (
  intent_id INTEGER NOT NULL REFERENCES intent(id) ON DELETE CASCADE,
  kb_id     INTEGER NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
  UNIQUE(intent_id, kb_id)
);

CREATE TABLE IF NOT EXISTS category (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  kb_id       INTEGER NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
  parent_id   INTEGER REFERENCES category(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  sort_order  INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now', 'localtime')),
  UNIQUE(kb_id, parent_id, name)
);

CREATE TABLE IF NOT EXISTS tag (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL UNIQUE,
  created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS qa (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  kb_id            INTEGER NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
  category_id      INTEGER REFERENCES category(id) ON DELETE SET NULL,
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

CREATE TABLE IF NOT EXISTS qa_tag (
  qa_id  INTEGER NOT NULL REFERENCES qa(id) ON DELETE CASCADE,
  tag_id INTEGER NOT NULL REFERENCES tag(id) ON DELETE CASCADE,
  UNIQUE(qa_id, tag_id)
);

CREATE TABLE IF NOT EXISTS query_log (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  kb_id            INTEGER REFERENCES kb(id) ON DELETE SET NULL,
  question         TEXT NOT NULL,
  intent_id        INTEGER,
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

CREATE INDEX IF NOT EXISTS idx_qa_kb       ON qa(kb_id);
CREATE INDEX IF NOT EXISTS idx_intent_kb   ON intent_kb(intent_id);
CREATE INDEX IF NOT EXISTS idx_qa_question ON qa(question);
CREATE INDEX IF NOT EXISTS idx_qa_tag      ON qa_tag(tag_id);
CREATE INDEX IF NOT EXISTS idx_log_time    ON query_log(created_at);
"""

# 单列索引：依赖 qa.category_id 列（v3 老库补列之后才能建）
INDEX_QA_CATEGORY = "CREATE INDEX IF NOT EXISTS idx_qa_category ON qa(category_id);"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """老库迁移：intent_l1/intent_l2/qa.l2_id → intent/intent_kb/qa.kb_id + category。

    - L1 → 一级分类；L2 → L1 下的二级分类；qa.l2_id → category_id
    - 每个旧 L1 同时生成一个全局意图并挂载原库（保持路由层可用）
    """
    # ---- QA：l2_id -> kb_id + category_id（经 intent_l2→intent_l1 回溯）----
    if _table_exists(conn, "qa") and "l2_id" in _columns(conn, "qa"):
        conn.execute("ALTER TABLE qa RENAME TO qa_v2")
        conn.executescript(SCHEMA)  # 老表已改名，此刻重建新表与索引才是安全的

        # 分类：L1 → 一级，L2 → 二级
        for l1 in conn.execute(
                "SELECT DISTINCT l1.id, l1.kb_id, l1.name, l1.sort_order FROM intent_l1 l1 "
                "JOIN intent_l2 l2 ON l2.l1_id = l1.id ORDER BY l1.sort_order, l1.id").fetchall():
            cur = conn.execute(
                "INSERT INTO category(kb_id, parent_id, name, sort_order) VALUES(?,NULL,?,?)",
                (l1["kb_id"], l1["name"], l1["sort_order"]))
            parent_cat_id = cur.lastrowid
            for l2 in conn.execute(
                    "SELECT id, name, sort_order FROM intent_l2 WHERE l1_id=? ORDER BY sort_order, id",
                    (l1["id"],)).fetchall():
                conn.execute(
                    "INSERT INTO category(kb_id, parent_id, name, sort_order) VALUES(?,?,?,?)",
                    (l1["kb_id"], parent_cat_id, l2["name"], l2["sort_order"]))

        conn.execute(
            "INSERT INTO qa(id, kb_id, category_id, question, question_aliases, answer, follow_ups, "
            "star_hint, status, hit_count, last_asked_at, quality, created_at, updated_at) "
            "SELECT qa.id, l1.kb_id, cat.id, qa.question, qa.question_aliases, qa.answer, qa.follow_ups, "
            "qa.star_hint, qa.status, qa.hit_count, qa.last_asked_at, qa.quality, qa.created_at, qa.updated_at "
            "FROM qa_v2 qa JOIN intent_l2 l2 ON qa.l2_id=l2.id JOIN intent_l1 l1 ON l2.l1_id=l1.id "
            "JOIN category cat ON cat.parent_id IS NOT NULL AND cat.name=l2.name "
            "JOIN category pcat ON pcat.id=cat.parent_id AND pcat.name=l1.name AND pcat.kb_id=l1.kb_id")
        conn.execute("DROP TABLE qa_v2")

    # ---- 意图：每个旧 L1 → 一个全局意图（同名合并），挂到原库 ----
    if _table_exists(conn, "intent_l1"):
        for l1 in conn.execute("SELECT * FROM intent_l1 ORDER BY sort_order, id").fetchall():
            row = conn.execute("SELECT id FROM intent WHERE name=?", (l1["name"],)).fetchone()
            if row:
                intent_id = row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO intent(name, description, sort_order) VALUES(?,?,?)",
                    (l1["name"], l1["description"], l1["sort_order"]))
                intent_id = cur.lastrowid
            conn.execute("INSERT OR IGNORE INTO intent_kb(intent_id, kb_id) VALUES(?,?)",
                         (intent_id, l1["kb_id"]))

    # ---- query_log：补 intent_id 列（旧列 l1_id/l2_id 保留作历史数据）----
    if _table_exists(conn, "query_log") and "intent_id" not in _columns(conn, "query_log"):
        conn.execute("ALTER TABLE query_log ADD COLUMN intent_id INTEGER")

    # ---- 清理旧表 ----
    conn.execute("DROP TABLE IF EXISTS intent_l2")
    conn.execute("DROP TABLE IF EXISTS intent_l1")


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
    legacy = _table_exists(conn, "intent_l1") or (
        _table_exists(conn, "qa") and "l2_id" in _columns(conn, "qa"))
    if legacy:
        # 备份老库文件后再迁移
        try:
            import shutil
            shutil.copy(config.DB_PATH, config.DB_PATH + ".bak-v2")
        except Exception:
            pass
        _migrate_v2(conn)  # 迁移内部会在合适时机执行 SCHEMA（老 qa 表必须先改名）
    else:
        conn.executescript(SCHEMA)
        # v3 → v4：qa 表缺 category_id 时补列（CREATE TABLE IF NOT EXISTS 不会改已有表）
        if _table_exists(conn, "qa") and "category_id" not in _columns(conn, "qa"):
            conn.execute("ALTER TABLE qa ADD COLUMN category_id INTEGER REFERENCES category(id) ON DELETE SET NULL")
    conn.execute(INDEX_QA_CATEGORY)
    # 默认知识库：面试FAQ（保持向后兼容：仅当表空时创建）
    count = conn.execute("SELECT COUNT(*) AS c FROM kb").fetchone()["c"]
    if count == 0:
        conn.execute(
            "INSERT INTO kb(name, description, is_default) VALUES(?,?,1)",
            ("面试FAQ", "个人面试问答知识库（默认）"),
        )
        conn.commit()
