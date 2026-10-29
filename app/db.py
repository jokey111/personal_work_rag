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

CREATE TABLE IF NOT EXISTS qa_embedding (
  qa_id     INTEGER PRIMARY KEY REFERENCES qa(id) ON DELETE CASCADE,
  model     TEXT NOT NULL,             -- 生成向量的 embedding 模型名（换模型可检测重刷）
  dim       INTEGER NOT NULL,
  vec       BLOB NOT NULL,             -- float32 小端序列化
  text_hash TEXT NOT NULL,             -- 问题+别名哈希，编辑后检测需重算
  updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_qa_kb       ON qa(kb_id);
CREATE INDEX IF NOT EXISTS idx_intent_kb   ON intent_kb(intent_id);
CREATE INDEX IF NOT EXISTS idx_qa_question ON qa(question);
CREATE INDEX IF NOT EXISTS idx_qa_tag      ON qa_tag(tag_id);
CREATE INDEX IF NOT EXISTS idx_log_time    ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_qa_emb_model ON qa_embedding(model);
"""

# 单列索引：依赖 qa.category_id 列（v3 老库补列之后才能建）
INDEX_QA_CATEGORY = "CREATE INDEX IF NOT EXISTS idx_qa_category ON qa(category_id);"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _migrate_v2_qa(conn: sqlite3.Connection) -> None:
    """把 qa_v2（含 l2_id 的老表）搬进 v4 qa 表：建分类映射 + 逐行搬运 + 兜底。

    覆盖两种入口状态：
    a) 完整 v2 库（intent_l1/intent_l2 存在）
    b) 半迁移库（qa_v2 存在、v4 qa 表已建；intent_l1/l2 可能仍在）
    幂等：qa 表已有同 id 的行跳过（INSERT OR IGNORE）。
    """
    # 1) 分类映射：L1→一级分类、L2→二级分类（同名已存在则复用）
    l2_to_cat: dict[int, int] = {}
    if _table_exists(conn, "intent_l1") and _table_exists(conn, "intent_l2"):
        for l1 in conn.execute(
                "SELECT DISTINCT l1.id, l1.kb_id, l1.name, l1.sort_order FROM intent_l1 l1 "
                "JOIN intent_l2 l2 ON l2.l1_id = l1.id ORDER BY l1.sort_order, l1.id").fetchall():
            row = conn.execute(
                "SELECT id FROM category WHERE kb_id=? AND parent_id IS NULL AND name=?",
                (l1["kb_id"], l1["name"])).fetchone()
            if row:
                parent_cat_id = row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO category(kb_id, parent_id, name, sort_order) VALUES(?,NULL,?,?)",
                    (l1["kb_id"], l1["name"], l1["sort_order"]))
                parent_cat_id = cur.lastrowid
            for l2 in conn.execute(
                    "SELECT id, name, sort_order FROM intent_l2 WHERE l1_id=? ORDER BY sort_order, id",
                    (l1["id"],)).fetchall():
                crow = conn.execute(
                    "SELECT id FROM category WHERE kb_id=? AND parent_id=? AND name=?",
                    (l1["kb_id"], parent_cat_id, l2["name"])).fetchone()
                if crow:
                    l2_to_cat[l2["id"]] = crow["id"]
                else:
                    cur = conn.execute(
                        "INSERT INTO category(kb_id, parent_id, name, sort_order) VALUES(?,?,?,?)",
                        (l1["kb_id"], parent_cat_id, l2["name"], l2["sort_order"]))
                    l2_to_cat[l2["id"]] = cur.lastrowid

    # 2) 逐行搬运：优先 l2→分类映射；映射不到经 intent 表回溯库；再不行进默认库、分类置空
    default_kb = conn.execute("SELECT id FROM kb ORDER BY is_default DESC, id LIMIT 1").fetchone()
    default_kb_id = default_kb["id"] if default_kb else None

    for q in conn.execute("SELECT * FROM qa_v2 ORDER BY id").fetchall():
        kb_id, category_id = None, None
        if q["l2_id"] is not None:
            cat_id = l2_to_cat.get(q["l2_id"])
            if cat_id:
                category_id = cat_id
                cat_row = conn.execute("SELECT kb_id FROM category WHERE id=?", (cat_id,)).fetchone()
                kb_id = cat_row["kb_id"] if cat_row else None
            elif _table_exists(conn, "intent_l2"):
                row = conn.execute(
                    "SELECT l1.kb_id FROM intent_l2 l2 JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l2.id=?",
                    (q["l2_id"],)).fetchone()
                kb_id = row["kb_id"] if row else None
        if kb_id is None:
            kb_id = default_kb_id
        conn.execute(
            "INSERT OR IGNORE INTO qa(id, kb_id, category_id, question, question_aliases, answer, follow_ups, "
            "star_hint, status, hit_count, last_asked_at, quality, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (q["id"], kb_id, category_id, q["question"], q["question_aliases"], q["answer"],
             q["follow_ups"], q["star_hint"], q["status"], q["hit_count"], q["last_asked_at"],
             q["quality"], q["created_at"], q["updated_at"]))
    # qa_v2 保留作归档（不 DROP），数据核对无误后可手动清理


def _migrate_v2_intents(conn: sqlite3.Connection) -> None:
    """每个旧 L1 → 一个全局意图（同名合并），挂到原库。"""
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


def _backup_db() -> None:
    """迁移前备份：先 checkpoint WAL，再拷贝主文件，保证备份含全部已提交数据。"""
    try:
        conn = get_conn()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        import shutil
        import time as _time
        stamp = _time.strftime("%Y%m%d-%H%M%S")
        shutil.copy(config.DB_PATH, f"{config.DB_PATH}.bak-{stamp}")
    except Exception as e:
        print(f"[db] 迁移前备份失败（继续执行迁移）: {e}")


def init_db() -> None:
    conn = get_conn()
    v2_qa = _table_exists(conn, "qa") and "l2_id" in _columns(conn, "qa")
    if v2_qa:
        # 老 v2 qa 表结构与 v4 SCHEMA 索引冲突，必须先改名再执行 SCHEMA
        _backup_db()
        conn.execute("ALTER TABLE qa RENAME TO qa_v2")
    conn.executescript(SCHEMA)  # 全部 IF NOT EXISTS：对已有表无副作用，对缺失表补建

    half_migrated = _table_exists(conn, "qa_v2")
    legacy_intents = _table_exists(conn, "intent_l1")
    if v2_qa or half_migrated or legacy_intents:
        if not v2_qa:  # v2_qa 场景在上面已备份过
            _backup_db()
        if half_migrated:
            # v2 搬运（含 half_migrated 自愈：qa_v2 存在、新 qa 可能已有部分数据）
            _migrate_v2_qa(conn)
        if legacy_intents:
            _migrate_v2_intents(conn)
        # 清理旧结构表（qa_v2 保留作归档，方便人工核对；确认无误后可手动 DROP）
        conn.execute("DROP TABLE IF EXISTS intent_l2")
        conn.execute("DROP TABLE IF EXISTS intent_l1")
        conn.commit()

    # v3 → v4：qa 表缺 category_id 时补列（CREATE TABLE IF NOT EXISTS 不会改已有表）
    if _table_exists(conn, "qa") and "category_id" not in _columns(conn, "qa"):
        conn.execute("ALTER TABLE qa ADD COLUMN category_id INTEGER "
                     "REFERENCES category(id) ON DELETE SET NULL")
    conn.execute(INDEX_QA_CATEGORY)

    # query_log：补 intent_id 列（历史 l1_id/l2_id 列保留）
    if _table_exists(conn, "query_log") and "intent_id" not in _columns(conn, "query_log"):
        conn.execute("ALTER TABLE query_log ADD COLUMN intent_id INTEGER")

    conn.commit()
    # 默认知识库：面试FAQ（仅当表空时创建）
    count = conn.execute("SELECT COUNT(*) AS c FROM kb").fetchone()["c"]
    if count == 0:
        conn.execute(
            "INSERT INTO kb(name, description, is_default) VALUES(?,?,1)",
            ("面试FAQ", "个人面试问答知识库（默认）"),
        )
        conn.commit()
