"""FastAPI 主应用：认证 + 知识库/意图/QA CRUD + 匹配 + 复盘统计。"""
import asyncio
import json
import secrets
import time
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import config, db, embedding, llm, md_import

app = FastAPI(title="personal_work_rag", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 单人自用，前端同机部署由 Nginx 反代
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)


# ---------- 认证 ----------
def make_token() -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": config.ADMIN_USERNAME, "exp": now + timedelta(hours=config.JWT_EXPIRE_HOURS)}
    return jwt.encode(payload, config.JWT_SECRET, algorithm="HS256")


def auth_dep(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None:
        raise HTTPException(401, "未登录")
    try:
        jwt.decode(cred.credentials, config.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "登录已过期，请重新登录")


class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginIn):
    if not secrets.compare_digest(body.username, config.ADMIN_USERNAME) or not secrets.compare_digest(
        body.password, config.ADMIN_PASSWORD
    ):
        raise HTTPException(401, "用户名或密码错误")
    return {"token": make_token(), "username": body.username}


@app.get("/api/health")
def health():
    return {"ok": True, "app": config.APP_NAME, "version": "0.2.0"}


# ---------- 知识库 ----------
class KBIn(BaseModel):
    name: str
    description: str = ""
    is_default: bool = False


@app.get("/api/kbs", dependencies=[Depends(auth_dep)])
def list_kbs():
    rows = db.get_conn().execute("SELECT * FROM kb ORDER BY is_default DESC, id").fetchall()
    out = []
    for r in rows:
        kb = dict(r)
        kb["qa_count"] = db.get_conn().execute(
            "SELECT COUNT(*) AS c FROM qa WHERE kb_id=? AND status='active'", (kb["id"],),
        ).fetchone()["c"]
        out.append(kb)
    return out


@app.post("/api/kbs", dependencies=[Depends(auth_dep)])
def create_kb(body: KBIn):
    with db.tx() as conn:
        if body.is_default:
            conn.execute("UPDATE kb SET is_default=0")
        cur = conn.execute(
            "INSERT INTO kb(name, description, is_default) VALUES(?,?,?)",
            (body.name.strip(), body.description.strip(), int(body.is_default)),
        )
        kb_id = cur.lastrowid
    return {"id": kb_id}


@app.put("/api/kbs/{kb_id}", dependencies=[Depends(auth_dep)])
def update_kb(kb_id: int, body: KBIn):
    with db.tx() as conn:
        if body.is_default:
            conn.execute("UPDATE kb SET is_default=0")
        cur = conn.execute(
            "UPDATE kb SET name=?, description=?, is_default=? WHERE id=?",
            (body.name.strip(), body.description.strip(), int(body.is_default), kb_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "知识库不存在")
    return {"ok": True}


@app.delete("/api/kbs/{kb_id}", dependencies=[Depends(auth_dep)])
def delete_kb(kb_id: int):
    conn = db.get_conn()
    row = conn.execute("SELECT COUNT(*) AS c FROM kb").fetchone()
    if row["c"] <= 1:
        raise HTTPException(400, "至少保留一个知识库")
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM kb WHERE id=?", (kb_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "知识库不存在")
    return {"ok": True}


# ---------- 意图（全局扁平列表，多对多挂载知识库） ----------
class IntentIn(BaseModel):
    name: str
    description: str = ""
    sort_order: int = 0
    kb_ids: list[int] = []


def _intent_row(intent_id: int) -> dict:
    row = db.get_conn().execute("SELECT * FROM intent WHERE id=?", (intent_id,)).fetchone()
    if not row:
        raise HTTPException(404, "意图不存在")
    return dict(row)


def _attach_kb_ids(conn, intent: dict) -> dict:
    intent["kb_ids"] = [r["kb_id"] for r in conn.execute(
        "SELECT kb_id FROM intent_kb WHERE intent_id=? ORDER BY kb_id", (intent["id"],)).fetchall()]
    return intent


@app.get("/api/intents", dependencies=[Depends(auth_dep)])
def get_intents():
    conn = db.get_conn()
    intents = [_attach_kb_ids(conn, dict(r)) for r in
               conn.execute("SELECT * FROM intent ORDER BY sort_order, id").fetchall()]
    for it in intents:
        if it["kb_ids"]:
            marks = ",".join("?" * len(it["kb_ids"]))
            it["qa_count"] = conn.execute(
                f"SELECT COUNT(*) AS c FROM qa WHERE kb_id IN ({marks}) AND status='active'",
                it["kb_ids"]).fetchone()["c"]
        else:
            it["qa_count"] = 0
    return intents


@app.post("/api/intents", dependencies=[Depends(auth_dep)])
def create_intent(body: IntentIn):
    with db.tx() as conn:
        cur = conn.execute(
            "INSERT INTO intent(name, description, sort_order) VALUES(?,?,?)",
            (body.name.strip(), body.description.strip(), body.sort_order))
        intent_id = cur.lastrowid
        for kb_id in body.kb_ids:
            conn.execute("INSERT OR IGNORE INTO intent_kb(intent_id, kb_id) VALUES(?,?)",
                         (intent_id, kb_id))
    return {"id": intent_id}


@app.put("/api/intents/{intent_id}", dependencies=[Depends(auth_dep)])
def update_intent(intent_id: int, body: IntentIn):
    _intent_row(intent_id)
    with db.tx() as conn:
        conn.execute(
            "UPDATE intent SET name=?, description=?, sort_order=? WHERE id=?",
            (body.name.strip(), body.description.strip(), body.sort_order, intent_id))
        conn.execute("DELETE FROM intent_kb WHERE intent_id=?", (intent_id,))
        for kb_id in body.kb_ids:
            conn.execute("INSERT OR IGNORE INTO intent_kb(intent_id, kb_id) VALUES(?,?)",
                         (intent_id, kb_id))
    return {"ok": True}


@app.delete("/api/intents/{intent_id}", dependencies=[Depends(auth_dep)])
def delete_intent(intent_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM intent WHERE id=?", (intent_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "意图不存在")
    return {"ok": True}


# ---------- 分类（二级树，QA 元数据） ----------
class CategoryIn(BaseModel):
    kb_id: int
    parent_id: int | None = None
    name: str
    sort_order: int = 0


def _category_children(conn, parent_id: int | None) -> list[int]:
    """收集 parent_id 及其全部后代 id（最多再一层）。"""
    ids = [parent_id] if parent_id else []
    if parent_id:
        ids += [r["id"] for r in conn.execute(
            "SELECT id FROM category WHERE parent_id=?", (parent_id,)).fetchall()]
    return ids


@app.get("/api/categories", dependencies=[Depends(auth_dep)])
def list_categories(kb_id: int):
    """返回二级分类树：[{...一级, children:[二级]}, ...]，带 qa_count。"""
    conn = db.get_conn()
    cats = [dict(r) for r in conn.execute(
        "SELECT * FROM category WHERE kb_id=? ORDER BY sort_order, id", (kb_id,)).fetchall()]
    by_parent: dict[int | None, list] = {}
    for c in cats:
        c["qa_count"] = conn.execute(
            "SELECT COUNT(*) AS c FROM qa WHERE category_id=? AND status='active'", (c["id"],)
        ).fetchone()["c"]
        by_parent.setdefault(c["parent_id"], []).append(c)
    for c in cats:
        c["children"] = by_parent.get(c["id"], [])
    return by_parent.get(None, [])


def _dup_category_name(conn, kb_id: int, parent_id: int | None, name: str, exclude_id: int | None = None) -> bool:
    sql = "SELECT 1 FROM category WHERE kb_id=? AND name=? AND parent_id IS ?"
    args: list = [kb_id, name, parent_id]
    if exclude_id:
        sql += " AND id<>?"
        args.append(exclude_id)
    return conn.execute(sql + " LIMIT 1", args).fetchone() is not None


@app.post("/api/categories", dependencies=[Depends(auth_dep)])
def create_category(body: CategoryIn):
    if body.parent_id is not None:
        parent = db.get_conn().execute(
            "SELECT * FROM category WHERE id=?", (body.parent_id,)).fetchone()
        if not parent:
            raise HTTPException(404, "父分类不存在")
        if parent["parent_id"] is not None:
            raise HTTPException(400, "分类最多两级，不能在二级分类下再建子分类")
        if parent["kb_id"] != body.kb_id:
            raise HTTPException(400, "父分类属于其他知识库")
    if _dup_category_name(db.get_conn(), body.kb_id, body.parent_id, body.name.strip()):
        raise HTTPException(400, "同级已存在同名分类")
    try:
        with db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO category(kb_id, parent_id, name, sort_order) VALUES(?,?,?,?)",
                (body.kb_id, body.parent_id, body.name.strip(), body.sort_order))
            return {"id": cur.lastrowid}
    except Exception:
        raise HTTPException(400, "创建失败：同级可能已存在同名分类")


@app.put("/api/categories/{cat_id}", dependencies=[Depends(auth_dep)])
def update_category(cat_id: int, body: CategoryIn):
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM category WHERE id=?", (cat_id,)).fetchone()
    if not row:
        raise HTTPException(404, "分类不存在")
    if body.parent_id is not None:
        if body.parent_id == cat_id:
            raise HTTPException(400, "父分类不能是自己")
        parent = conn.execute("SELECT * FROM category WHERE id=?", (body.parent_id,)).fetchone()
        if not parent:
            raise HTTPException(404, "父分类不存在")
        if parent["parent_id"] is not None:
            raise HTTPException(400, "分类最多两级")
        if parent["kb_id"] != body.kb_id:
            raise HTTPException(400, "父分类属于其他知识库")
    # 二级降为一级时，其子分类一并上提为一级（挂到其 kb 下）
    if row["parent_id"] is not None and body.parent_id is None:
        with db.tx() as txn:
            txn.execute("UPDATE category SET parent_id=NULL, kb_id=? WHERE parent_id=?",
                        (row["kb_id"], cat_id))
            txn.execute(
                "UPDATE category SET kb_id=?, parent_id=?, name=?, sort_order=? WHERE id=?",
                (body.kb_id, body.parent_id, body.name.strip(), body.sort_order, cat_id))
        return {"ok": True}
    if _dup_category_name(db.get_conn(), body.kb_id, body.parent_id, body.name.strip(), exclude_id=cat_id):
        raise HTTPException(400, "同级已存在同名分类")
    try:
        with db.tx() as txn:
            cur = txn.execute(
                "UPDATE category SET kb_id=?, parent_id=?, name=?, sort_order=? WHERE id=?",
                (body.kb_id, body.parent_id, body.name.strip(), body.sort_order, cat_id))
            if cur.rowcount == 0:
                raise HTTPException(404, "分类不存在")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "更新失败：同级可能已存在同名分类")
    return {"ok": True}


@app.delete("/api/categories/{cat_id}", dependencies=[Depends(auth_dep)])
def delete_category(cat_id: int):
    """删除分类：子分类一并删除；其下 QA 的 category_id 置空（QA 本身保留）。"""
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM category WHERE id=?", (cat_id,)).fetchone()
    if not row:
        raise HTTPException(404, "分类不存在")
    with db.tx() as txn:
        txn.execute("UPDATE qa SET category_id=NULL WHERE category_id IN "
                    "(SELECT id FROM category WHERE parent_id=?)", (cat_id,))
        txn.execute("UPDATE qa SET category_id=NULL WHERE category_id=?", (cat_id,))
        txn.execute("DELETE FROM category WHERE parent_id=?", (cat_id,))
        txn.execute("DELETE FROM category WHERE id=?", (cat_id,))
    return {"ok": True}


@app.post("/api/categories/{cat_id}/move", dependencies=[Depends(auth_dep)])
def move_category(cat_id: int, direction: str):
    """同级内上移/下移（direction: up/down），交换 sort_order 并归一化。"""
    if direction not in ("up", "down"):
        raise HTTPException(400, "direction 只能是 up 或 down")
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM category WHERE id=?", (cat_id,)).fetchone()
    if not row:
        raise HTTPException(404, "分类不存在")
    # 同级按当前排序规则取邻居
    siblings = conn.execute(
        "SELECT id FROM category WHERE kb_id=? AND parent_id IS ? "
        "ORDER BY sort_order, id",
        (row["kb_id"], row["parent_id"])).fetchall()
    ids = [r["id"] for r in siblings]
    idx = ids.index(cat_id)
    swap_idx = idx - 1 if direction == "up" else idx + 1
    if swap_idx < 0 or swap_idx >= len(ids):
        return {"ok": True, "moved": False}  # 已在边界
    with db.tx() as txn:
        # 归一化同级 sort_order = 数组下标，再交换目标与邻居
        for i, cid in enumerate(ids):
            txn.execute("UPDATE category SET sort_order=? WHERE id=?", (i, cid))
        txn.execute("UPDATE category SET sort_order=? WHERE id=?", (idx, ids[swap_idx]))
        txn.execute("UPDATE category SET sort_order=? WHERE id=?", (swap_idx, cat_id))
    return {"ok": True, "moved": True}


# ---------- 标签（扁平，QA 元数据，全局） ----------
class TagIn(BaseModel):
    name: str


@app.get("/api/tags", dependencies=[Depends(auth_dep)])
def list_tags(kb_id: int | None = None):
    """标签列表（带使用次数）；kb_id 给定时只统计该库的使用数。"""
    conn = db.get_conn()
    if kb_id:
        rows = conn.execute(
            "SELECT t.id, t.name, COUNT(qt.qa_id) AS usage_count FROM tag t "
            "LEFT JOIN qa_tag qt ON qt.tag_id=t.id "
            "LEFT JOIN qa ON qa.id=qt.qa_id AND qa.kb_id=? "
            "GROUP BY t.id HAVING (usage_count IS NOT NULL) OR COUNT(qt.qa_id)=0 "
            "ORDER BY usage_count IS NULL, usage_count DESC, t.id", (kb_id,)).fetchall()
        # 上面的写法容易把计数搞乱，改为简洁口径：usage = 该库内挂此标签的 QA 数
        rows = conn.execute(
            "SELECT t.id, t.name, "
            "(SELECT COUNT(*) FROM qa_tag qt JOIN qa ON qa.id=qt.qa_id "
            " WHERE qt.tag_id=t.id AND qa.kb_id=?) AS usage_count "
            "FROM tag t ORDER BY usage_count DESC, t.id", (kb_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT t.id, t.name, COUNT(qt.qa_id) AS usage_count FROM tag t "
            "LEFT JOIN qa_tag qt ON qt.tag_id=t.id "
            "GROUP BY t.id ORDER BY usage_count DESC, t.id").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/tags", dependencies=[Depends(auth_dep)])
def create_tag(body: TagIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "标签名不能为空")
    row = db.get_conn().execute("SELECT id FROM tag WHERE name=?", (name,)).fetchone()
    if row:
        return {"id": row["id"], "existed": True}
    with db.tx() as conn:
        cur = conn.execute("INSERT INTO tag(name) VALUES(?)", (name,))
        return {"id": cur.lastrowid}


@app.put("/api/tags/{tag_id}", dependencies=[Depends(auth_dep)])
def update_tag(tag_id: int, body: TagIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "标签名不能为空")
    try:
        with db.tx() as conn:
            cur = conn.execute("UPDATE tag SET name=? WHERE id=?", (name, tag_id))
            if cur.rowcount == 0:
                raise HTTPException(404, "标签不存在")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "更新失败：可能已存在同名标签")
    return {"ok": True}


@app.delete("/api/tags/{tag_id}", dependencies=[Depends(auth_dep)])
def delete_tag(tag_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM tag WHERE id=?", (tag_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "标签不存在")
    return {"ok": True}


# ---------- QA ----------
class QAIn(BaseModel):
    kb_id: int
    category_id: int | None = None
    tag_ids: list[int] = []
    question: str
    question_aliases: str = ""
    answer: str
    # 以下字段已从表单移除（保留列存历史数据；前端不传时 create 用默认值、update 不覆盖）
    follow_ups: str | None = None
    star_hint: str | None = None
    status: str | None = None
    quality: int | None = None


def _save_qa_tags(conn, qa_id: int, tag_ids: list[int]):
    conn.execute("DELETE FROM qa_tag WHERE qa_id=?", (qa_id,))
    for tid in set(tag_ids):
        conn.execute("INSERT OR IGNORE INTO qa_tag(qa_id, tag_id) VALUES(?,?)", (qa_id, tid))


def _qa_tags(conn, qa_ids: list[int]) -> dict[int, list[dict]]:
    if not qa_ids:
        return {}
    marks = ",".join("?" * len(qa_ids))
    out: dict[int, list] = {}
    for r in conn.execute(
            f"SELECT qt.qa_id, t.id, t.name FROM qa_tag qt JOIN tag t ON qt.tag_id=t.id "
            f"WHERE qt.qa_id IN ({marks}) ORDER BY t.name", qa_ids).fetchall():
        out.setdefault(r["qa_id"], []).append({"id": r["id"], "name": r["name"]})
    return out


@app.get("/api/qas", dependencies=[Depends(auth_dep)])
def list_qas(kb_id: int, q: str = "", status: str = "",
             category_id: int | None = None, tag_id: int | None = None):
    """QA 列表：支持关键词 / 状态 / 分类（含子分类）/ 标签筛选，返回分类与标签。"""
    sql = ("SELECT qa.*, kb.name AS kb_name, "
           "CASE WHEN c1.parent_id IS NULL THEN c1.name ELSE c2.name END AS category_l1, "
           "CASE WHEN c1.parent_id IS NULL THEN NULL ELSE c1.name END AS category_l2, "
           "CASE WHEN c1.parent_id IS NULL THEN c1.id ELSE c2.id END AS top_category_id "
           "FROM qa JOIN kb ON qa.kb_id=kb.id "
           "LEFT JOIN category c1 ON qa.category_id=c1.id "
           "LEFT JOIN category c2 ON c1.parent_id=c2.id WHERE qa.kb_id=?")
    args: list = [kb_id]
    if category_id:
        sql += " AND (qa.category_id=? OR qa.category_id IN (SELECT id FROM category WHERE parent_id=?))"
        args += [category_id, category_id]
    if tag_id:
        sql += " AND qa.id IN (SELECT qa_id FROM qa_tag WHERE tag_id=?)"
        args.append(tag_id)
    if q:
        sql += " AND (qa.question LIKE ? OR qa.question_aliases LIKE ? OR qa.answer LIKE ?)"
        like = f"%{q}%"
        args += [like, like, like]
    if status:
        sql += " AND qa.status=?"
        args.append(status)
    sql += " ORDER BY qa.updated_at DESC LIMIT 500"
    rows = [dict(r) for r in db.get_conn().execute(sql, args).fetchall()]
    tags = _qa_tags(db.get_conn(), [r["id"] for r in rows])
    for r in rows:
        r["tags"] = tags.get(r["id"], [])
    return rows


@app.post("/api/qas", dependencies=[Depends(auth_dep)])
def create_qa(body: QAIn):
    with db.tx() as conn:
        cur = conn.execute(
            "INSERT INTO qa(kb_id, category_id, question, question_aliases, answer, follow_ups, star_hint, status, quality) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (body.kb_id, body.category_id, body.question.strip(), body.question_aliases.strip(),
             body.answer.strip(), body.follow_ups or "", body.star_hint or "",
             body.status or "active", body.quality if body.quality is not None else 3),
        )
        qa_id = cur.lastrowid
        _save_qa_tags(conn, qa_id, body.tag_ids)
    _sync_qa_embedding(qa_id)
    return {"id": qa_id}


@app.put("/api/qas/{qa_id}", dependencies=[Depends(auth_dep)])
def update_qa(qa_id: int, body: QAIn):
    with db.tx() as conn:
        # 表单移除的字段不再更新（保留库中原值）
        cur = conn.execute(
            "UPDATE qa SET kb_id=?, category_id=?, question=?, question_aliases=?, answer=?, "
            "updated_at=datetime('now','localtime') WHERE id=?",
            (body.kb_id, body.category_id, body.question.strip(), body.question_aliases.strip(),
             body.answer.strip(), qa_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "QA 不存在")
        _save_qa_tags(conn, qa_id, body.tag_ids)
    _sync_qa_embedding(qa_id)
    return {"ok": True}


@app.delete("/api/qas/{qa_id}", dependencies=[Depends(auth_dep)])
def delete_qa(qa_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM qa WHERE id=?", (qa_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "QA 不存在")
    embedding.store.remove(qa_id)  # 级联删了 qa_embedding 行，这里同步内存
    return {"ok": True}


def _sync_qa_embedding(qa_id: int) -> None:
    """QA 写入后同步向量（同步阻塞，单条毫秒级；embedding 不可用时静默跳过，
    检索侧 ensure_ready 会惰性补齐）。"""
    import asyncio as _aio

    async def _do():
        try:
            row = db.get_conn().execute(
                "SELECT id, question, question_aliases, status FROM qa WHERE id=?",
                (qa_id,)).fetchone()
            if row is None or row["status"] != "active":
                embedding.store.remove(qa_id)
                return
            vec = (await embedding.embed_async([embedding.text_of(dict(row))]))[0]
            embedding.store.upsert(qa_id, vec, embedding._hash(embedding.text_of(dict(row))))
        except embedding.EmbeddingError:
            pass  # API 挂了不阻塞 CRUD；下次检索时 ensure_ready 补

    try:
        loop = _aio.get_running_loop()
        loop.create_task(_do())
    except RuntimeError:
        _aio.run(_do())


class ImportIn(BaseModel):
    items: list[QAIn]


@app.post("/api/qas/import", dependencies=[Depends(auth_dep)])
def import_qas(body: ImportIn):
    ok_count = 0
    for item in body.items:
        try:
            create_qa(item)
            ok_count += 1
        except HTTPException:
            continue
    return {"imported": ok_count, "total": len(body.items)}


@app.get("/api/qas/export", dependencies=[Depends(auth_dep)])
def export_qas(kb_id: int):
    return list_qas(kb_id=kb_id)


class MarkdownImportIn(BaseModel):
    kb_id: int
    markdown: str
    dry_run: bool = False  # True 只解析校验不落库


@app.post("/api/qas/import-markdown", dependencies=[Depends(auth_dep)])
def import_markdown(body: MarkdownImportIn):
    """Markdown 批量导入：dry_run=True 仅预览解析结果。"""
    parsed = md_import.parse_markdown(body.markdown)
    if body.dry_run:
        preview = [
            {"l1": s["name"], "description": s["description"], "count": len(s["questions"]),
             "questions": [q["question"] for q in s["questions"][:5]]}
            for s in parsed["sections"]
        ]
        return {"dry_run": True, "sections": preview, "errors": parsed["errors"], "total": parsed["total"]}
    result = md_import.import_parsed(body.kb_id, parsed)
    return result


# ---------- 匹配 ----------
class MatchIn(BaseModel):
    kb_id: int
    question: str
    top_k: int | None = None       # 覆盖 default_top_k
    min_score: float | None = None  # 覆盖 match_min_score（仅本次）


# ---------- 匹配参数（settings 表运行时配置） ----------
SETTING_SPECS = {  # key: (类型, 下限, 上限)
    "match_min_score": (float, 0.0, 1.0),
    "route_min_score": (float, 0.0, 1.0),
    "keyword_match_min_score": (float, 0.0, 1.0),
    "keyword_direct_score": (float, 0.0, 1.0),
    "vector_match_score": (float, 0.0, 1.0),
    "vector_fallback_score": (float, 0.0, 1.0),
    "vector_top_k": (int, 1, 10),
    "default_top_k": (int, 1, 10),
}
_SETTING_DEFAULTS = {
    "match_min_score": lambda: config.MATCH_MIN_SCORE,
    "route_min_score": lambda: config.ROUTE_MIN_SCORE,
    "keyword_match_min_score": lambda: config.KEYWORD_MATCH_MIN_SCORE,
    "keyword_direct_score": lambda: config.KEYWORD_DIRECT_SCORE,
    "vector_match_score": lambda: config.VECTOR_MATCH_SCORE,
    "vector_fallback_score": lambda: config.VECTOR_FALLBACK_SCORE,
    "vector_top_k": lambda: config.VECTOR_TOP_K,
    "default_top_k": lambda: 3,
}


def get_settings() -> dict:
    rows = {r["key"]: r["value"] for r in
            db.get_conn().execute("SELECT key, value FROM settings").fetchall()}
    out = {}
    for key, (typ, lo, hi) in SETTING_SPECS.items():
        try:
            v = typ(rows[key]) if key in rows else typ(_SETTING_DEFAULTS[key]())
        except (TypeError, ValueError):
            v = typ(_SETTING_DEFAULTS[key]())
        out[key] = max(lo, min(hi, v))
    out["answer_mode"] = rows.get("answer_mode", "strict")  # strict/loose
    return out


@app.get("/api/settings", dependencies=[Depends(auth_dep)])
def read_settings():
    return get_settings()


class SettingsIn(BaseModel):
    values: dict[str, float | int | str]


def _valid_answer_mode(v: str) -> bool:
    return v in ("strict", "loose")


@app.put("/api/settings", dependencies=[Depends(auth_dep)])
def update_settings(body: SettingsIn):
    for key, value in body.values.items():
        if key == "answer_mode":
            if not _valid_answer_mode(str(value)):
                raise HTTPException(400, "answer_mode 只能是 strict 或 loose")
            with db.tx() as conn:
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
            continue
        if key not in SETTING_SPECS:
            raise HTTPException(400, f"未知参数: {key}")
        typ, lo, hi = SETTING_SPECS[key]
        try:
            value = typ(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} 值非法")
        if not (lo <= value <= hi):
            raise HTTPException(400, f"{key} 需在 {lo}~{hi} 之间")
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    return get_settings()


# ---------- 匹配（三级漏斗：向量0.72 → 关键词 → 向量兜底0.55+LLM核验 → 严格/宽松） ----------
async def _run_match(body: MatchIn, emit) -> dict:
    """执行完整匹配；emit(stage, data) 用于流式上报进度，非流式传 no-op.

    链路（用户query漏斗）：
      Step0 关键词直配（毫秒级，≥直配线直接命中，省所有 LLM/embedding 调用）
      → Step1 向量检索（标准阈值 vector_match_score）
      → Step2 关键词检索（二元组重合度 ≥ keyword_match_min_score）
      → Step3 向量兜底（阈值 vector_fallback_score，top-k）+ LLM 核验能否回答
      → 全部落空：严格模式=提示暂无答案；宽松模式=LLM 自身知识回答
    """
    t0 = time.perf_counter()
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    conn = db.get_conn()
    s = get_settings()
    top_k = body.top_k or int(s["default_top_k"])
    min_score = body.min_score if body.min_score is not None else s["match_min_score"]
    kw_min = s["keyword_match_min_score"]
    direct_score = s["keyword_direct_score"]
    vec_match = s["vector_match_score"]
    vec_fallback = s["vector_fallback_score"]
    vec_top_k = int(s["vector_top_k"])
    answer_mode = s["answer_mode"]
    trace: dict = {"params": {"top_k": top_k, "match_min_score": min_score,
                              "keyword_match_min_score": kw_min,
                              "keyword_direct_score": direct_score,
                              "vector_match_score": vec_match,
                              "vector_fallback_score": vec_fallback,
                              "vector_top_k": vec_top_k, "answer_mode": answer_mode}}

    all_qas = [dict(r) for r in conn.execute(
        "SELECT qa.* FROM qa WHERE qa.status='active'").fetchall()]
    if not all_qas:
        raise HTTPException(400, "知识库为空，请先添加 QA")
    qa_map = {q["id"]: q for q in all_qas}

    matched: list[llm.MatchCandidate] = []
    route_view: dict = {}
    funnel_stage = ""

    # ---- Step 0 关键词直配（扫描所有库，毫秒级）----
    tp = time.perf_counter()
    direct, d_detail = llm.keyword_match_global(question, all_qas, direct_score)
    trace["prematch"] = {**d_detail, "ms": round((time.perf_counter() - tp) * 1000)}
    await emit("prematch", trace["prematch"])
    if direct:
        matched = [llm.MatchCandidate(c.qa_id, c.score, f"关键词直配（重合度 {c.score}）")
                   for c in direct]
        funnel_stage = "direct"
        route_view = {"intent_id": None, "intent_name": "关键词直配", "kb_names": [],
                      "confidence": matched[0].score, "degraded": False,
                      "reason": "关键词重合度达标，直接命中"}

    # ---- Step 1 向量检索（标准阈值）----
    if not matched:
        tp = time.perf_counter()
        try:
            hits, v_detail = await embedding.search_async(question, top_k=top_k)
        except embedding.EmbeddingError as e:
            hits, v_detail = [], {"engine": "vector", "available": False, "reason": str(e)}
        trace["vector"] = {**v_detail,
                           "hits": [(h.qa_id, h.score) for h in hits],
                           "ms": round((time.perf_counter() - tp) * 1000)}
        await emit("vector", trace["vector"])
        passed = [h for h in hits if h.score >= vec_match]
        if passed:
            matched = [llm.MatchCandidate(h.qa_id, h.score, f"向量检索（相似度 {h.score}）")
                       for h in passed[:top_k]]
            funnel_stage = "vector"
            route_view = {"intent_id": None, "intent_name": "向量检索", "kb_names": [],
                          "confidence": passed[0].score, "degraded": False,
                          "reason": f"语义相似度 {passed[0].score} ≥ {vec_match}"}

    # ---- Step 2 关键词检索 ----
    if not matched:
        tp = time.perf_counter()
        kw_hits, kw_detail = llm.keyword_match_global(question, all_qas, kw_min)
        trace["keyword"] = {**kw_detail,
                            "hits": [(c.qa_id, c.score) for c in kw_hits],
                            "ms": round((time.perf_counter() - tp) * 1000)}
        await emit("keyword", trace["keyword"])
        if kw_hits:
            matched = kw_hits[:top_k]
            funnel_stage = "keyword"
            route_view = {"intent_id": None, "intent_name": "关键词检索", "kb_names": [],
                          "confidence": kw_hits[0].score, "degraded": True,
                          "reason": f"关键词重合度 {kw_hits[0].score} ≥ {kw_min}"}

    # ---- Step 3 向量兜底 + LLM 核验 ----
    verify_ctx: list[dict] = []
    if not matched:
        tp = time.perf_counter()
        try:
            fb_hits, fb_detail = await embedding.search_async(question, top_k=vec_top_k)
        except embedding.EmbeddingError as e:
            fb_hits, fb_detail = [], {"engine": "vector", "available": False, "reason": str(e)}
        fb_passed = [h for h in fb_hits if h.score >= vec_fallback]
        trace["fallback_vector"] = {**fb_detail,
                                    "threshold": vec_fallback,
                                    "hits": [(h.qa_id, h.score) for h in fb_passed],
                                    "ms": round((time.perf_counter() - tp) * 1000)}
        await emit("fallback_vector", trace["fallback_vector"])
        if fb_passed:
            cands = [qa_map[h.qa_id] for h in fb_passed if h.qa_id in qa_map]
            try:
                t4 = time.perf_counter()
                verified = await llm.verify_candidates(question, cands)
                trace["verify"] = {"results": [(v.qa_id, v.can_answer, v.score, v.reason)
                                               for v in verified],
                                   "ms": round((time.perf_counter() - t4) * 1000)}
                await emit("verify", trace["verify"])
                ok = [v for v in verified if v.can_answer]
                verify_ctx = [qa_map[v.qa_id] for v in verified if v.qa_id in qa_map]
                if ok:
                    matched = [llm.MatchCandidate(v.qa_id, v.score, f"兜底核验通过：{v.reason}")
                               for v in ok[:top_k]]
                    funnel_stage = "verified"
                    route_view = {"intent_id": None, "intent_name": "兜底核验", "kb_names": [],
                                  "confidence": ok[0].score, "degraded": True,
                                  "reason": f"向量兜底候选经 LLM 核验可回答（相似度 {fb_passed[0].score}）"}
            except llm.LLMError as e:
                trace["verify"] = {"error": str(e)}
                await emit("verify", trace["verify"])

    trace["funnel"] = funnel_stage or "none"

    # ---- 命中处理与日志 ----
    qa_details, hit_ids = [], []
    if matched:
        for c in matched[:top_k]:
            qa = qa_map[c.qa_id]
            hit_ids.append(c.qa_id)
            qa_details.append({
                "qa_id": qa["id"], "question": qa["question"], "answer": qa["answer"],
                "follow_ups": qa["follow_ups"], "star_hint": qa["star_hint"],
                "hit_count": qa["hit_count"], "quality": qa["quality"],
                "score": c.score, "reason": c.reason,
            })
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db.tx() as txn:
            for qa_id in hit_ids:
                txn.execute(
                    "UPDATE qa SET hit_count=hit_count+1, last_asked_at=? WHERE id=?", (now, qa_id))

    # ---- 全部落空：严格 / 宽松 ----
    free_text = None
    if not matched:
        if answer_mode == "loose":
            try:
                tp = time.perf_counter()
                free_text = await llm.free_answer(question, verify_ctx)
                trace["free_answer"] = {"ok": True,
                                        "ms": round((time.perf_counter() - tp) * 1000)}
            except llm.LLMError as e:
                trace["free_answer"] = {"ok": False, "error": str(e)}
            await emit("free_answer", trace.get("free_answer", {"ok": False}))
        else:
            trace["free_answer"] = {"ok": False, "skipped": "strict"}

    if not route_view:
        route_view = {"intent_id": None, "intent_name": None, "kb_names": [],
                      "confidence": 0.0, "degraded": False,
                      "reason": "三级检索均未命中"}

    source = "match" if hit_ids else "new"
    with db.tx() as txn:
        cur = txn.execute(
            "INSERT INTO query_log(kb_id, question, intent_id, route_confidence, matched_qa_ids, source) "
            "VALUES(?,?,?,?,?,?)",
            (body.kb_id, question, None, route_view["confidence"],
             json.dumps(hit_ids), source),
        )
        log_id = cur.lastrowid

    trace["total_ms"] = round((time.perf_counter() - t0) * 1000)
    result = {
        "log_id": log_id,
        "route": route_view,
        "matched": qa_details,
        "source": source,
        "answer_mode": answer_mode,
        "free_answer": free_text,
        "trace": trace,
    }
    await emit("done", result)
    return result


async def _noop(stage, data):
    return None


@app.post("/api/match", dependencies=[Depends(auth_dep)])
async def match(body: MatchIn):
    return await _run_match(body, emit=_noop)


@app.post("/api/match/stream", dependencies=[Depends(auth_dep)])
async def match_stream(body: MatchIn):
    """SSE 流式匹配：依次推送 route / match / fallback / done 阶段事件。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(stage: str, data):
        await queue.put("data: " + json.dumps({"stage": stage, "data": data}, ensure_ascii=False) + "\n\n")

    async def worker():
        try:
            await _run_match(body, emit)
        except HTTPException as e:
            await emit("error", {"detail": e.detail})
        except Exception as e:  # 未知异常也以 error 事件返回，避免前端无限等
            await emit("error", {"detail": f"匹配失败: {e}"})
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())

    async def gen():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class FeedbackIn(BaseModel):
    feedback: str  # useful / useless


@app.post("/api/log/{log_id}/feedback", dependencies=[Depends(auth_dep)])
def feedback(log_id: int, body: FeedbackIn):
    if body.feedback not in ("useful", "useless"):
        raise HTTPException(400, "feedback 只能是 useful 或 useless")
    with db.tx() as conn:
        cur = conn.execute("UPDATE query_log SET feedback=? WHERE id=?", (body.feedback, log_id))
        if cur.rowcount == 0:
            raise HTTPException(404, "日志不存在")
    return {"ok": True}


# ---------- 复盘统计（全局：跨知识库聚合） ----------
@app.get("/api/stats/overview", dependencies=[Depends(auth_dep)])
def stats_overview(kb_id: int | None = None):
    conn = db.get_conn()
    where, args = ("WHERE kb_id=? AND status='active'", [kb_id]) if kb_id else ("WHERE status='active'", [])
    total = conn.execute(
        f"SELECT COUNT(*) AS c FROM qa {where}", args).fetchone()["c"]
    lwhere, largs = ("WHERE kb_id=?", [kb_id]) if kb_id else ("", [])
    logs = conn.execute(f"SELECT source FROM query_log {lwhere}", largs).fetchall()
    hit = sum(1 for r in logs if r["source"] == "match")
    new = sum(1 for r in logs if r["source"] == "new")
    # 近 14 天命中趋势
    trend = []
    for r in conn.execute(
        "SELECT date(created_at) AS d, SUM(CASE WHEN source='match' THEN 1 ELSE 0 END) AS hit, "
        f"COUNT(*) AS total FROM query_log {lwhere} GROUP BY date(created_at) "
        "ORDER BY d DESC LIMIT 14", largs).fetchall():
        trend.append({"date": r["d"], "hit": r["hit"], "total": r["total"]})
    trend.reverse()
    return {"qa_total": total, "query_total": len(logs), "hit": hit, "new": new,
            "hit_rate": round(hit / len(logs), 3) if logs else None, "trend": trend}


@app.get("/api/stats/hot", dependencies=[Depends(auth_dep)])
def stats_hot(kb_id: int | None = None):
    conn = db.get_conn()
    where, args = ("WHERE kb_id=? AND status='active' AND hit_count>0", [kb_id]) \
        if kb_id else ("WHERE status='active' AND hit_count>0", [])
    rows = conn.execute(
        f"SELECT id, question, hit_count, quality FROM qa {where} "
        "ORDER BY hit_count DESC LIMIT 10", args).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/stats/top_categories", dependencies=[Depends(auth_dep)])
def stats_top_categories():
    """Top10 知识库-一级分类（按 QA 累计被问次数）。"""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT kb.name AS kb_name, c.name AS category_name, "
        "SUM(qa.hit_count) AS hits, COUNT(qa.id) AS qa_count "
        "FROM qa JOIN kb ON qa.kb_id=kb.id "
        "LEFT JOIN category c ON qa.category_id=c.id "
        "LEFT JOIN category pc ON c.parent_id=pc.id "
        "WHERE qa.status='active' "
        "GROUP BY kb.name, COALESCE(pc.name, c.name) "
        "HAVING SUM(qa.hit_count) > 0 "
        "ORDER BY hits DESC LIMIT 10").fetchall()
    return [{"kb_name": r["kb_name"],
             "category_name": r["category_name"] or "（未分类）",
             "hits": r["hits"], "qa_count": r["qa_count"]} for r in rows]


@app.get("/api/stats/top_tags", dependencies=[Depends(auth_dep)])
def stats_top_tags():
    """Top10 知识库-标签（按 QA 累计被问次数）。"""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT kb.name AS kb_name, t.name AS tag_name, "
        "SUM(qa.hit_count) AS hits, COUNT(DISTINCT qa.id) AS qa_count "
        "FROM qa_tag qt JOIN qa ON qt.qa_id=qa.id AND qa.status='active' "
        "JOIN tag t ON qt.tag_id=t.id "
        "JOIN kb ON qa.kb_id=kb.id "
        "GROUP BY kb.name, t.name "
        "HAVING SUM(qa.hit_count) > 0 "
        "ORDER BY hits DESC LIMIT 10").fetchall()
    return [{"kb_name": r["kb_name"], "tag_name": r["tag_name"],
             "hits": r["hits"], "qa_count": r["qa_count"]} for r in rows]


@app.get("/api/stats/unmatched", dependencies=[Depends(auth_dep)])
def stats_unmatched(kb_id: int):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, question, intent_id, created_at FROM query_log "
        "WHERE kb_id=? AND source='new' ORDER BY created_at DESC LIMIT 200", (kb_id,)).fetchall()
    return [dict(r) for r in rows]


class FromLogIn(BaseModel):
    kb_id: int
    answer: str
    question_aliases: str = ""
    category_id: int | None = None
    tag_ids: list[int] = []
    question: str | None = None  # 不传则用日志原话


@app.post("/api/qas/from_log/{log_id}", dependencies=[Depends(auth_dep)])
def qa_from_log(log_id: int, body: FromLogIn):
    conn = db.get_conn()
    log = conn.execute("SELECT * FROM query_log WHERE id=?", (log_id,)).fetchone()
    if not log:
        raise HTTPException(404, "日志不存在")
    qa_in = QAIn(
        kb_id=body.kb_id,
        category_id=body.category_id,
        tag_ids=body.tag_ids,
        question=(body.question or log["question"]).strip(),
        question_aliases=body.question_aliases,
        answer=body.answer,
    )
    result = create_qa(qa_in)
    with db.tx() as txn:
        txn.execute("UPDATE query_log SET source='match', matched_qa_ids=? WHERE id=?",
                    (json.dumps([result["id"]]), log_id))
    return result


@app.on_event("startup")
def startup():
    errors = config.validate()
    for e in errors:
        print(f"[config 警告] {e}")
    db.init_db()
    embedding.store.load()  # 向量缓存预热（表可能为空，检索时惰性补算）


# ---------- 向量管理 ----------
@app.get("/api/vectors/status", dependencies=[Depends(auth_dep)])
def vector_status():
    """向量库状态：总数 / 缺失数（stale）。"""
    return {"total": embedding.store.count, "stale": len(embedding.stale_ids()),
            "model": config.EMBEDDING_MODEL, "available": embedding.store.available}


@app.post("/api/vectors/rebuild", dependencies=[Depends(auth_dep)])
async def vector_rebuild():
    """全量重建向量（换 embedding 模型或文本批量变更后）。"""
    try:
        result = await embedding.rebuild_all()
        return {"ok": True, **result}
    except embedding.EmbeddingError as e:
        raise HTTPException(502, f"embedding 调用失败: {e}")
