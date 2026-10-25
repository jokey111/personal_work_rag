"""FastAPI 主应用：认证 + 知识库/意图/QA CRUD + 匹配 + 复盘统计。"""
import json
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import config, db, llm, md_import

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
            "SELECT COUNT(*) AS c FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=?",
            (kb["id"],),
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


# ---------- 意图树 ----------
class L1In(BaseModel):
    kb_id: int
    name: str
    description: str = ""
    sort_order: int = 0


class L2In(BaseModel):
    l1_id: int
    name: str
    description: str = ""
    sort_order: int = 0


def _kb_of_l1(l1_id: int) -> int:
    row = db.get_conn().execute("SELECT kb_id FROM intent_l1 WHERE id=?", (l1_id,)).fetchone()
    if not row:
        raise HTTPException(404, "一级意图不存在")
    return row["kb_id"]


@app.get("/api/intents", dependencies=[Depends(auth_dep)])
def get_intents(kb_id: int):
    conn = db.get_conn()
    l1s = [dict(r) for r in conn.execute(
        "SELECT * FROM intent_l1 WHERE kb_id=? ORDER BY sort_order, id", (kb_id,)).fetchall()]
    l2s = [dict(r) for r in conn.execute(
        "SELECT l2.* FROM intent_l2 l2 JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=? ORDER BY l2.sort_order, l2.id",
        (kb_id,)).fetchall()]
    l2_by_l1: dict[int, list] = {}
    for l2 in l2s:
        qa_count = conn.execute("SELECT COUNT(*) AS c FROM qa WHERE l2_id=?", (l2["id"],)).fetchone()["c"]
        l2["qa_count"] = qa_count
        l2_by_l1.setdefault(l2["l1_id"], []).append(l2)
    for l1 in l1s:
        l1["children"] = l2_by_l1.get(l1["id"], [])
    return l1s


@app.post("/api/intents/l1", dependencies=[Depends(auth_dep)])
def create_l1(body: L1In):
    try:
        with db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO intent_l1(kb_id, name, description, sort_order) VALUES(?,?,?,?)",
                (body.kb_id, body.name.strip(), body.description.strip(), body.sort_order),
            )
            return {"id": cur.lastrowid}
    except Exception:
        raise HTTPException(400, "创建失败：同级可能已存在同名大类")


@app.put("/api/intents/l1/{l1_id}", dependencies=[Depends(auth_dep)])
def update_l1(l1_id: int, body: L1In):
    with db.tx() as conn:
        cur = conn.execute(
            "UPDATE intent_l1 SET kb_id=?, name=?, description=?, sort_order=? WHERE id=?",
            (body.kb_id, body.name.strip(), body.description.strip(), body.sort_order, l1_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "一级意图不存在")
    return {"ok": True}


@app.delete("/api/intents/l1/{l1_id}", dependencies=[Depends(auth_dep)])
def delete_l1(l1_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM intent_l1 WHERE id=?", (l1_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "一级意图不存在")
    return {"ok": True}


@app.post("/api/intents/l2", dependencies=[Depends(auth_dep)])
def create_l2(body: L2In):
    _kb_of_l1(body.l1_id)
    try:
        with db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO intent_l2(l1_id, name, description, sort_order) VALUES(?,?,?,?)",
                (body.l1_id, body.name.strip(), body.description.strip(), body.sort_order),
            )
            return {"id": cur.lastrowid}
    except Exception:
        raise HTTPException(400, "创建失败：同大类下可能已存在同名子意图")


@app.put("/api/intents/l2/{l2_id}", dependencies=[Depends(auth_dep)])
def update_l2(l2_id: int, body: L2In):
    with db.tx() as conn:
        cur = conn.execute(
            "UPDATE intent_l2 SET l1_id=?, name=?, description=?, sort_order=? WHERE id=?",
            (body.l1_id, body.name.strip(), body.description.strip(), body.sort_order, l2_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "二级意图不存在")
    return {"ok": True}


@app.delete("/api/intents/l2/{l2_id}", dependencies=[Depends(auth_dep)])
def delete_l2(l2_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM intent_l2 WHERE id=?", (l2_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "二级意图不存在")
    return {"ok": True}


# ---------- QA ----------
class QAIn(BaseModel):
    l2_id: int
    question: str
    question_aliases: str = ""
    answer: str
    follow_ups: str = ""
    star_hint: str = ""
    status: str = "active"
    quality: int = 3


@app.get("/api/qas", dependencies=[Depends(auth_dep)])
def list_qas(kb_id: int, l1_id: int | None = None, l2_id: int | None = None, q: str = "", status: str = ""):
    sql = (
        "SELECT qa.*, l2.name AS l2_name, l2.l1_id, l1.name AS l1_name, l1.kb_id "
        "FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=?"
    )
    args: list = [kb_id]
    if l2_id:
        sql += " AND qa.l2_id=?"
        args.append(l2_id)
    elif l1_id:
        sql += " AND l2.l1_id=?"
        args.append(l1_id)
    if q:
        sql += " AND (qa.question LIKE ? OR qa.question_aliases LIKE ? OR qa.answer LIKE ?)"
        like = f"%{q}%"
        args += [like, like, like]
    if status:
        sql += " AND qa.status=?"
        args.append(status)
    sql += " ORDER BY qa.updated_at DESC LIMIT 500"
    return [dict(r) for r in db.get_conn().execute(sql, args).fetchall()]


@app.post("/api/qas", dependencies=[Depends(auth_dep)])
def create_qa(body: QAIn):
    with db.tx() as conn:
        cur = conn.execute(
            "INSERT INTO qa(l2_id, question, question_aliases, answer, follow_ups, star_hint, status, quality) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (body.l2_id, body.question.strip(), body.question_aliases.strip(), body.answer.strip(),
             body.follow_ups.strip(), body.star_hint.strip(), body.status, body.quality),
        )
        return {"id": cur.lastrowid}


@app.put("/api/qas/{qa_id}", dependencies=[Depends(auth_dep)])
def update_qa(qa_id: int, body: QAIn):
    with db.tx() as conn:
        cur = conn.execute(
            "UPDATE qa SET l2_id=?, question=?, question_aliases=?, answer=?, follow_ups=?, star_hint=?, "
            "status=?, quality=?, updated_at=datetime('now','localtime') WHERE id=?",
            (body.l2_id, body.question.strip(), body.question_aliases.strip(), body.answer.strip(),
             body.follow_ups.strip(), body.star_hint.strip(), body.status, body.quality, qa_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "QA 不存在")
    return {"ok": True}


@app.delete("/api/qas/{qa_id}", dependencies=[Depends(auth_dep)])
def delete_qa(qa_id: int):
    with db.tx() as conn:
        cur = conn.execute("DELETE FROM qa WHERE id=?", (qa_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "QA 不存在")
    return {"ok": True}


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


# ---------- 匹配（两级 LLM 路由） ----------
class MatchIn(BaseModel):
    kb_id: int
    question: str
    top_k: int = 3


@app.post("/api/match", dependencies=[Depends(auth_dep)])
async def match(body: MatchIn):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    conn = db.get_conn()

    l1_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM intent_l1 WHERE kb_id=? ORDER BY sort_order, id", (body.kb_id,)).fetchall()]
    l2_rows = [dict(r) for r in conn.execute(
        "SELECT l2.* FROM intent_l2 l2 JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=?",
        (body.kb_id,)).fetchall()]
    if not l1_rows:
        raise HTTPException(400, "该知识库还没有意图分类，请先创建")

    route_res = await llm.route(question, l1_rows, l2_rows)

    qas, matched = [], []
    degraded = route_res.degraded
    if route_res.l2_id:
        qas = [dict(r) for r in conn.execute(
            "SELECT * FROM qa WHERE l2_id=? AND status='active'", (route_res.l2_id,)).fetchall()]
    if qas:
        m_res = await llm.match_candidates(question, qas)
        matched = m_res.candidates
        degraded = m_res.degraded
    elif not route_res.l2_id and config.KEYWORD_FALLBACK:
        # 路由未定位 L2（LLM 不可用或低置信）→ 全库关键词兜底，尽量给候选而不是直接判"新问题"
        qas = [dict(r) for r in conn.execute(
            "SELECT qa.* FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id "
            "JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=? AND qa.status='active'",
            (body.kb_id,)).fetchall()]
        if qas:
            matched = llm.keyword_match_global(question, qas)
            degraded = True

    l1_name = next((r["name"] for r in l1_rows if r["id"] == route_res.l1_id), None)
    l2_name = next((r["name"] for r in l2_rows if r["id"] == route_res.l2_id), None)

    qa_details = []
    hit_ids = []
    if matched:
        qa_map = {q["id"]: q for q in qas}
        for c in matched[: body.top_k]:
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

    source = "match" if hit_ids else "new"
    with db.tx() as txn:
        cur = txn.execute(
            "INSERT INTO query_log(kb_id, question, l1_id, l2_id, route_confidence, matched_qa_ids, source) "
            "VALUES(?,?,?,?,?,?,?)",
            (body.kb_id, question, route_res.l1_id, route_res.l2_id,
             route_res.confidence, json.dumps(hit_ids), source),
        )
        log_id = cur.lastrowid

    return {
        "log_id": log_id,
        "route": {"l1_id": route_res.l1_id, "l2_id": route_res.l2_id, "l1_name": l1_name,
                  "l2_name": l2_name, "confidence": route_res.confidence,
                  "reason": route_res.reason, "degraded": degraded},
        "matched": qa_details,
        "source": source,
    }


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


# ---------- 复盘统计 ----------
@app.get("/api/stats/overview", dependencies=[Depends(auth_dep)])
def stats_overview(kb_id: int):
    conn = db.get_conn()
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id JOIN intent_l1 l1 ON l2.l1_id=l1.id "
        "WHERE l1.kb_id=? AND qa.status='active'", (kb_id,)).fetchone()["c"]
    logs = conn.execute("SELECT source FROM query_log WHERE kb_id=?", (kb_id,)).fetchall()
    hit = sum(1 for r in logs if r["source"] == "match")
    new = sum(1 for r in logs if r["source"] == "new")
    # 近 14 天命中趋势
    trend = []
    for r in conn.execute(
        "SELECT date(created_at) AS d, SUM(CASE WHEN source='match' THEN 1 ELSE 0 END) AS hit, "
        "COUNT(*) AS total FROM query_log WHERE kb_id=? GROUP BY date(created_at) "
        "ORDER BY d DESC LIMIT 14", (kb_id,)).fetchall():
        trend.append({"date": r["d"], "hit": r["hit"], "total": r["total"]})
    trend.reverse()
    return {"qa_total": total, "query_total": len(logs), "hit": hit, "new": new,
            "hit_rate": round(hit / len(logs), 3) if logs else None, "trend": trend}


@app.get("/api/stats/hot", dependencies=[Depends(auth_dep)])
def stats_hot(kb_id: int):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT qa.id, qa.question, qa.hit_count, qa.quality, l2.name AS l2_name "
        "FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id JOIN intent_l1 l1 ON l2.l1_id=l1.id "
        "WHERE l1.kb_id=? AND qa.status='active' AND qa.hit_count>0 "
        "ORDER BY qa.hit_count DESC LIMIT 10", (kb_id,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/stats/unmatched", dependencies=[Depends(auth_dep)])
def stats_unmatched(kb_id: int):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, question, l1_id, l2_id, created_at FROM query_log "
        "WHERE kb_id=? AND source='new' ORDER BY created_at DESC LIMIT 200", (kb_id,)).fetchall()
    return [dict(r) for r in rows]


class FromLogIn(BaseModel):
    l2_id: int
    answer: str
    question_aliases: str = ""
    follow_ups: str = ""
    star_hint: str = ""
    question: str | None = None  # 不传则用日志原话


@app.post("/api/qas/from_log/{log_id}", dependencies=[Depends(auth_dep)])
def qa_from_log(log_id: int, body: FromLogIn):
    conn = db.get_conn()
    log = conn.execute("SELECT * FROM query_log WHERE id=?", (log_id,)).fetchone()
    if not log:
        raise HTTPException(404, "日志不存在")
    qa_in = QAIn(
        l2_id=body.l2_id,
        question=(body.question or log["question"]).strip(),
        question_aliases=body.question_aliases,
        answer=body.answer,
        follow_ups=body.follow_ups,
        star_hint=body.star_hint,
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
