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


# ---------- QA ----------
class QAIn(BaseModel):
    kb_id: int
    question: str
    question_aliases: str = ""
    answer: str
    follow_ups: str = ""
    star_hint: str = ""
    status: str = "active"
    quality: int = 3


@app.get("/api/qas", dependencies=[Depends(auth_dep)])
def list_qas(kb_id: int, q: str = "", status: str = ""):
    sql = "SELECT qa.*, kb.name AS kb_name FROM qa JOIN kb ON qa.kb_id=kb.id WHERE qa.kb_id=?"
    args: list = [kb_id]
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
            "INSERT INTO qa(kb_id, question, question_aliases, answer, follow_ups, star_hint, status, quality) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (body.kb_id, body.question.strip(), body.question_aliases.strip(), body.answer.strip(),
             body.follow_ups.strip(), body.star_hint.strip(), body.status, body.quality),
        )
        return {"id": cur.lastrowid}


@app.put("/api/qas/{qa_id}", dependencies=[Depends(auth_dep)])
def update_qa(qa_id: int, body: QAIn):
    with db.tx() as conn:
        cur = conn.execute(
            "UPDATE qa SET kb_id=?, question=?, question_aliases=?, answer=?, follow_ups=?, star_hint=?, "
            "status=?, quality=?, updated_at=datetime('now','localtime') WHERE id=?",
            (body.kb_id, body.question.strip(), body.question_aliases.strip(), body.answer.strip(),
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
    "default_top_k": (int, 1, 10),
}
_SETTING_DEFAULTS = {
    "match_min_score": lambda: config.MATCH_MIN_SCORE,
    "route_min_score": lambda: config.ROUTE_MIN_SCORE,
    "keyword_match_min_score": lambda: config.KEYWORD_MATCH_MIN_SCORE,
    "keyword_direct_score": lambda: config.KEYWORD_DIRECT_SCORE,
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
    return out


@app.get("/api/settings", dependencies=[Depends(auth_dep)])
def read_settings():
    return get_settings()


class SettingsIn(BaseModel):
    values: dict[str, float | int]


@app.put("/api/settings", dependencies=[Depends(auth_dep)])
def update_settings(body: SettingsIn):
    for key, value in body.values.items():
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


# ---------- 匹配（两级 LLM 路由，流式过程 + 三层兜底） ----------
async def _run_match(body: MatchIn, emit) -> dict:
    """执行完整匹配；emit(stage, data) 用于流式上报进度，非流式传 no-op。

    链路：Step0 全部库关键词直配（≥直配线直接命中，省 LLM）
          → 未命中 Step1 LLM 意图识别 → 意图挂载知识库合并池
          → Step2 池内全量精配（一次 LLM 调用出 top-k）
          →（未达标时）全部库关键词兜底 → 记日志。
    """
    t0 = time.perf_counter()
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    conn = db.get_conn()
    s = get_settings()
    top_k = body.top_k or int(s["default_top_k"])
    min_score = body.min_score if body.min_score is not None else s["match_min_score"]
    route_min = s["route_min_score"]
    kw_min = s["keyword_match_min_score"]
    direct_score = s["keyword_direct_score"]
    trace: dict = {"params": {"top_k": top_k, "match_min_score": min_score,
                              "route_min_score": route_min, "keyword_match_min_score": kw_min,
                              "keyword_direct_score": direct_score}}

    # 全局资源只查一次：所有库 active QA + 全局意图（含挂载的库）
    all_qas = [dict(r) for r in conn.execute(
        "SELECT qa.* FROM qa WHERE qa.status='active'").fetchall()]
    intents = []
    for r in conn.execute("SELECT * FROM intent ORDER BY sort_order, id").fetchall():
        it = dict(r)
        kb_rows = conn.execute(
            "SELECT kb.id, kb.name FROM intent_kb ik JOIN kb ON ik.kb_id=kb.id WHERE ik.intent_id=?",
            (r["id"],)).fetchall()
        it["kb_ids"] = [k["id"] for k in kb_rows]
        it["kb_names"] = [k["name"] for k in kb_rows]
        intents.append(it)
    if not intents:
        raise HTTPException(400, "还没有任何意图，请先在题库管理页创建意图并挂载知识库")

    # ---- Step 0 关键词直配（扫描所有库，毫秒级；命中则省去两轮 LLM 调用）----
    route_res = None
    qas, matched, degraded = [], [], False
    tp = time.perf_counter()
    direct, d_detail = llm.keyword_match_global(question, all_qas, direct_score) if all_qas \
        else ([], {"engine": "none", "pool_size": 0, "threshold": direct_score, "best_score": 0.0})
    trace["prematch"] = {**d_detail, "ms": round((time.perf_counter() - tp) * 1000)}
    await emit("prematch", trace["prematch"])
    if direct:
        qas = all_qas
        matched = [llm.MatchCandidate(c.qa_id, c.score, f"关键词直配（重合度 {c.score}）")
                   for c in direct]
    else:
        # ---- Step 1 LLM 意图识别（意图 → 挂载的知识库）----
        t1 = time.perf_counter()
        route_res = await llm.route(question, intents, route_min)
        trace["route"] = {**route_res.detail, "ms": round((time.perf_counter() - t1) * 1000)}
        intent = next((it for it in intents if it["id"] == route_res.intent_id), None)
        await emit("route", {
            "intent_id": route_res.intent_id,
            "intent_name": intent["name"] if intent else None,
            "kb_names": intent["kb_names"] if intent else [],
            "confidence": route_res.confidence, "reason": route_res.reason,
            "degraded": route_res.degraded, "ms": trace["route"]["ms"], **route_res.detail,
        })

        # ---- Step 2 全量精配（意图挂载的所有知识库 QA 合并池，一次 LLM 调用）----
        pool = []
        if intent and intent["kb_ids"]:
            kb_set = set(intent["kb_ids"])
            pool = [q for q in all_qas if q["kb_id"] in kb_set]
        if pool:
            t2 = time.perf_counter()
            m_res = await llm.match_candidates(question, pool, min_score=min_score, kw_min=kw_min, limit=top_k)
            matched, degraded = m_res.candidates, m_res.degraded
            trace["match"] = {**m_res.detail, "ms": round((time.perf_counter() - t2) * 1000)}
            await emit("match", trace["match"])
            if not matched and config.KEYWORD_FALLBACK:
                # 意图池内没匹配到（含低分被滤）→ 全部库关键词兜底（意图可能识别错）
                t3 = time.perf_counter()
                global_matched, g_detail = llm.keyword_match_global(question, all_qas, kw_min)
                trace["fallback"] = {**g_detail, "trigger": "意图池内无达标候选",
                                     "ms": round((time.perf_counter() - t3) * 1000)}
                await emit("fallback", trace["fallback"])
                if global_matched:
                    qas, matched, degraded = all_qas, global_matched, True
        elif config.KEYWORD_FALLBACK:
            # 意图未定位或挂载库为空 → 全部库关键词兜底，尽量给候选而不是直接判"新问题"
            t3 = time.perf_counter()
            if all_qas:
                matched, g_detail = llm.keyword_match_global(question, all_qas, kw_min)
                degraded = True
                trace["fallback"] = {**g_detail, "trigger": "意图未定位",
                                     "ms": round((time.perf_counter() - t3) * 1000)}
                await emit("fallback", trace["fallback"])
                if matched:
                    qas = all_qas
            else:
                trace["fallback"] = {"engine": "none", "trigger": "意图未定位且库内无 QA"}

    # ---- 命中处理与日志 ----
    qa_details, hit_ids = [], []
    if matched:
        qa_map = {q["id"]: q for q in qas}
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

    source = "match" if hit_ids else "new"
    with db.tx() as txn:
        cur = txn.execute(
            "INSERT INTO query_log(kb_id, question, intent_id, route_confidence, matched_qa_ids, source) "
            "VALUES(?,?,?,?,?,?)",
            (body.kb_id, question,
             route_res.intent_id if route_res else None,
             route_res.confidence if route_res else (matched[0].score if matched else 0.0),
             json.dumps(hit_ids), source),
        )
        log_id = cur.lastrowid

    if route_res:
        intent = next((it for it in intents if it["id"] == route_res.intent_id), None)
        route_view = {"intent_id": route_res.intent_id,
                      "intent_name": intent["name"] if intent else None,
                      "kb_names": intent["kb_names"] if intent else [],
                      "confidence": route_res.confidence,
                      "reason": route_res.reason, "degraded": degraded}
    else:
        # 关键词直配命中：未走 LLM 路由
        route_view = {"intent_id": None, "intent_name": "关键词直配", "kb_names": [],
                      "confidence": matched[0].score if matched else trace["prematch"].get("best_score", 0.0),
                      "reason": "关键词重合度达标，直接命中，未调用 LLM", "degraded": False}

    trace["total_ms"] = round((time.perf_counter() - t0) * 1000)
    result = {
        "log_id": log_id,
        "route": route_view,
        "matched": qa_details,
        "source": source,
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


# ---------- 复盘统计 ----------
@app.get("/api/stats/overview", dependencies=[Depends(auth_dep)])
def stats_overview(kb_id: int):
    conn = db.get_conn()
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM qa WHERE kb_id=? AND status='active'", (kb_id,)).fetchone()["c"]
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
        "SELECT id, question, hit_count, quality FROM qa "
        "WHERE kb_id=? AND status='active' AND hit_count>0 "
        "ORDER BY hit_count DESC LIMIT 10", (kb_id,)).fetchall()
    return [dict(r) for r in rows]


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
        kb_id=body.kb_id,
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
