"""LLM 路由模块：Step1 意图路由 + Step2 候选精配。

OpenAI 兼容 chat/completions；失败时抛 LLMError，由上层降级关键词匹配。
"""
import json
import re
from dataclasses import dataclass, field

import httpx

from . import config


class LLMError(Exception):
    pass


ROUTE_SYSTEM_PROMPT = (
    "你是面试题库的意图路由器。根据面试官的问题，从意图树中选择最匹配的分类。\n"
    "规则：\n"
    "1. 先定位 L1 大类，再在该大类的 L2 中选择；只能输出树中存在的 id\n"
    "2. 分类拿不准时 confidence 给低分；完全无法归类输出 l1_id=\"unknown\"\n"
    "3. 只输出 JSON，不要解释\n"
    '输出格式：{"l1_id":"...","l2_id":"...","confidence":0.0,"reason":"..."}'
)

MATCH_SYSTEM_PROMPT = (
    "你是面试 QA 匹配器。给定候选问题列表（含变体问法），选出与目标问题语义最接近的 1~3 条。\n"
    "规则：\n"
    "1. 语义相近即算匹配：换了说法、部分重合、口语化表达都算\n"
    "2. 只有候选能**直接回答**目标问题时才算匹配，score 给 0.7 以上；"
    "仅话题沾边（同主题但回答不了这个问题，如问 ROI 却给'商业价值判断'）不算匹配，宁缺毋滥\n"
    "3. 即使都不对口，也要输出语义最接近的候选并如实给低分（0.2~0.6）并说明为什么回答不了，"
    "由系统按阈值过滤；仅当候选与问题毫无关联时才输出空列表\n"
    "4. 按 score 降序输出最多 3 条；只输出 JSON\n"
    '输出格式：{"matched":[{"id":1,"score":0.95,"reason":"..."}]}'
)


@dataclass
class RouteResult:
    l1_id: int | None
    l2_id: int | None
    confidence: float
    reason: str
    degraded: bool = False  # True=关键词兜底
    detail: dict = field(default_factory=dict)  # 过程明细：engine/best_score/threshold/pool_size


@dataclass
class MatchCandidate:
    qa_id: int
    score: float
    reason: str


@dataclass
class MatchResult:
    candidates: list[MatchCandidate] = field(default_factory=list)
    degraded: bool = False
    detail: dict = field(default_factory=dict)  # 过程明细：engine/pool_size/min_score/raw/dropped/best_score


async def _chat_json(system: str, user: str) -> dict:
    """调用 chat/completions 并解析 JSON 输出。"""
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY 未配置")
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:200]}") from e
    except (httpx.HTTPError, KeyError, IndexError) as e:
        raise LLMError(f"LLM 调用失败: {e}") from e
    return _extract_json(content)


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象（容忍 markdown 代码块包裹）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM 输出非 JSON: {text[:200]}") from e


def _build_intent_tree(l1_rows: list[dict], l2_rows: list[dict]) -> list[dict]:
    tree: list[dict] = []
    l2_by_l1: dict[int, list[dict]] = {}
    for l2 in l2_rows:
        l2_by_l1.setdefault(l2["l1_id"], []).append(
            {"id": l2["id"], "name": l2["name"], "description": l2["description"]}
        )
    for l1 in l1_rows:
        tree.append(
            {
                "id": l1["id"],
                "name": l1["name"],
                "description": l1["description"],
                "children": l2_by_l1.get(l1["id"], []),
            }
        )
    return tree


def _keyword_route(question: str, l1_rows: list[dict], l2_rows: list[dict],
                   min_score: float = 0.3) -> RouteResult:
    """关键词兜底路由：问题与 L2 名称/描述的字符重合度打分。"""
    best_l2, best_score = None, 0.0
    for l2 in l2_rows:
        text = f"{l2['name']} {l2['description'] or ''}"
        hits = sum(1 for ch in set(question) if ch in text and not ch.isspace())
        score = hits / max(len(set(question) - {" "}), 1)
        if score > best_score:
            best_score, best_l2 = score, l2
    detail = {"engine": "keyword", "pool_size": len(l2_rows),
              "best_score": round(best_score, 2), "threshold": min_score}
    if best_l2 and best_score >= min_score:
        return RouteResult(best_l2["l1_id"], best_l2["id"], round(best_score * 0.8, 2),
                           "关键词兜底", degraded=True, detail=detail)
    return RouteResult(None, None, 0.0, "关键词兜底未定位", degraded=True, detail=detail)


def _keyword_match(question: str, qas: list[dict],
                   min_score: float = 0.5) -> tuple[list[MatchCandidate], dict]:
    """关键词兜底精配：与 question+aliases 的字符重合度。返回 (候选, 过程明细)。"""
    scored_all: list[tuple[int, float]] = []
    q_chars = set(question) - {" "}
    for qa in qas:
        text = f"{qa['question']} {qa['question_aliases'] or ''}"
        hits = sum(1 for ch in q_chars if ch in text)
        scored_all.append((qa["id"], round(hits / max(len(q_chars), 1), 2)))
    kept = [MatchCandidate(i, s, "关键词兜底") for i, s in scored_all if s >= min_score]
    kept.sort(key=lambda c: -c.score)
    detail = {"engine": "keyword", "pool_size": len(qas), "threshold": min_score,
              "best_score": max((s for _, s in scored_all), default=0.0)}
    return kept[:3], detail


def keyword_match_global(question: str, qas: list[dict],
                         min_score: float = 0.5) -> tuple[list[MatchCandidate], dict]:
    """路由失败或 L2 内无命中时的全库关键词兜底匹配。"""
    return _keyword_match(question, qas, min_score)


async def route(question: str, l1_rows: list[dict], l2_rows: list[dict],
                route_min_score: float = 0.3) -> RouteResult:
    """Step 1：意图路由（L1 → L2）。"""
    if not l1_rows:
        return RouteResult(None, None, 0.0, "意图树为空", detail={"engine": "none"})
    tree = _build_intent_tree(l1_rows, l2_rows)
    try:
        data = await _chat_json(ROUTE_SYSTEM_PROMPT, json.dumps({"intent_tree": tree, "question": question}, ensure_ascii=False))
        l1_id = data.get("l1_id")
        l2_id = data.get("l2_id")
        valid_l1 = {r["id"] for r in l1_rows}
        valid_l2 = {r["id"] for r in l2_rows}
        l1_id = int(l1_id) if l1_id not in (None, "unknown") and str(l1_id).isdigit() and int(l1_id) in valid_l1 else None
        l2_id = int(l2_id) if l2_id not in (None, "unknown") and str(l2_id).isdigit() and int(l2_id) in valid_l2 else None
        return RouteResult(l1_id, l2_id, float(data.get("confidence", 0) or 0), str(data.get("reason", "")),
                           detail={"engine": "llm", "pool_size": len(l2_rows)})
    except LLMError:
        if config.KEYWORD_FALLBACK:
            return _keyword_route(question, l1_rows, l2_rows, route_min_score)
        raise


async def match_candidates(question: str, qas: list[dict],
                           min_score: float | None = None,
                           kw_min: float = 0.5, limit: int = 3) -> MatchResult:
    """Step 2：候选精配，输入为该 L2 下所有 active QA。"""
    if not qas:
        return MatchResult(detail={"engine": "none", "pool_size": 0})
    if min_score is None:
        min_score = config.MATCH_MIN_SCORE
    try:
        cands = [{"id": q["id"], "question": q["question"], "aliases": q["question_aliases"] or ""} for q in qas]
        data = await _chat_json(MATCH_SYSTEM_PROMPT, json.dumps({"candidates": cands, "question": question}, ensure_ascii=False))
        valid = {q["id"] for q in qas}
        raw: list[MatchCandidate] = []
        for item in data.get("matched", []):
            try:
                qa_id = int(item["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if qa_id in valid:
                raw.append(MatchCandidate(qa_id, float(item.get("score", 0) or 0), str(item.get("reason", ""))))
        raw.sort(key=lambda c: -c.score)
        # 过滤"擦边"低分候选：宁缺毋滥，空结果会触发上层全库二次兜底
        q_text = {q["id"]: q["question"] for q in qas}
        kept = [c for c in raw if c.score >= min_score]
        dropped = [{"id": c.qa_id, "question": q_text.get(c.qa_id, ""),
                    "score": c.score, "reason": c.reason} for c in raw if c.score < min_score]
        detail = {"engine": "llm", "pool_size": len(qas), "min_score": min_score,
                  "raw_count": len(raw),
                  "best_raw_score": max((c.score for c in raw), default=None),
                  "dropped": dropped}
        return MatchResult(kept[:limit], detail=detail)
    except LLMError:
        if config.KEYWORD_FALLBACK:
            cands, detail = _keyword_match(question, qas, kw_min)
            detail["degraded"] = True
            return MatchResult(cands, degraded=True, detail=detail)
        raise
