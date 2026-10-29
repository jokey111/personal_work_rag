"""LLM 路由模块：Step1 意图路由（意图→知识库） + Step2 候选精配。

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
    "你是面试题库的意图路由器。根据面试官的问题，从意图列表中选择最匹配的一个意图。"
    "每个意图挂载了若干知识库，选中意图后将在其挂载的知识库中检索答案。\n"
    "规则：\n"
    "1. 只能输出列表中存在的 intent_id\n"
    "2. 分类拿不准时 confidence 给低分；完全无法归类输出 intent_id=\"unknown\"\n"
    "3. 只输出 JSON，不要解释\n"
    '输出格式：{"intent_id":"...","confidence":0.0,"reason":"..."}'
)

MATCH_SYSTEM_PROMPT = (
    "你是面试 QA 匹配器。给定候选问题列表，选出与目标问题语义最接近的 1~3 条。\n"
    "规则：\n"
    "1. 语义相近即算匹配：换了说法、部分重合、口语化表达都算\n"
    "2. 只有候选能**直接回答**目标问题时才算匹配，score 给 0.7 以上；"
    "仅话题沾边（同主题但回答不了这个问题，如问 ROI 却给'商业价值判断'）不算匹配，宁缺毋滥\n"
    "3. 即使都不对口，也要输出语义最接近的候选并如实给低分（0.2~0.6）并说明为什么回答不了，"
    "由系统按阈值过滤；仅当候选与问题毫无关联时才输出空列表\n"
    "4. 按 score 降序输出最多 3 条；只输出 JSON\n"
    '输出格式：{"matched":[{"id":1,"score":0.95,"reason":"..."}]}'
)


VERIFY_SYSTEM_PROMPT = (
    "你是知识库候选核验器。给定用户问题和若干候选QA（问题+答案），"
    "判断每个候选是否能直接回答用户问题。\n"
    "规则：\n"
    "1. 语义相近、能直接回答该问题 → can_answer=true；只是同话题但答非所问 → false\n"
    "2. 严格判断，宁可拒绝，不可硬凑\n"
    "3. 只输出 JSON\n"
    '输出格式：{"verified":[{"id":1,"can_answer":true,"score":0.9,"reason":"..."}]}'
)

FREE_ANSWER_SYSTEM_PROMPT = (
    "你是知识库助手。用户的问题在知识库中没有找到可用的答案，请用你自身的知识回答。\n"
    "规则：\n"
    "1. 直接、简洁、有条理地回答问题本身\n"
    "2. 如果问题涉及事实但你不确定，如实说明，不要编造\n"
    "3. 用中文回答，使用 Markdown 格式"
)

GEN_ANSWER_SYSTEM_PROMPT = (
    "你是知识库助手。基于给定的知识库QA内容回答用户问题。\n"
    "规则：\n"
    "1. 优先且忠实于知识库内容组织答案，不要添加知识库里没有的信息\n"
    "2. 知识库内容与问题部分相关时，只回答相关部分\n"
    "3. 用中文回答，使用 Markdown 格式，简洁有条理"
)


@dataclass
class RouteResult:
    intent_id: int | None
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


def _bigrams(text: str) -> set[str]:
    """按空白分段后取相邻二元组（避免跨段拼接产生假连续片段）。"""
    grams: set[str] = set()
    for seg in text.split():
        if len(seg) == 1:
            grams.add(seg)
        else:
            grams.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return grams


def _overlap_score(question: str, text: str) -> float:
    """提问与候选文本的重合度：提问的相邻二元组在文本中连续出现的比例。

    相比单字重合度，要求"怎么做"这类词组连续出现才算命中，
    高频字堆出来的虚高分大幅减少，区分度更高。
    """
    qg = _bigrams(question)
    if not qg:
        return 0.0
    hit = sum(1 for g in qg if g in text)
    return round(hit / len(qg), 2)


def _keyword_route(question: str, intents: list[dict],
                   min_score: float = 0.3) -> RouteResult:
    """关键词兜底路由（LLM 不可用时）：问题与意图名称/描述的二元组重合度打分。"""
    best, best_score = None, 0.0
    for it in intents:
        text = f"{it['name']} {it['description'] or ''}"
        score = _overlap_score(question, text)
        if score > best_score:
            best_score, best = score, it
    detail = {"engine": "keyword", "pool_size": len(intents),
              "best_score": round(best_score, 2), "threshold": min_score}
    if best and best_score >= min_score:
        return RouteResult(best["id"], round(best_score * 0.8, 2),
                           "关键词兜底", degraded=True, detail=detail)
    return RouteResult(None, 0.0, "关键词兜底未定位", degraded=True, detail=detail)


def _keyword_match(question: str, qas: list[dict],
                   min_score: float = 0.5) -> tuple[list[MatchCandidate], dict]:
    """关键词兜底精配：与 question+aliases 的二元组连续重合度。返回 (候选, 过程明细)。"""
    scored_all: list[tuple[int, float]] = []
    for qa in qas:
        text = f"{qa['question']} {qa['question_aliases'] or ''}"
        scored_all.append((qa["id"], _overlap_score(question, text)))
    kept = [MatchCandidate(i, s, "关键词兜底") for i, s in scored_all if s >= min_score]
    kept.sort(key=lambda c: -c.score)
    detail = {"engine": "keyword", "pool_size": len(qas), "threshold": min_score,
              "best_score": max((s for _, s in scored_all), default=0.0), "kept": len(kept)}
    return kept[:3], detail


def keyword_match_global(question: str, qas: list[dict],
                         min_score: float = 0.7) -> tuple[list[MatchCandidate], dict]:
    """路由失败或 L2 内无命中时的全库关键词兜底匹配。"""
    return _keyword_match(question, qas, min_score)


@dataclass
class VerifyItem:
    qa_id: int
    can_answer: bool
    score: float
    reason: str


async def verify_candidates(question: str, qas: list[dict]) -> list[VerifyItem]:
    """兜底核验：向量/关键词捞回的候选，让 LLM 判断能否直接回答用户问题。"""
    if not qas:
        return []
    cands = [{"id": q["id"], "question": q["question"],
              "answer": (q["answer"] or "")[:500]} for q in qas]
    try:
        data = await _chat_json(VERIFY_SYSTEM_PROMPT, json.dumps(
            {"question": question, "candidates": cands}, ensure_ascii=False))
    except LLMError:
        raise
    valid = {q["id"] for q in qas}
    out: list[VerifyItem] = []
    for item in data.get("verified", []):
        try:
            qa_id = int(item["id"])
        except (KeyError, ValueError, TypeError):
            continue
        if qa_id in valid:
            out.append(VerifyItem(qa_id, bool(item.get("can_answer", False)),
                                  float(item.get("score", 0) or 0),
                                  str(item.get("reason", ""))))
    out.sort(key=lambda v: -v.score)
    return out


async def free_answer(question: str, context_qas: list[dict] | None = None) -> str:
    """宽松模式自答：无可用知识库答案时，用模型自身知识回答。

    context_qas 非空时（兜底候选全部被核验拒绝），把候选作为参考上下文一并提供，
    但明确告知模型这些候选不能直接回答，仅供参考话题方向。
    """
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY 未配置")
    user_parts = [f"用户问题：{question}"]
    if context_qas:
        refs = "\n\n".join(
            f"【参考候选（已判定无法直接回答该问题，仅提供话题背景）】\n问：{q['question']}\n答：{(q['answer'] or '')[:300]}"
            for q in context_qas[:3])
        user_parts.append(refs)
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": FREE_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:200]}") from e
    except (httpx.HTTPError, KeyError, IndexError) as e:
        raise LLMError(f"LLM 调用失败: {e}") from e


async def gen_answer(question: str, qa: dict) -> str:
    """基于单条命中的 QA 生成自然语言回答（宽松模式下代替直接贴原文）。"""
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY 未配置")
    user = (f"用户问题：{question}\n\n知识库QA：\n问：{qa['question']}\n"
            f"答：{qa['answer']}")
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": GEN_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:200]}") from e
    except (httpx.HTTPError, KeyError, IndexError) as e:
        raise LLMError(f"LLM 调用失败: {e}") from e


async def route(question: str, intents: list[dict],
                route_min_score: float = 0.3) -> RouteResult:
    """Step 1：意图路由（扁平意图列表 → 选一个意图，取其挂载的知识库检索）。"""
    if not intents:
        return RouteResult(None, 0.0, "意图列表为空", detail={"engine": "none"})
    items = [{"id": it["id"], "name": it["name"],
              "description": it["description"] or "",
              "knowledge_bases": it.get("kb_names") or []} for it in intents]
    try:
        data = await _chat_json(ROUTE_SYSTEM_PROMPT, json.dumps(
            {"intents": items, "question": question}, ensure_ascii=False))
        raw_id = data.get("intent_id")
        valid = {r["id"] for r in intents}
        intent_id = int(raw_id) if raw_id not in (None, "unknown") and str(raw_id).isdigit() \
            and int(raw_id) in valid else None
        return RouteResult(intent_id, float(data.get("confidence", 0) or 0), str(data.get("reason", "")),
                           detail={"engine": "llm", "pool_size": len(intents)})
    except LLMError:
        if config.KEYWORD_FALLBACK:
            return _keyword_route(question, intents, route_min_score)
        raise


async def match_candidates(question: str, qas: list[dict],
                           min_score: float | None = None,
                           kw_min: float = 0.7, limit: int = 3) -> MatchResult:
    """Step 2：候选精配，输入为意图挂载知识库内所有 active QA。

    为控制 token，只发问题名不发别名（直配阶段已用别名做过字符匹配）。
    """
    if not qas:
        return MatchResult(detail={"engine": "none", "pool_size": 0})
    if min_score is None:
        min_score = config.MATCH_MIN_SCORE
    try:
        cands = [{"id": q["id"], "question": q["question"]} for q in qas]
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
