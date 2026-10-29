"""向量检索模块：embedding 调用（OpenAI 兼容 /embeddings）+ SQLite 持久化 + 内存余弦检索。

设计（千级 QA 规模，无需 FAISS）：
  - qa_embedding 表存 float32 blob（与 qa 同库同事务，永不失同步）
  - 进程内缓存归一化矩阵 + qa_id 列表，检索=numpy 点积，毫秒级
  - QA 增删改 → upsert/remove 单条；换 embedding 模型 → 全量 rebuild
  - embedding API 挂了：检索层自动降级（vector engine 标记 unavailable），
    不影响主链路其余步骤
"""
import hashlib
import struct
from dataclasses import dataclass

import httpx

from . import config, db


class EmbeddingError(Exception):
    pass


@dataclass
class VectorHit:
    qa_id: int
    score: float  # 余弦相似度 0~1（向量已归一化）


def text_of(qa: dict) -> str:
    """QA 的向量化文本：问题 + 别名（答案不入向量，检索目标是"问对问"）。"""
    return f"{qa['question']} {qa['question_aliases'] or ''}".strip()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------- embedding API ----------

async def embed_async(texts: list[str]) -> list[list[float]]:
    """批量异步调用 /embeddings；返回与 texts 等长的向量列表。

    智谱单次 input 上限 64 条，超出自动分批串行调用。
    """
    if not texts:
        return []
    if not config.LLM_API_KEY:
        raise EmbeddingError("LLM_API_KEY 未配置，无法调用 embedding")
    BATCH = 32  # 留余量（官方上限 64，防超长文本触发 token 限制）
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            payload = {"model": config.EMBEDDING_MODEL, "input": batch}
            headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
            try:
                resp = await client.post(
                    f"{config.LLM_BASE_URL.rstrip('/')}/embeddings",
                    json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                raise EmbeddingError(f"embedding HTTP {e.response.status_code}: {e.response.text[:200]}") from e
            except (httpx.HTTPError, KeyError, IndexError) as e:
                raise EmbeddingError(f"embedding 调用失败: {e}") from e
            vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            if len(vecs) != len(batch):
                raise EmbeddingError(f"embedding 返回数量不符: {len(vecs)}/{len(batch)}")
            out.extend(vecs)
    return out


# ---------- 内存向量库 ----------

class VectorStore:
    """进程内向量缓存：启动时从 qa_embedding 表加载，写路径同步更新表和缓存。"""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._matrix = None      # numpy.ndarray (n, dim)，已 L2 归一化
        self._dim = 0
        self._model = ""
        self._ready = False      # 加载过一次（表可能为空，但结构就绪）
        self._lock_missing = False

    @property
    def available(self) -> bool:
        """缓存里有向量且维度合法时可用。"""
        return self._ready and self._matrix is not None and len(self._ids) > 0

    @property
    def count(self) -> int:
        return len(self._ids)

    def load(self) -> None:
        """从表全量加载（启动 / rebuild 后调用）。"""
        import numpy as np
        conn = db.get_conn()
        rows = conn.execute(
            "SELECT qa_id, model, dim, vec FROM qa_embedding WHERE model=?",
            (config.EMBEDDING_MODEL,)).fetchall()
        ids, vecs = [], []
        for r in rows:
            ids.append(r["qa_id"])
            vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
        if vecs:
            m = np.vstack(vecs).astype(np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = m / norms
            self._dim = self._matrix.shape[1]
        else:
            self._matrix, self._dim = None, 0
        self._ids = ids
        self._model = config.EMBEDDING_MODEL
        self._ready = True

    def upsert(self, qa_id: int, vec: list[float], text_hash: str) -> None:
        """写入表 + 更新内存（单条）。"""
        import numpy as np
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO qa_embedding(qa_id, model, dim, vec, text_hash, updated_at) "
            "VALUES(?,?,?,?,?,datetime('now','localtime')) "
            "ON CONFLICT(qa_id) DO UPDATE SET model=excluded.model, dim=excluded.dim, "
            "vec=excluded.vec, text_hash=excluded.text_hash, updated_at=excluded.updated_at",
            (qa_id, config.EMBEDDING_MODEL, len(vec),
             struct.pack(f"<{len(vec)}f", *vec), text_hash))
        conn.commit()
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        if qa_id in self._ids:
            idx = self._ids.index(qa_id)
            if self._matrix is not None and self._matrix.shape[0] > idx:
                self._matrix[idx] = v
        else:
            self._ids.append(qa_id)
            self._matrix = v.reshape(1, -1) if self._matrix is None else np.vstack([self._matrix, v])
        self._ready = True

    def remove(self, qa_id: int) -> None:
        import numpy as np
        conn = db.get_conn()
        conn.execute("DELETE FROM qa_embedding WHERE qa_id=?", (qa_id,))
        conn.commit()
        if qa_id in self._ids:
            idx = self._ids.index(qa_id)
            del self._ids[idx]
            if self._matrix is not None and self._matrix.shape[0] > idx:
                self._matrix = np.delete(self._matrix, idx, axis=0)

    def search(self, query_vec: list[float], top_k: int = 5) -> list[VectorHit]:
        """余弦相似度 top-k（query_vec 会归一化）。"""
        if not self.available:
            return []
        import numpy as np
        q = np.asarray(query_vec, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0:
            return []
        q = q / n
        scores = self._matrix @ q
        k = min(top_k, len(self._ids))
        if k <= 0:
            return []
        top_idx = np.argsort(-scores)[:k]
        return [VectorHit(self._ids[i], round(float(scores[i]), 4))
                for i in top_idx if scores[i] > 0]


def _hash_public(text: str) -> str:
    return _hash(text)


# 进程级单例
store = VectorStore()


def stale_ids() -> list[int]:
    """找出需要重算向量的 qa：缺向量 / 文本变更 / 模型不一致。"""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT qa.id, qa.question, qa.question_aliases, e.model, e.text_hash "
        "FROM qa LEFT JOIN qa_embedding e ON e.qa_id = qa.id "
        "WHERE qa.status='active'").fetchall()
    stale = []
    for r in rows:
        h = _hash(f"{r['question']} {r['question_aliases'] or ''}".strip())
        if r["model"] is None or r["text_hash"] != h or r["model"] != config.EMBEDDING_MODEL:
            stale.append(r["id"])
    return stale


async def ensure_ready() -> dict:
    """启动/调用时惰性同步：补算缺失向量。返回统计信息。"""
    if not store._ready:
        store.load()
    stale = stale_ids()
    if not stale:
        return {"ok": True, "updated": 0, "total": store.count}
    conn = db.get_conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, question, question_aliases FROM qa WHERE id IN "
        f"({','.join('?' * len(stale))}) AND status='active'", stale).fetchall()]
    if not rows:
        return {"ok": True, "updated": 0, "total": store.count}
    texts = [text_of(q) for q in rows]
    vecs = await embed_async(texts)
    for q, v in zip(rows, vecs):
        store.upsert(q["id"], v, _hash(text_of(q)))
    return {"ok": True, "updated": len(rows), "total": store.count}


async def search_async(text: str, top_k: int = 5) -> tuple[list[VectorHit], dict]:
    """检索入口：确保索引就绪 → embed 提问 → 余弦 top-k。返回 (命中, 明细)。"""
    sync = await ensure_ready()
    if not store.available:
        return [], {"engine": "vector", "available": False, "reason": "向量库为空或未就绪",
                    "synced": sync}
    vec = (await embed_async([text]))[0]
    hits = store.search(vec, top_k)
    return hits, {"engine": "vector", "available": True, "pool_size": store.count,
                  "top_k": top_k, "synced": sync}


async def rebuild_all() -> dict:
    """全量重建（换 embedding 模型或文本批量变更后调用）。"""
    conn = db.get_conn()
    conn.execute("DELETE FROM qa_embedding")
    conn.commit()
    store._ids, store._matrix, store._ready = [], None, False
    return await ensure_ready()
