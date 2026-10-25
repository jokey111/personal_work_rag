"""应用配置：从环境变量 / .env 读取，单人部署友好。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# --- 基础 ---
APP_NAME = "personal_work_rag"
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "faq.db"

# --- 认证（单人） ---
ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin")
# 强口令：必须通过环境变量或 .env 设置，长度 >= 8
ADMIN_PASSWORD = _env("ADMIN_PASSWORD", "")
JWT_SECRET = _env("JWT_SECRET", "")
JWT_EXPIRE_HOURS = int(_env("JWT_EXPIRE_HOURS", "720"))  # 30 天

# --- LLM（OpenAI 兼容接口） ---
LLM_BASE_URL = _env("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_API_KEY = _env("LLM_API_KEY", "")
LLM_MODEL = _env("LLM_MODEL", "glm-4.7-flash")  # 免费档即可
LLM_TIMEOUT_SECONDS = float(_env("LLM_TIMEOUT_SECONDS", "30"))

# --- 兜底：LLM 不可用时走关键词匹配 ---
KEYWORD_FALLBACK = _env("KEYWORD_FALLBACK", "1") == "1"
# Step2 精配的最低命中分（低于此分的 LLM 候选视为"擦边"不返回，触发全库二次兜底）
MATCH_MIN_SCORE = float(_env("MATCH_MIN_SCORE", "0.7"))

# 默认阈值（可在页面「匹配参数」中运行时调整，存 settings 表；此处为首次默认值）
ROUTE_MIN_SCORE = float(_env("ROUTE_MIN_SCORE", "0.3"))          # 关键词路由定位线
MATCH_MIN_SCORE = float(_env("MATCH_MIN_SCORE", "0.7"))          # LLM 精配最低命中分
KEYWORD_MATCH_MIN_SCORE = float(_env("KEYWORD_MATCH_MIN_SCORE", "0.5"))  # 关键词精配命中线

# --- 备份保留天数 ---
BACKUP_KEEP_DAYS = int(_env("BACKUP_KEEP_DAYS", "30"))


def validate() -> list[str]:
    """启动时校验必填项，返回错误列表（空=通过）。"""
    errors = []
    if len(ADMIN_PASSWORD) < 8:
        errors.append("ADMIN_PASSWORD 未设置或长度不足 8 位")
    if len(JWT_SECRET) < 16:
        errors.append("JWT_SECRET 未设置或长度不足 16 位")
    if not LLM_API_KEY:
        errors.append("LLM_API_KEY 未设置（匹配功能将不可用，仅关键词兜底）")
    return errors
