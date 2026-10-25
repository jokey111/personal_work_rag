# personal_work_rag · 个人知识库（面试 FAQ）

单人使用的个人知识库系统：多知识库 + 二级意图体系 + 两级 LLM 意图路由匹配 + 复盘闭环。
设计文档见：《面试FAQ题库_方案二详细设计》（v0.2 多库版）。

## 功能

- **多知识库**：面试 FAQ / 个人项目 / 课程知识库……提问时选库，场景预过滤
- **二级意图体系**：L1 大类 → L2 子意图 → QA 条目，分类管理，随时增删
- **语义泛化匹配**：两级 LLM 路由（Step1 意图分类 → Step2 候选精配），换说法也能命中；LLM 不可用时自动降级关键词匹配
- **QA 管理**：标准问题 / 变体问法 / 答案 / 常见追问 / STAR 要点 / 质量自评
- **复盘闭环**：命中计数、高频 Top10、未命中新问题池（一键转正为 QA）、有用/没用反馈
- **部署简单**：FastAPI + SQLite(WAL) 单机，一键脚本

## 技术栈

Python 3.10+ · FastAPI · SQLite · Vue3（CDN 单页，无构建） · OpenAI 兼容 LLM API

## 快速开始（本地）

```bash
cd personal_work_rag
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# Linux/macOS: .venv/bin/pip install -r requirements.txt

cp .env.example .env   # 填入 ADMIN_PASSWORD / JWT_SECRET / LLM_API_KEY

.venv/Scripts/python -m uvicorn app.main:app --port 8000   # Linux: .venv/bin/...
# 打开 http://127.0.0.1:8000 （前端静态由 Nginx 托管，本地开发直接访问 /docs 看 API）
```

## 服务器部署（Ubuntu/CentOS）

```bash
# 1. 服务器上拉代码（示例放家目录；任意目录均可，用 APP_DIR 告诉部署脚本即可）
git clone https://github.com/jokey111/personal_work_rag.git ~/personal_work_rag
cd ~/personal_work_rag

# 2. 一键部署（装依赖、建 .env、起 systemd、配 Nginx、每日备份；家目录会自动开 Nginx 遍历权限）
bash deploy/deploy.sh

# 3. 编辑 .env 填入 LLM_API_KEY，重启
vim .env && sudo systemctl restart personal-rag
```

访问 `http://服务器IP:8090/`（默认 8090 端口，避免与其他服务冲突；换端口：`APP_PORT=8080 bash deploy/deploy.sh`）。
**记得在云厂商安全组放行对应 TCP 端口。**
有域名后：改 `deploy/nginx.conf` 的 server_name → `sudo certbot --nginx -d 你的域名`。

## 匹配链路

```
面试官原话
  → Step1 意图路由：LLM 从意图树(L1/L2含描述)分类，输出 l1/l2/confidence
  → Step2 候选精配：该 L2 下 QA(问题+变体) 语义选 Top-3
  → 命中: hit_count+1 | 未命中: 进新问题池 → 复盘后一键转正
```

temperature=0；每次匹配 2 次调用 <1000 tokens，glm-4.7-flash 免费档即可。

## 目录结构

```
app/
  config.py    # 配置（.env）
  db.py        # SQLite 建表/连接（WAL）
  llm.py       # 两级路由 + 关键词兜底
  main.py      # FastAPI 全部端点
web/
  index.html   # Vue3 单页（登录/匹配/管理/统计）
deploy/
  deploy.sh    # 服务器一键部署
  nginx.conf   # Nginx 模板
  personal-rag.service  # systemd
data/          # SQLite 数据（git 忽略）
```

## API 概览

`POST /api/login` · `GET/POST/PUT/DELETE /api/kbs` · `GET /api/intents?kb_id=` ·
`POST/PUT/DELETE /api/intents/l1|l2` · `GET/POST/PUT/DELETE /api/qas` ·
`POST /api/match` · `POST /api/log/{id}/feedback` ·
`GET /api/stats/overview|hot|unmatched` · `POST /api/qas/from_log/{id}`

FastAPI 自带交互文档：`/docs`

## 安全提示

- 全部接口（除 login/health）需 JWT
- `.env` 与 `data/` 不入 git；题库数据库每日自动备份（保留 30 天）
- 请勿把 8000 端口直接暴露公网，统一走 Nginx 80/443
