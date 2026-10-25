#!/usr/bin/env bash
# personal_work_rag 一键部署脚本（在云服务器上执行）
# 用法：bash deploy/deploy.sh
set -euo pipefail

APP_DIR="/opt/personal_work_rag"
echo "==> 1/6 安装系统依赖"
if command -v apt-get >/dev/null; then
    apt-get update -y && apt-get install -y python3-venv nginx sqlite3
elif command -v yum >/dev/null; then
    yum install -y python3 nginx sqlite3
else
    echo "未识别的包管理器，请手动安装 python3-venv nginx sqlite3" && exit 1
fi

echo "==> 2/6 Python 虚拟环境与依赖"
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "==> 3/6 初始化 .env（已存在则跳过）"
if [ ! -f .env ]; then
    ADMIN_PW=$(openssl rand -base64 16 | tr -d '/+=' | head -c 16)
    JWT_SK=$(openssl rand -hex 32)
    cat > .env <<EOF
ADMIN_USERNAME=admin
ADMIN_PASSWORD=${ADMIN_PW}
JWT_SECRET=${JWT_SK}
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=请填入你的APIKey
LLM_MODEL=glm-4.7-flash
EOF
    echo "    已生成 .env，初始密码：${ADMIN_PW}（登录后请自行修改 .env 并重启）"
    echo "    ⚠️ 请编辑 ${APP_DIR}/.env 填入 LLM_API_KEY"
fi

echo "==> 4/6 systemd 服务"
cp deploy/personal-rag.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now personal-rag

echo "==> 5/6 Nginx"
rm -f /etc/nginx/sites-enabled/default
cp deploy/nginx.conf /etc/nginx/sites-available/personal-rag.conf 2>/dev/null || cp deploy/nginx.conf /etc/nginx/conf.d/personal-rag.conf
ln -sf /etc/nginx/sites-available/personal-rag.conf /etc/nginx/sites-enabled/personal-rag.conf 2>/dev/null || true
nginx -t && systemctl reload nginx

echo "==> 6/6 每日备份 cron（SQLite → data/backups/，保留30天）"
( crontab -l 2>/dev/null || true; echo "0 4 * * * sqlite3 ${APP_DIR}/data/faq.db \".backup '${APP_DIR}/data/faq.db.bak'\" && find ${APP_DIR}/data -name 'faq.db.bak*' -mtime +30 -delete" ) | crontab -

echo ""
echo "✅ 部署完成：http://服务器IP/  （健康检查 /api/health）"
echo "   数据库：${APP_DIR}/data/faq.db    配置：${APP_DIR}/.env"
