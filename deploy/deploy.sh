#!/usr/bin/env bash
# personal_work_rag 一键部署脚本（在云服务器上执行）
# 用法：bash deploy/deploy.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/personal_work_rag}"   # 部署目录，默认根目录下；可 APP_DIR=/opt/xxx bash deploy/deploy.sh 覆盖
APP_PORT="${APP_PORT:-8090}"   # 对外端口，默认 8090；80 被占可 APP_PORT=8080 bash deploy/deploy.sh

echo "==> 0/6 检查部署目录与端口 ${APP_PORT} 占用"
if [ ! -d "$APP_DIR" ]; then
    echo "❌ 目录不存在：${APP_DIR}（请先 git clone 到该目录，或用 APP_DIR=xxx 指定实际目录）"
    exit 1
fi
if ss -ltn "( sport = :${APP_PORT} )" | grep -q LISTEN; then
    echo "❌ 端口 ${APP_PORT} 已被占用（可能是 Docker demo 或其他服务），请换端口重试："
    echo "   APP_PORT=<其他端口> bash deploy/deploy.sh"
    ss -ltnp "sport = :${APP_PORT}" 2>/dev/null || true
    exit 1
fi

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
sed "s#/personal_work_rag#${APP_DIR}#g" deploy/personal-rag.service > /etc/systemd/system/personal-rag.service
systemctl daemon-reload
systemctl enable --now personal-rag

echo "==> 5/6 Nginx（对外端口 ${APP_PORT}）"
rm -f /etc/nginx/sites-enabled/default
sed -e "s#root .*#root ${APP_DIR}/web;#" deploy/nginx.conf > /etc/nginx/sites-available/personal-rag.conf 2>/dev/null || sed -e "s#root .*#root ${APP_DIR}/web;#" deploy/nginx.conf > /etc/nginx/conf.d/personal-rag.conf
ln -sf /etc/nginx/sites-available/personal-rag.conf /etc/nginx/sites-enabled/personal-rag.conf 2>/dev/null || true
sed -i "s/^    listen .*/    listen ${APP_PORT};/" /etc/nginx/sites-available/personal-rag.conf 2>/dev/null || sed -i "s/^    listen .*/    listen ${APP_PORT};/" /etc/nginx/conf.d/personal-rag.conf
nginx -t && systemctl reload nginx

echo "==> 6/6 每日备份 cron（SQLite，保留30天）"
( crontab -l 2>/dev/null || true; echo "0 4 * * * sqlite3 ${APP_DIR}/data/faq.db \".backup '${APP_DIR}/data/faq.db.bak'\" && find ${APP_DIR}/data -name 'faq.db.bak*' -mtime +30 -delete" ) | crontab -

echo ""
echo "✅ 部署完成：http://服务器IP:${APP_PORT}/  （健康检查 http://服务器IP:${APP_PORT}/api/health）"
echo "   数据库：${APP_DIR}/data/faq.db    配置：${APP_DIR}/.env"
echo "   ⚠️ 云厂商安全组需放行 TCP ${APP_PORT} 入站"
