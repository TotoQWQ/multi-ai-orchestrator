#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    多AI协作编排系统 - Web 版启动脚本      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 安装依赖
echo "[1/2] 检查并安装依赖..."
pip3 install -r requirements.txt -q 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] pip install 失败，尝试手动安装..."
    pip3 install openai httpx fastapi uvicorn jinja2
fi
echo "[完成] 依赖检查通过"

# 检查 API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    if [ ! -f ".env" ]; then
        echo ""
        echo "[提示] 未设置 DEEPSEEK_API_KEY"
        echo "       请在 .env 文件中添加：DEEPSEEK_API_KEY=你的key"
        echo "       或设置环境变量：export DEEPSEEK_API_KEY=你的key"
        echo ""
        read -p "请输入你的 DeepSeek API Key: " DEEPSEEK_API_KEY
        export DEEPSEEK_API_KEY
    fi
fi

# 启动服务
echo ""
echo "[2/2] 启动 Web 服务..."
echo ""
echo "服务地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

python3 web_app.py
