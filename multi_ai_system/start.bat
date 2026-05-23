@echo off
cd /d "%~dp0"

echo.
echo === Multi-AI Orchestrator - Web Launcher ===
echo.

:: --- Check Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

:: --- Install dependencies ---
echo [1/2] Installing dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    pip install openai httpx fastapi uvicorn jinja2
)
echo [Done]

:: --- Check API Key ---
if "%DEEPSEEK_API_KEY%"=="" (
    if not exist .env (
        echo.
        echo [WARNING] DEEPSEEK_API_KEY is not set.
        echo.
        echo Please set it before running:
        echo   Option 1: set DEEPSEEK_API_KEY=sk-your-key
        echo   Option 2: create .env file with:
        echo             DEEPSEEK_API_KEY=sk-your-key
        echo.
        pause
        exit /b 1
    )
)

:: --- Start server ---
echo.
echo [2/2] Starting web server...
echo.
echo Server: http://localhost:5000
echo Press Ctrl+C to stop
echo.

python web_app.py

pause
