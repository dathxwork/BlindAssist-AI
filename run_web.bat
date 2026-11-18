@echo off
chcp 65001 > nul
echo =========================================================
echo    BlindAssist AI v3.0 - Khoi dong Giao dien Web Chinh
echo    Dia chi truy cap: http://localhost:8000
echo =========================================================
cd /d "%~dp0"
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
pause
