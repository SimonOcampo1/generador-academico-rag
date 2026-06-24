@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Generador Academico RAG
echo ===============================================
echo   Generador Academico RAG - iniciando...
echo ===============================================
echo.

REM 0) Modelo local: Phi-4-mini (3.8B), mejor redaccion/grounding que modelos mas chicos en CPU.
REM    Modelo siempre cargado, toda la CPU a una sola generacion.
set "OLLAMA_MODEL=phi4-mini"
set "OLLAMA_KEEP_ALIVE=30m"
set "OLLAMA_NUM_PARALLEL=1"

REM 1) Encender el modelo local (Ollama). Si ya corre, no pasa nada.
start "Ollama" /min cmd /c "ollama serve"
timeout /t 2 >nul

REM 2) Entorno Python local a esta maquina (se crea/instala solo la primera vez).
call scripts\setup-venv.bat
if %errorlevel% neq 0 goto :fin

REM 3) Levantar la web app (la primera vez construye las bases: SQLite + Chroma).
start "WebApp" /min "%PYEXE%" -m uvicorn app.main:app --port 8000

REM 4) Abrir el navegador
timeout /t 5 >nul
start "" http://localhost:8000

echo Listo. Se abrio http://localhost:8000
echo Para cerrar todo, cerra las ventanas "Ollama" y "WebApp".

:fin
pause
