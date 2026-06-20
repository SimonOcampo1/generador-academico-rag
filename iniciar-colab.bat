@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Generador Academico RAG (LLM en Colab)
echo ===============================================
echo   Generador Academico RAG - modo Colab (GPU)
echo ===============================================
echo.

REM URL FIJA del tunel de Colab. Editar UNA sola vez con tu dominio ngrok (el mismo del notebook).
REM El LLM corre en Colab; este .bat solo levanta la web local apuntando alla.
set "OLLAMA_URL=https://astound-cottage-smile.ngrok-free.dev"
set "OLLAMA_MODEL=qwen2.5:14b-instruct"
REM Techo de tokens alto: el 14b escribe mas y NO loopea, asi no corta los artefactos a la mitad.
REM (es un tope; si el modelo termina antes, no agrega tiempo)
set "OLLAMA_NUM_PREDICT=1200"

echo Antes de seguir: el notebook colab\generar_en_colab.ipynb tiene que estar corriendo
echo (Entorno de ejecucion -^> Ejecutar todo). Backend: %OLLAMA_URL%
echo.

REM No se enciende Ollama local: rag.py apunta a OLLAMA_URL (el tunel de Colab).
if exist ".venv\Scripts\python.exe" (
  start "WebApp" /min cmd /c ".venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
) else (
  start "WebApp" /min cmd /c "python -m uvicorn app.main:app --port 8000"
)

timeout /t 5 >nul
start "" http://localhost:8000

echo Listo. Se abrio http://localhost:8000 (generacion via Colab).
echo Manten la pestana de Colab abierta durante la demo.
pause
