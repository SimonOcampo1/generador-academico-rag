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
REM keep_alive infinito: el modelo queda FIJO en la GPU de Colab toda la sesion. Sin esto se
REM descarga por inactividad y la siguiente generacion arranca en frio (~60 s para recargar el
REM 14B antes del primer token). Con -1, solo la 1ra generacion es lenta; el resto arranca en ~3 s.
set "OLLAMA_KEEP_ALIVE=-1"

echo Antes de seguir: el notebook colab\generar_en_colab.ipynb tiene que estar corriendo
echo (Entorno de ejecucion -^> Ejecutar todo). Backend: %OLLAMA_URL%
echo.

REM Entorno Python local a esta maquina (se crea/instala solo la primera vez).
call scripts\setup-venv.bat
if %errorlevel% neq 0 goto :fin

REM No se enciende Ollama local: rag.py apunta a OLLAMA_URL (el tunel de Colab).
start "WebApp" /min "%PYEXE%" -m uvicorn app.main:app --port 8000

timeout /t 5 >nul
start "" http://localhost:8000

echo Listo. Se abrio http://localhost:8000 (generacion via Colab).
echo Manten la pestana de Colab abierta durante la demo.

:fin
pause
