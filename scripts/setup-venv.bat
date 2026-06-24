@echo off
REM Bootstrap de un entorno Python LOCAL a la maquina (fuera de OneDrive).
REM
REM Por que: el proyecto se sincroniza por OneDrive entre varias maquinas, pero un venv queda
REM atado al interprete de la PC donde se creo (.venv\Scripts\python.exe apunta a una ruta fija).
REM Si ese .venv se sincroniza a otra maquina, deja de funcionar. Por eso guardamos el venv en
REM %LOCALAPPDATA% (que NO se sincroniza): cada maquina tiene el suyo, creado una sola vez.
REM
REM Deja la ruta del python del venv en la variable PYEXE para que la use quien llame a este .bat.

set "VENV_DIR=%LOCALAPPDATA%\generador-academico-rag-venv"
set "PYEXE=%VENV_DIR%\Scripts\python.exe"

REM Si el venv local ya existe y tiene las dependencias clave, no hay nada que hacer.
"%PYEXE%" -c "import uvicorn, fastapi, chromadb, pandas" >nul 2>&1
if %errorlevel%==0 exit /b 0

echo.
echo Preparando entorno Python local (solo la primera vez en esta maquina)...
echo Carpeta: %VENV_DIR%
echo.

REM Buscar un interprete base: primero el py launcher, si no python en PATH.
set "BASEPY="
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3"
if not defined BASEPY (
  python -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BASEPY=python"
)
if not defined BASEPY (
  echo ERROR: no se encontro Python. Instalalo desde https://www.python.org/downloads/
  echo        ^(marca "Add python.exe to PATH"^) y volve a ejecutar este lanzador.
  pause
  exit /b 1
)

REM (Re)crear el venv local e instalar las dependencias del proyecto.
%BASEPY% -m venv "%VENV_DIR%"
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo ERROR: fallo la instalacion de dependencias. Revisa tu conexion e intenta de nuevo.
  pause
  exit /b 1
)
echo.
echo Entorno listo.
exit /b 0
