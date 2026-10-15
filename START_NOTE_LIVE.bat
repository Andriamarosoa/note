@echo off
setlocal
cd /d "%~dp0"
if exist ".venv-live\Scripts\python.exe" (
  ".venv-live\Scripts\python.exe" scripts\live_notes.py --microphone %*
) else (
  echo Creez d'abord l'environnement avec les commandes du guide LIVE.md.
  exit /b 1
)
if errorlevel 1 pause
