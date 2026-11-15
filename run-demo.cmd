@echo off
setlocal
cd /d "%~dp0"
where node.exe >nul 2>nul
if errorlevel 1 (
  echo Node.js 18 or newer is required. Install it, then run this file again.
  pause
  exit /b 1
)
start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 1200; Start-Process 'http://127.0.0.1:4173/#experiment'"
node scripts\serve.mjs
if errorlevel 1 pause
