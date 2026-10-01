@echo off
rem Double-click to start Chunkyard. It opens in your browser; close this window to stop it.
cd /d "%~dp0"
where uv >nul 2>nul
if %errorlevel%==0 (
  uv run python -m chunkyard
) else (
  "%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" run python -m chunkyard
)
pause
