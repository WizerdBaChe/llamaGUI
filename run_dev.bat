@echo off
setlocal EnableDelayedExpansion
title LlamaGUI - Dev Mode

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "PYTHONPATH=%ROOT%\app"
set "LLAMAGUI_ROOT=%ROOT%"
set "LLAMAGUI_VARIANT=cuda"
set "LLAMAGUI_BIN=%ROOT%\bin\cuda"
set "LLAMAGUI_GRADIO_PORT=7860"
set "LLAMAGUI_API_PORT=8000"

echo.
echo LlamaGUI  -  Dev Mode  (system Python)
echo ================================================
echo Root : %ROOT%
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found in PATH.
    echo Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Show Python path for debugging
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set "SYS_PY=%%i"
echo Python : %SYS_PY%

:: Check / install required packages
python -c "import gradio, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing required packages...
    pip install gradio "fastapi[standard]" "uvicorn[standard]" pywebview pystray pillow --quiet
    if errorlevel 1 (
        echo ERROR: Package install failed.
        pause
        exit /b 1
    )
)

:: Verify Gradio version
for /f "delims=" %%v in ('python -c "import gradio; print(gradio.__version__)"') do set "GR_VER=%%v"
echo Gradio  : v%GR_VER%
echo.

echo Starting LlamaGUI...
echo Gradio  -^> http://127.0.0.1:%LLAMAGUI_GRADIO_PORT%
echo FastAPI -^> http://127.0.0.1:%LLAMAGUI_API_PORT%
echo.

python "%ROOT%\app\main.py"

echo.
echo LlamaGUI exited.
pause
endlocal
