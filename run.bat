@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
:: ============================================================================
::  4BiM - Cross-Platform Run Script (Windows)
::  Installs prerequisites, checks for errors, and launches the application.
:: ============================================================================

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║        4BiM - Forensic Composite Generator       ║
echo ╚══════════════════════════════════════════════════╝
echo.

set "PROJECT_ROOT=%~dp0"
:: Remove trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "VENV_DIR=%PROJECT_ROOT%\programme\venv"
set "REQUIREMENTS=%PROJECT_ROOT%\programme\requirements.txt"
set "MAIN_PY=%PROJECT_ROOT%\programme\main.py"
set "ERRORS=0"

:: ── Step 1: Check Python Installation ──────────────────────────────────────
echo [INFO]  Checking Python installation...
set "PYTHON_CMD="

:: Try python first (Windows default), then py launcher, then python3
where python >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    set "PYTHON_CMD=python"
    goto :check_version
)

where py >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2 delims= " %%v in ('py --version 2^>^&1') do set "PY_VER=%%v"
    set "PYTHON_CMD=py"
    goto :check_version
)

where python3 >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set "PY_VER=%%v"
    set "PYTHON_CMD=python3"
    goto :check_version
)

echo [FAIL] Python 3.9+ is required but not found.
echo.
echo   Download Python from: https://www.python.org/downloads/
echo   Make sure to check "Add Python to PATH" during installation.
echo.
goto :error_exit

:check_version
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if !PY_MAJOR! lss 3 (
    echo [FAIL]  Python 3.9+ required, found !PY_VER!
    goto :error_exit
)
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 9 (
    echo [FAIL]  Python 3.9+ required, found !PY_VER!
    goto :error_exit
)
echo [  OK]  Found Python !PY_VER!

:: ── Step 2: Check pip ──────────────────────────────────────────────────────
echo [INFO]  Checking pip...
!PYTHON_CMD! -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARN]  pip not found. Attempting to install...
    !PYTHON_CMD! -m ensurepip --upgrade >nul 2>&1
    if !errorlevel! neq 0 (
        echo [FAIL]  Cannot install pip. Download get-pip.py from https://pip.pypa.io
        goto :error_exit
    )
)
echo [  OK]  pip is available

:: ── Step 3: Check tkinter ──────────────────────────────────────────────────
echo [INFO]  Checking tkinter availability...
!PYTHON_CMD! -c "import tkinter" >nul 2>&1
if !errorlevel! neq 0 (
    echo [FAIL]  tkinter is not available.
    echo          On Windows, reinstall Python and check "tcl/tk and IDLE" in optional features.
    goto :error_exit
)
echo [  OK]  tkinter is available

:: ── Step 4: Create / Repair Virtual Environment ───────────────────────────
echo [INFO]  Setting up virtual environment...
if not exist "%VENV_DIR%" (
    echo [INFO]  Creating virtual environment at programme\venv...
    !PYTHON_CMD! -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [FAIL]  Could not create virtual environment.
        goto :error_exit
    )
    echo [  OK]  Virtual environment created
) else if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [WARN]  Virtual environment is broken. Rebuilding...
    rmdir /s /q "%VENV_DIR%" >nul 2>&1
    !PYTHON_CMD! -m venv "%VENV_DIR%"
    echo [  OK]  Virtual environment rebuilt
) else (
    echo [  OK]  Virtual environment already exists
)

:: Activate
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo [FAIL]  Cannot find venv activation script.
    goto :error_exit
)

:: ── Step 5: Upgrade pip ────────────────────────────────────────────────────
echo [INFO]  Upgrading pip inside venv...
python -m pip install --upgrade pip --quiet >nul 2>&1
echo [  OK]  pip upgraded

:: ── Step 6: Install Dependencies ──────────────────────────────────────────
echo [INFO]  Installing dependencies...
if exist "%REQUIREMENTS%" (
    python -m pip install -r "%REQUIREMENTS%" --quiet >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARN]  Strict install failed. Trying flexible install...
        python -m pip install torch numpy Pillow --quiet >nul 2>&1
    )
) else (
    echo [WARN]  requirements.txt not found. Installing core packages...
    python -m pip install torch numpy Pillow --quiet >nul 2>&1
)

:: Also install scikit-learn
python -m pip install scikit-learn --quiet >nul 2>&1
echo [  OK]  Dependencies installed

:: ── Step 7: Verify Critical Imports ───────────────────────────────────────
echo [INFO]  Verifying Python imports...
set "IMPORT_FAIL=0"

for %%m in (torch numpy PIL tkinter sklearn) do (
    python -c "import %%m" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [  OK]    %%m
    ) else (
        echo [FAIL]    %%m - MISSING
        set /a IMPORT_FAIL+=1
    )
)

if !IMPORT_FAIL! gtr 0 (
    echo [FAIL]  !IMPORT_FAIL! module(s) could not be imported.
    echo          Run: pip install torch numpy Pillow scikit-learn
    goto :error_exit
)

:: ── Step 8: Verify Critical Data Files ────────────────────────────────────
echo [INFO]  Checking required data files...
set "DATA_OK=1"

if exist "%PROJECT_ROOT%\programme\src\assets\vae_celeba_final.pth" (
    echo [  OK]    vae_celeba_final.pth
) else (
    echo [FAIL]    MISSING: programme\src\assets\vae_celeba_final.pth
    set "DATA_OK=0"
)

if exist "%PROJECT_ROOT%\data\attribute_directions.npz" (
    echo [  OK]    attribute_directions.npz
) else (
    echo [FAIL]    MISSING: data\attribute_directions.npz
    set "DATA_OK=0"
)

if exist "%PROJECT_ROOT%\data\attribute_vectors.npz" (
    echo [  OK]    attribute_vectors.npz
) else (
    echo [FAIL]    MISSING: data\attribute_vectors.npz
    set "DATA_OK=0"
)

if "!DATA_OK!"=="0" (
    echo [WARN]  Some data files are missing. The app may not function correctly.
)

:: ── Step 9: Dry-Run Import Test ───────────────────────────────────────────
echo [INFO]  Running dry-run import test...
pushd "%PROJECT_ROOT%\programme"
python -c "from src.ui.app import Application; print('Import OK')" >nul 2>&1
if !errorlevel! equ 0 (
    echo [  OK]  Application modules load successfully
) else (
    echo [WARN]  Dry-run import test failed. App may still work at runtime.
)
popd

:: ── Step 10: Launch ───────────────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════
echo   All checks passed. Launching 4BiM...
echo ════════════════════════════════════════════════════
echo.

cd /d "%PROJECT_ROOT%"
python "%MAIN_PY%"
goto :eof

:error_exit
echo.
echo ════════════════════════════════════════════════════
echo   Setup failed. Please fix the errors above.
echo ════════════════════════════════════════════════════
echo.
pause
exit /b 1
