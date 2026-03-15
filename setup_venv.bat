@echo off
REM ---------------------------------------------------------------
REM setup_venv.bat — Create a Python venv for the CAD Assembly Pipeline
REM
REM Usage:
REM   setup_venv.bat              (creates .venv in the current dir)
REM   setup_venv.bat C:\path\env  (creates venv at the given path)
REM
REM Requires: Python 3.10+ (cadquery/OCP wheels need >=3.10)
REM ---------------------------------------------------------------
setlocal enabledelayedexpansion

set "VENV_DIR=%~1"
if "%VENV_DIR%"=="" set "VENV_DIR=.venv"

set "SCRIPT_DIR=%~dp0"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"
set "TOPOPT_REQ_FILE=%SCRIPT_DIR%requirements-topopt.txt"
set "TOPOPT_DL4TO_REQ_FILE=%SCRIPT_DIR%requirements-topopt-dl4to.txt"
set "TOPOPT_PYMOTO_REQ_FILE=%SCRIPT_DIR%requirements-topopt-pymoto.txt"
if "%INSTALL_TOPOPT%"=="" set "INSTALL_TOPOPT=0"
if "%TOPOPT_SOLVERS%"=="" set "TOPOPT_SOLVERS=both"

REM --- Check Python is available --------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ from https://www.python.org
    exit /b 1
)

REM --- Check Python version -------------------------------------------------
for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%v"
for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version_info.major)"') do set "PY_MAJOR=%%v"
for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version_info.minor)"') do set "PY_MINOR=%%v"

if !PY_MAJOR! lss 3 (
    echo ERROR: Python 3.10+ required ^(found %PY_VER%^)
    exit /b 1
)
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 10 (
    echo ERROR: Python 3.10+ required ^(found %PY_VER%^)
    exit /b 1
)

echo Using Python %PY_VER%

REM --- Create venv -----------------------------------------------------------
echo Creating virtual environment at: %VENV_DIR%
python -m venv "%VENV_DIR%"

REM --- Activate and install --------------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
pip install --upgrade pip --quiet

echo Installing dependencies from requirements.txt...
pip install -r "%REQ_FILE%"

if "%INSTALL_TOPOPT%"=="1" (
    echo Installing optional topology-optimization stack...
    if /I "%TOPOPT_SOLVERS%"=="both" (
        if not "%TORCH_INDEX_URL%"=="" (
            echo Installing torch/torchvision from custom index: %TORCH_INDEX_URL%
            pip install torch torchvision --index-url "%TORCH_INDEX_URL%"
        ) else (
            echo Installing torch/torchvision from default index
            pip install torch torchvision
        )
        pip install -r "%TOPOPT_REQ_FILE%"
    ) else if /I "%TOPOPT_SOLVERS%"=="dl4to" (
        if not "%TORCH_INDEX_URL%"=="" (
            echo Installing torch/torchvision from custom index: %TORCH_INDEX_URL%
            pip install torch torchvision --index-url "%TORCH_INDEX_URL%"
        ) else (
            echo Installing torch/torchvision from default index
            pip install torch torchvision
        )
        pip install -r "%TOPOPT_DL4TO_REQ_FILE%"
    ) else if /I "%TOPOPT_SOLVERS%"=="pymoto" (
        pip install -r "%TOPOPT_PYMOTO_REQ_FILE%"
    ) else (
        echo ERROR: TOPOPT_SOLVERS must be one of: both, dl4to, pymoto
        exit /b 1
    )
)

echo.
echo ============================================
echo   Setup complete!
echo   Activate with:  %VENV_DIR%\Scripts\activate.bat
echo ============================================
echo.
echo Quick start:
echo   python assemble.py outer_1.step inner_1.stl -o assembly.step --render assembly.png
echo.
echo Optional native topology optimization:
echo   set INSTALL_TOPOPT=1
echo   set TOPOPT_SOLVERS=dl4to
echo   set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
echo   setup_venv.bat
echo   set TOPOPT_SOLVERS=pymoto
echo   setup_venv.bat
echo.
echo Run tests:
echo   python -m pytest test_assemble.py -v
