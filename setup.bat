@echo off
setlocal enabledelayedexpansion

:: Initialize arrays for storing Python paths and versions
set "count=0"
set "found_python=0"

:: Check for Python versions in PATH
for /f "tokens=*" %%i in ('where python 2^>nul') do (
    set /a count+=1
    set "python_path[!count!]=%%i"
    for /f "tokens=*" %%v in ('%%i -V 2^>^&1') do (
        set "python_version[!count!]=%%v"
        set "found_python=1"
    )
)

:: Check WindowsApps directory for additional Python versions
for /f "tokens=*" %%i in ('dir /b /ad "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\Python*" 2^>nul') do (
    set "current_python=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\%%i\python.exe"
    if exist "!current_python!" (
        for /f "tokens=*" %%v in ('"!current_python!" -V 2^>^&1') do (
            set /a count+=1
            set "python_path[!count!]=!current_python!"
            set "python_version[!count!]=%%v"
            set "found_python=1"
        )
    )
)

if !found_python! equ 0 (
    echo Python not found
    exit /b 1
)

:: Display available Python versions
echo Available Python versions:
for /l %%i in (1,1,!count!) do (
    echo %%i. !python_version[%%i]! - !python_path[%%i]!
)

:: Prompt user to select version
set /p "choice=Enter the number of the Python version you want to use (1-!count!): "

:: Validate input
if !choice! lss 1 (
    echo Invalid selection
    exit /b 1
)
if !choice! gtr !count! (
    echo Invalid selection
    exit /b 1
)

:: Set selected Python version
set "PYTHON=!python_path[%choice%]!"
echo Selected: !python_version[%choice%]!

:setup
:: Create and set up the virtual environment
"!PYTHON!" -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    exit /b 1
)

:: Activate virtual environment and run commands
call venv\Scripts\activate.bat
python -m pip install uv
python -m uv clean
python -m uv pip install --upgrade pip
python -m uv pip install wheel
python -m uv pip install -r requirements.txt
venv\Scripts\activate & pre-commit install & pre-commit install --hook-type commit-msg & venv\Scripts\deactivate
deactivate
echo Setup complete.
