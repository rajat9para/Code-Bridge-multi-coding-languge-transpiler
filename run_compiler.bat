@echo off
title CodeBridge - Multi-Backend Compiler Framework
color 0A

echo ================================================================
echo                    CODEBRIDGE COMPILER
echo              Multi-Backend Compiler Framework
echo ================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo [INFO] Python found. Checking dependencies...
echo.

:: Install Python dependencies
echo [STEP 1] Installing Python dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully!
echo.

:: Start Backend Server
echo [STEP 2] Starting Backend Server on port 5000...
start "CodeBridge Backend" cmd /k "cd /d "%~dp0backend" && python app.py"
timeout /t 3 /nobreak >nul
echo [OK] Backend server started!
echo.

:: Start Frontend Server
echo [STEP 3] Starting Frontend Server on port 8080...
start "CodeBridge Frontend" cmd /k "cd /d "%~dp0frontend" && python -m http.server 8080"
timeout /t 2 /nobreak >nul
echo [OK] Frontend server started!
echo.

:: Open Browser
echo [STEP 4] Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:8080
echo.

echo ================================================================
echo                    SERVERS RUNNING!
echo ================================================================
echo.
echo  Backend:  http://localhost:5000
echo  Frontend: http://localhost:8080
echo.
echo  Close this window to stop all servers
echo  Or press Ctrl+C in server windows to stop them
echo.
echo ================================================================
echo.

:: Keep the window open
pause