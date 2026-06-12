@echo off
setlocal
cd /d "%~dp0"

call package.bat
if errorlevel 1 exit /b %errorlevel%

if not defined MINDGRASP_CONDA_ENV set "MINDGRASP_CONDA_ENV=E:\XWJ\anaconda\envs\uno"
if not defined MINDGRASP_CONDA_PACK set "MINDGRASP_CONDA_PACK=E:\XWJ\anaconda\Scripts\conda-pack.exe"

set "RUNTIME_PARENT=%~dp0dist\python_demo\runtime"
set "RUNTIME_DIR=%RUNTIME_PARENT%\python"
set "RUNTIME_ZIP=%RUNTIME_PARENT%\python_env.zip"

if not exist "%RUNTIME_PARENT%" mkdir "%RUNTIME_PARENT%"
if exist "%RUNTIME_DIR%" rmdir /s /q "%RUNTIME_DIR%"
if exist "%RUNTIME_ZIP%" del /f /q "%RUNTIME_ZIP%"

"%MINDGRASP_CONDA_PACK%" -p "%MINDGRASP_CONDA_ENV%" -o "%RUNTIME_ZIP%" --ignore-missing-files --force
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%RUNTIME_ZIP%' -DestinationPath '%RUNTIME_DIR%' -Force"
if errorlevel 1 exit /b %errorlevel%
if exist "%RUNTIME_ZIP%" del /f /q "%RUNTIME_ZIP%"

echo.
echo [MindGrasp] Portable package is ready:
echo   %~dp0dist\python_demo\python_demo.exe
echo.
echo Copy the whole dist\python_demo folder to the target PC. Do not copy only python_demo.exe.
endlocal
