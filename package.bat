@echo off
setlocal
cd /d "%~dp0"
set "USERPROFILE=%~dp0.build_home"
set "HOME=%~dp0.build_home"
if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"

if not defined MINDGRASP_BUILD_PYTHON set "MINDGRASP_BUILD_PYTHON=E:\XWJ\anaconda\envs\uno\python.exe"
"%MINDGRASP_BUILD_PYTHON%" -m PyInstaller --noconfirm --clean python_demo.spec
endlocal
