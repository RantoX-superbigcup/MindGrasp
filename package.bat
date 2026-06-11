cd /d "%~dp0"
set "USERPROFILE=%~dp0.build_home"
set "HOME=%~dp0.build_home"
if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"
D:\anaconda3\envs\pytorch\python.exe -m PyInstaller --noconfirm --clean python_demo.spec
