$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot
python .\run_embodied_demo.py --config .\configs\default_config.json --command E --pretty