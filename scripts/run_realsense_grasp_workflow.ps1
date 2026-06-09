param(
    [string]$Choice = "",
    [switch]$NoVis
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$env:Path = "$torchLib;$env:Path"

$argsList = @("-B", ".\run_realsense_grasp_workflow.py")
if ($Choice -ne "") {
    $argsList += @("--choice", $Choice)
}
if ($NoVis) {
    $argsList += "--no-vis"
}

python @argsList
