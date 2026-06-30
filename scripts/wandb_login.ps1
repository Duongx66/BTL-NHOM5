Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$wandbExe = "C:\Users\Duong\AppData\Local\Python\pythoncore-3.14-64\Scripts\wandb.exe"
$pythonExe = "C:\Users\Duong\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if (Test-Path $wandbExe) {
    & $wandbExe login
}
elseif (Test-Path $pythonExe) {
    & $pythonExe -m wandb login
}
else {
    Write-Error "Could not find W&B CLI. Install dependencies with: pip install -r requirements.txt"
}
