Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"  # the project venv; a bare `python` may be another interpreter
$port = 8000, 8001, 8002 | Where-Object { -not (Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue) } | Select-Object -First 1
if (-not $port) { throw "Ports 8000-8002 are all busy; free one (see docs/RUN_DEMO.md)." }
& $py scripts/serve.py --port $port @args
