# Run the DrishtiSR demo (Windows PowerShell)

Paste into a fresh PowerShell window:

    Set-Location D:\SIH\DrishtiSR
    powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1

The wrapper uses the project venv and the first free port of 8000/8001/8002. Without the wrapper: `D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/serve.py --port 8000`
Open the URL it prints, e.g. **http://127.0.0.1:8000/** (it opens by itself unless you pass `--no-browser`).

Port busy? Find the PID, then free it:

    Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object LocalPort, OwningProcess
    Stop-Process -Id <PID>

Stop the demo: run the two lines above on the port it printed.

Troubleshooting:
- `No module named 'uvicorn'` → a bare `python` here is `.venv-1` (Python 3.14, no deps). Check with `python -c "import sys; print(sys.executable)"`, then use the full path `D:\SIH\DrishtiSR\.venv\Scripts\python.exe` or the wrapper.
