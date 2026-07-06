# Waits for the running satdino-f100-s0 probe cell to finish, then starts the
# sequential grid queue (scripts/run_grid_queue.py). Launched detached so the
# hand-off survives any session; safe to re-run (the queue skips done cells).
$repo = "D:\JHU-xView3"
$probeFinal = Join-Path $repo "runs\satdino-f100-s0\final_metrics.json"

while (-not (Test-Path $probeFinal)) {
    Start-Sleep -Seconds 120
}

Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList "scripts/run_grid_queue.py" `
    -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $repo "runs\logs\grid_queue.log") `
    -RedirectStandardError (Join-Path $repo "runs\logs\grid_queue.err.log")
