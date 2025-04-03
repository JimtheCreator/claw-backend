while ($true) {
    $process = Get-Process uvicorn -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Host "Restarting FastAPI..."
        .\scripts\run.ps1
    }
    Start-Sleep -Seconds 10
}