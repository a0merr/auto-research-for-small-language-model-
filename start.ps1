# start.ps1 -- Launch Autoresearch on Windows
# Run this from the autoresearch folder:
#   .\start.ps1

Write-Host ""
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host "   AUTORESEARCH -- Claude-powered ML Research" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyver = python --version 2>&1
    Write-Host "  [ok] $pyver" -ForegroundColor Green
} catch {
    Write-Host "  [error] Python not found. Install from python.org" -ForegroundColor Red
    exit 1
}

# Check ANTHROPIC_API_KEY
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host ""
    Write-Host "  [!] ANTHROPIC_API_KEY not set." -ForegroundColor Yellow
    Write-Host "  Set it with:" -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."' -ForegroundColor White
    Write-Host ""
    $key = Read-Host "  Or paste your API key now (leave blank to skip agent)"
    if ($key) {
        $env:ANTHROPIC_API_KEY = $key
        Write-Host "  [ok] API key set for this session." -ForegroundColor Green
    }
}

# Install dependencies
Write-Host ""
Write-Host "  Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt -q
Write-Host "  [ok] Dependencies ready." -ForegroundColor Green

# Run data prep
Write-Host ""
Write-Host "  Preparing data (one-time setup)..." -ForegroundColor Cyan
python prepare.py

# Start dashboard server in background
Write-Host ""
Write-Host "  Starting dashboard server..." -ForegroundColor Cyan
$dashJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python dashboard/server.py
}
Start-Sleep -Seconds 1
Write-Host "  [ok] Dashboard running at http://localhost:8080" -ForegroundColor Green

# Open browser
Start-Process "http://localhost:8080"

# Start runner
Write-Host ""
Write-Host "  Starting autoresearch runner..." -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""
python runner.py --experiments 20

# Cleanup
Write-Host ""
Write-Host "  Stopping dashboard server..." -ForegroundColor Cyan
Stop-Job $dashJob
Remove-Job $dashJob
Write-Host "  Done!" -ForegroundColor Green
