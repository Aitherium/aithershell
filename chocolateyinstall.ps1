# Chocolatey Install Script
$ErrorActionPreference = 'Stop'

$packageName = 'aithershell'
$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
$fileName = 'aither.exe'
$filePath = Join-Path $toolsDir $fileName

# Verify file exists
if (-not (Test-Path $filePath)) {
    throw "Expected aither.exe not found at $filePath"
}

# Install to Program Files
$installDir = "$env:ProgramFiles\AitherShell"
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

Copy-Item -Path $filePath -Destination "$installDir\aither.exe" -Force

# Add to PATH
$pathItem = $installDir
$envPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
if ($envPath -notlike "*$pathItem*") {
    $envPath += ";$pathItem"
    [Environment]::SetEnvironmentVariable('PATH', $envPath, 'Machine')
}

Write-Host "✅ AitherShell installed to $installDir"
Write-Host ""
Write-Host "📖 Quick start:"
Write-Host "  1. Set your license key:"
Write-Host "     setx AITHERIUM_LICENSE_KEY `"your-license-key`""
Write-Host ""
Write-Host "  2. Run AitherShell:"
Write-Host "     aither --help"
Write-Host "     aither prompt `"Hello, AitherOS!`""
Write-Host ""
Write-Host "🔑 Get your free license at: https://aitherium.com/free"
Write-Host ""
