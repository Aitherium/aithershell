# Chocolatey Uninstall Script
$ErrorActionPreference = 'Stop'

$installDir = "$env:ProgramFiles\AitherShell"

# Remove from Program Files
if (Test-Path $installDir) {
    Remove-Item -Path $installDir -Recurse -Force
    Write-Host "✅ AitherShell uninstalled from $installDir"
}

# Optionally remove from PATH
$envPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
$newPath = $envPath -replace ";?$installDir;?", ""
if ($newPath -ne $envPath) {
    [Environment]::SetEnvironmentVariable('PATH', $newPath, 'Machine')
    Write-Host "✅ Removed from PATH"
}

Write-Host ""
Write-Host "Thank you for using AitherShell!"
Write-Host "Feedback: support@aitherium.com"
