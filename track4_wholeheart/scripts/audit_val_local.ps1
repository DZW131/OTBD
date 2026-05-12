$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ValRoot = "D:\work\Wholeheart_Val_Dataset"
$Out = Join-Path $RepoRoot "track4_wholeheart\reports\val_audit_local.json"

python (Join-Path $RepoRoot "track4_wholeheart\scripts\audit_dataset.py") --root $ValRoot --output $Out
Write-Host "Wrote $Out"
