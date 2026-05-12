$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ProjectDir
$ProjectName = Split-Path -Leaf $ProjectDir
$ArchivePath = Join-Path $ParentDir "test_nnunetv2_tta_server.tar.gz"

if (Test-Path $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath
}

tar `
  --exclude="*/__pycache__" `
  --exclude="*.pyc" `
  --exclude="*.pyo" `
  --exclude="*/checkpoint_best.pth" `
  --exclude="*/training_log_*.txt" `
  --exclude="*/progress.png" `
  --exclude="*/debug.json" `
  -czf $ArchivePath `
  -C $ParentDir `
  $ProjectName

Write-Host "Created: $ArchivePath"
