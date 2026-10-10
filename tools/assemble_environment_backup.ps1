param(
    [ValidateSet('main','speakers','all')][string]$Environment = 'all',
    [string]$Directory = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$downloadRoot = (Resolve-Path -LiteralPath $Directory).Path
$manifest = Get-Content -LiteralPath (Join-Path $downloadRoot 'assets-manifest.json') -Raw | ConvertFrom-Json
$names = if ($Environment -eq 'main') { @('windows-main-venv-py312.zip') } elseif ($Environment -eq 'speakers') { @('windows-speakers-venv-py312.zip') } else { @('windows-main-venv-py312.zip','windows-speakers-venv-py312.zip') }
foreach ($name in $names) {
    $asset = @($manifest.assets | Where-Object { $_.filename -eq $name })[0]
    if (-not $asset) { throw ('Missing archive record: ' + $name) }
    $destination = Join-Path $downloadRoot $name
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $asset.sha256) {
            Write-Host ('[OK] ZIP already verified: ' + $name)
            continue
        }
        throw ('An existing ZIP has a different checksum. Keep or rename it before assembling: ' + $name)
    }
    $parts = @($asset.parts | Sort-Object filename)
    if (-not $parts.Count) { throw ('Download the complete ZIP for this archive: ' + $name) }
    foreach ($part in $parts) {
        if ($part.filename -notmatch ('^' + [regex]::Escape($name) + '\.part\d{3}$')) { throw 'Invalid part filename' }
        $partPath = Join-Path $downloadRoot $part.filename
        if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) { throw ('Missing part: ' + $part.filename) }
        if ((Get-Item -LiteralPath $partPath).Length -ne $part.bytes -or (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $part.sha256) { throw ('Part checksum mismatch: ' + $part.filename) }
    }
    $temporary = $destination + '.assembling'
    if ((Test-Path -LiteralPath $temporary) -and ((Get-Item -LiteralPath $temporary).Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw 'Refusing a linked temporary file' }
    $writer = [System.IO.File]::Open($temporary, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        foreach ($part in $parts) {
            $reader = [System.IO.File]::OpenRead((Join-Path $downloadRoot $part.filename))
            try { $reader.CopyTo($writer, 1048576) } finally { $reader.Dispose() }
        }
    } finally { $writer.Dispose() }
    if ((Get-Item -LiteralPath $temporary).Length -ne $asset.bytes -or (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.sha256) { throw ('Assembled ZIP checksum mismatch: ' + $name) }
    Move-Item -LiteralPath $temporary -Destination $destination
    Write-Host ('[OK] Assembled and verified: ' + $name)
}
