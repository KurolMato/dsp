param(
    [ValidateSet('run', 'music', 'speakers', 'base')][string]$Mode = 'run',
    [switch]$CheckOnly,
    [switch]$EnsureOnly
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)

function Quote-ProcessArgument([string]$Value) {
    # Classic py.exe parses the version selector from the raw command line;
    # quoting "-3.12" makes it fall through to the default Python instead.
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Probe-Python([string]$Executable, [string[]]$PrefixArgs = @()) {
    try {
        $query = "import sys,struct,json; print(json.dumps(dict(version=list(sys.version_info[:2]),bits=struct.calcsize('P')*8,prefix=sys.prefix,base=getattr(sys,'_base_executable',sys.executable))))"
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $Executable
        $allArgs = @($PrefixArgs) + @('-c', $query)
        $info.Arguments = (($allArgs | ForEach-Object { Quote-ProcessArgument $_ }) -join ' ')
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        $null = $process.Start()
        if (-not $process.WaitForExit(15000)) {
            $process.Kill()
            $process.Dispose()
            return $null
        }
        $text = $process.StandardOutput.ReadToEnd()
        $code = $process.ExitCode
        $process.Dispose()
        if ($code -ne 0) { return $null }
        $data = $text.Trim() | ConvertFrom-Json
        if ($data.version[0] -ne 3 -or $data.version[1] -ne 12 -or $data.bits -ne 64) { return $null }
        return $data
    } catch { return $null }
}

function Get-LauncherVersions([string]$Executable) {
    $process = $null
    try {
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $Executable
        $info.Arguments = '-0p'
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        $null = $process.Start()
        if (-not $process.WaitForExit(15000)) { $process.Kill(); return $null }
        # Classic launchers write the list to stderr. Capture it directly so
        # Windows PowerShell does not interpret those lines as native errors.
        $listing = $process.StandardOutput.ReadToEnd() + "`n" + $process.StandardError.ReadToEnd()
        if ($process.ExitCode -eq 0) { return $listing }
    } catch { return $null }
    finally { if ($process) { $process.Dispose() } }
    return $null
}

function Find-LocalPython {
    $candidates = @()
    $launchers = @(Get-Command py.exe -CommandType Application -All -ErrorAction SilentlyContinue)
    foreach ($launcherPath in @($launchers | ForEach-Object { $_.Source } | Select-Object -Unique)) {
        # Command discovery may return several executables. Invoke one path at
        # a time; WindowsApps aliases may redirect to the Store/install manager.
        if (-not $launcherPath -or $launcherPath -match '\\WindowsApps\\') { continue }
        # List first, so the newer Python install manager cannot auto-install
        # a missing global interpreter as a side effect of probing -3.12.
        $installed = Get-LauncherVersions $launcherPath
        if ($installed -and $installed -match '3\.12') {
            $data = Probe-Python $launcherPath @('-3.12')
            if ($data -and (Test-Path -LiteralPath $data.base -PathType Leaf)) {
                Write-Host "[INFO] Found Python 3.12 through: $launcherPath"
                return [string]$data.base
            }
        }
    }
    foreach ($name in @('python.exe', 'python3.exe')) {
        foreach ($command in @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)) {
            if ($command -and $command.Source -notmatch '\\WindowsApps\\') { $candidates += $command.Source }
        }
    }
    foreach ($key in @('HKCU:\SOFTWARE\Python\PythonCore\3.12\InstallPath',
                       'HKLM:\SOFTWARE\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path -LiteralPath $key) {
            $properties = Get-ItemProperty -LiteralPath $key
            if ($properties.ExecutablePath) { $candidates += $properties.ExecutablePath }
            $directory = (Get-Item -LiteralPath $key).GetValue('')
            if ($directory) { $candidates += (Join-Path $directory 'python.exe') }
        }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $data = Probe-Python $candidate
        if ($data -and (Test-Path -LiteralPath $data.base -PathType Leaf)) { return [string]$data.base }
    }
    throw 'No usable local 64-bit Python 3.12 found. Install Python 3.12 on THIS computer, then run setup again.'
}

function Assert-WorkspaceChild([string]$Path) {
    $absolute = [System.IO.Path]::GetFullPath($Path)
    $prefix = $ProjectRoot.TrimEnd('\') + '\'
    if (-not $absolute.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Environment path is outside the project. No files have been moved.'
    }
    return $absolute
}

function Invoke-Python([string]$Executable, [string[]]$CommandArgs) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Executable @CommandArgs
        $exitStatus = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($exitStatus -ne 0) { throw "Python command failed (exit $exitStatus). See the output above." }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $envName = if ($Mode -eq 'speakers') { '.venv-speakers' } else { '.venv' }
    $envRoot = Assert-WorkspaceChild (Join-Path $ProjectRoot $envName)
    $envPython = Join-Path $envRoot 'Scripts\python.exe'
    $probe = if (Test-Path -LiteralPath $envPython -PathType Leaf) { Probe-Python $envPython } else { $null }
    $valid = $false
    if ($probe -and (Test-Path -LiteralPath $probe.base -PathType Leaf)) {
        $actualPrefix = [System.IO.Path]::GetFullPath([string]$probe.prefix).TrimEnd('\')
        $valid = $actualPrefix -eq $envRoot.TrimEnd('\')
    }
    if ($CheckOnly) {
        if (-not $valid) {
            Write-Host "[ERROR] $envName is missing or was copied from another computer. Run the setup script on this computer."
            exit 1
        }
        Write-Host "[OK] $envName interpreter works on this computer."
        exit 0
    }
    $recreated = $false
    if (-not $valid) {
        # Find a working base interpreter before touching the old environment.
        $basePython = Find-LocalPython
        if (Test-Path -LiteralPath $envRoot) {
            $item = Get-Item -LiteralPath $envRoot -Force
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw 'Environment is a link/junction. Recreate a normal project environment manually.'
            }
            $backupPath = Assert-WorkspaceChild ($envRoot + '.broken_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6))
            Write-Host "[INFO] Keeping the unusable environment at $backupPath"
            Move-Item -LiteralPath $envRoot -Destination $backupPath
        }
        Write-Host "[INFO] Creating $envName with this computer's Python: $basePython"
        Invoke-Python $basePython @('-m', 'venv', $envRoot)
        if (-not (Probe-Python $envPython)) { throw 'The new environment could not start.' }
        $recreated = $true
    }
    if ($EnsureOnly) {
        Write-Host "[OK] $envName is ready; dependency installation was skipped."
        exit 0
    }
    $installMode = $Mode
    if ($Mode -eq 'run') {
        $dependencyCode = 'import numpy,scipy,matplotlib,sounddevice,soundfile,onnxruntime,torch,av; from pymss import MSSeparator,download_model,resolve_model'
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $envPython '-c' $dependencyCode 2>$null; $dependenciesReady = $LASTEXITCODE -eq 0 }
        finally { $ErrorActionPreference = $previousPreference }
        if ($recreated -or -not $dependenciesReady) { $installMode = 'music' } else { $installMode = 'none' }
    }
    if ($installMode -eq 'speakers') {
        Invoke-Python $envPython @('-X','utf8','-m','pip','install','torch==2.7.1','torchaudio==2.7.1','--index-url','https://download.pytorch.org/whl/cpu')
        Invoke-Python $envPython @('-X','utf8','-m','pip','install','-r',(Join-Path $ProjectRoot 'requirements-speakers.txt'))
    } elseif ($installMode -eq 'music') {
        Invoke-Python $envPython @('-X','utf8','-m','pip','install','torch>=2.7.1','--index-url','https://download.pytorch.org/whl/cpu')
        Invoke-Python $envPython @('-X','utf8','-m','pip','install','-r',(Join-Path $ProjectRoot 'requirements-music.txt'))
    } elseif ($installMode -eq 'base') {
        Invoke-Python $envPython @('-X','utf8','-m','pip','install','-r',(Join-Path $ProjectRoot 'requirements.txt'))
    }
    if ($installMode -ne 'none') { Invoke-Python $envPython @('-X','utf8','-m','pip','check') }
    if ($installMode -eq 'music') {
        # pip check only checks package metadata, not whether native libraries load.
        Invoke-Python $envPython @('-X','utf8','-c','import torch,av; from pymss import MSSeparator,download_model,resolve_model; print("Music dependencies OK")')
    }
    if ($Mode -eq 'run') {
        Invoke-Python $envPython @('-X','utf8',(Join-Path $ProjectRoot 'check_env.py'))
        Invoke-Python $envPython @('-X','utf8',(Join-Path $ProjectRoot 'app.py'))
    } else { Write-Host '[OK] Setup complete. Start run_windows.bat.' }
    exit 0
} catch {
    Write-Host ('[ERROR] ' + $_.Exception.Message)
    exit 1
}
