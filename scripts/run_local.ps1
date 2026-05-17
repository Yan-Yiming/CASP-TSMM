param(
    [switch]$All,
    [int]$Warmup = 0,
    [int]$Runs = 1,
    [string]$ResultRoot = "web\results",
    [string]$RunId = "",
    [string]$VsDevCmd = "G:\Visual Studio\Common7\Tools\VsDevCmd.bat"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ($RunId -eq "") {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}

$ResultDir = Join-Path $ResultRoot $RunId
New-Item -ItemType Directory -Force $ResultDir | Out-Null
New-Item -ItemType Directory -Force "obj" | Out-Null

$sources = @(
    "src\benchmark.cpp",
    "src\reference.cpp",
    "src\tsmm_registry.cpp",
    "src\tsmm\naive.cpp",
    "src\tsmm\opt.cpp"
)

$compile = "call `"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && cl /nologo /std:c++17 /EHsc /O2 /openmp /D_CRT_SECURE_NO_WARNINGS $($sources -join ' ') /Fo:obj\ /Fe:obj\benchmark.exe"
cmd /c $compile

$mode = if ($All) { "--all" } else { "--required-only" }
$common = @("--output-dir", $ResultDir, $mode, "--warmup", "$Warmup", "--runs", "$Runs")

foreach ($layout in @("row", "col")) {
    Write-Host "=== Running layout: $layout ==="
    .\obj\benchmark.exe @common --layout $layout
}

python scripts\collect_gflops.py $ResultDir --csv (Join-Path $ResultDir "gflops.csv") --json (Join-Path $ResultDir "gflops_summary.json")

Write-Host "Result dir: $ResultDir"
Write-Host "GFLOPS CSV: $(Join-Path $ResultDir 'gflops.csv')"
