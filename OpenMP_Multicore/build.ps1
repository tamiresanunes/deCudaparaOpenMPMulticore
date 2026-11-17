# Compila e prepara a execução no Windows com MSYS2 UCRT64, sem depender de CMake.
param([string]$Compiler = 'C:\msys64\ucrt64\bin\c++.exe')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$build = Join-Path $root 'build'
if (-not (Test-Path -LiteralPath $Compiler)) { throw "Compilador não encontrado: $Compiler" }
New-Item -ItemType Directory -Force (Join-Path $build 'input'), (Join-Path $build 'data\restart'), (Join-Path $build 'transient\data') | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'input\iterations.dat') -Destination (Join-Path $build 'input\iterations.dat')
Copy-Item -LiteralPath (Join-Path $root 'input\reference.dat') -Destination (Join-Path $build 'input\reference.dat')
$sources = @(Get-ChildItem -LiteralPath $root -Filter '*.cpp' | ForEach-Object FullName) +
           @(Get-ChildItem -LiteralPath $root -Filter '*.c' | ForEach-Object FullName)
$env:PATH = (Split-Path -Parent $Compiler) + ';' + $env:PATH
& $Compiler -std=c++17 -O2 -fopenmp -x c++ $sources -o (Join-Path $build 'cylinder_solver.exe')
if ($LASTEXITCODE -ne 0) { throw "Compilação falhou com código $LASTEXITCODE" }
Write-Output "Executável: $(Join-Path $build 'cylinder_solver.exe')"
