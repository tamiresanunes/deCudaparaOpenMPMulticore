# Solver multicore com OpenMP

Esta pasta contém a conversão em C++17 do solver em `../c_cuda`. As equações e a sequência de etapas foram mantidas. Cada antigo kernel percorre a mesma grade de índices em memória da CPU por meio de `cpu_for_each`; as varreduras são paralelizadas com OpenMP e completam antes da etapa seguinte. `max_reduce.cpp` usa uma redução OpenMP para a norma infinita dos resíduos.

## Compilar

Requisitos: compilador C++17 com OpenMP e CMake 3.16 ou superior.

```sh
cmake -S OpenMP_Multicore -B OpenMP_Multicore/build -DCMAKE_BUILD_TYPE=Release
cmake --build OpenMP_Multicore/build --config Release
```

No Windows com MSYS2 UCRT64, use o ambiente UCRT64 e o compilador `C:/msys64/ucrt64/bin/c++.exe`. A execução depende das DLLs desse ambiente estarem no `PATH`.

No PowerShell, também é possível compilar sem CMake com `./OpenMP_Multicore/build.ps1` a partir da raiz do repositório. O script prepara `build/` e copia os arquivos de entrada.

## Executar

Execute a partir do diretório `OpenMP_Multicore/build`, pois os arquivos de entrada e saída usam caminhos relativos:

```sh
cd OpenMP_Multicore/build
OMP_NUM_THREADS=4 ./cylinder_solver 204 800000
```

Os argumentos são `imax` e o limite de iterações de pseudotempo por passo. Sem argumentos, são usados 204 e 800000. `input/iterations.dat` define, entre outros parâmetros, o tempo físico final e o modo de inicialização (0: condições iniciais; 1: reinício; 2: reinício com extensão de domínio). O executável salva resultados em `data/`.

## Verificação curta

```sh
cd OpenMP_Multicore/build
ctest --output-on-failure
```

O teste usa uma malha 12 × 19 e um único passo físico com dois limites de iteração. Compare os campos com a versão CUDA para validar equivalência numérica no seu ambiente; diferenças pequenas por ordem de operações de ponto flutuante são esperadas. O código CUDA original permanece em `../c_cuda`. O script `convert.py` registra o processo de conversão, mas os arquivos `.cpp` resultantes já fazem parte do projeto.

## Comparar numericamente com CUDA

Execute as duas versões com `imax=12`, `itc_max=2` e os mesmos arquivos de `tests/smoke/input/`. Use pastas de trabalho distintas, pois ambas escrevem `data/output_variables.dat` e `data/restart/*.dat`. O executável já existente `../c_cuda/cylinder_solver.out` é Linux ELF; rode em Linux ou WSL com GPU CUDA. Confirme que ele foi compilado da mesma revisão do código CUDA antes de usá-lo como referência.

No Linux/WSL, a partir da raiz do repositório:

```sh
mkdir -p OpenMP_Multicore/validation/cuda/input OpenMP_Multicore/validation/cuda/data/restart
cp OpenMP_Multicore/tests/smoke/input/*.dat OpenMP_Multicore/validation/cuda/input/
cd OpenMP_Multicore/validation/cuda
../../../c_cuda/cylinder_solver.out 12 2
```

Rode a versão OpenMP em `OpenMP_Multicore/tests/smoke` conforme a seção anterior. Depois, na raiz do repositório:

```sh
python OpenMP_Multicore/tests/compare_outputs.py OpenMP_Multicore/validation/cuda OpenMP_Multicore/tests/smoke
```

O comparador verifica o mesmo número de linhas, rejeita `NaN`/infinito e mostra o erro máximo e RMS de `x`, `y`, `u`, `v`, `p`, `T`, `C` e dos campos de reinício. O critério padrão é `|CUDA − OpenMP| <= 2e-6 + 1e-5 * max(|CUDA|, |OpenMP|)` por valor. A saída usa seis casas decimais, por isso uma comparação exata de ponto flutuante não é apropriada. Ajuste com `--atol` e `--rtol` e registre os valores escolhidos. Um resultado `EQUIVALENTES dentro das tolerancias` com código de saída zero atesta equivalência para **esse caso de teste e essas tolerâncias**. Repita com mais passos e tamanhos de malha para ampliar a validação.

## Organização

- `comum.*` e `functions.h`: estado compartilhado e interfaces do solver.
- `cpu_runtime.h`: execução paralela, alocação e temporização na CPU.
- `res*.cpp`, `solve_*.cpp`, `upwind_*.cpp`: resíduos, transporte e integração temporal.
- `bc*.cpp`: condições de contorno.
- `initial.cpp`, `output.cpp`: entrada, reinício e saída.
- `tests/smoke`: configuração mínima para verificar compilação e um passo.

As matrizes ainda usam os nomes `dev_*` herdados do original para facilitar a comparação fórmula a fórmula; são vetores comuns da CPU.

## Bateria de estresse OpenMP

No PowerShell, a partir da raiz do repositório, compile e execute:

```powershell
.\OpenMP_Multicore\build.ps1
python .\OpenMP_Multicore\tests\run_stress.py --timeout 120 --max-grid 512
```

O roteiro cria casos independentes em `build/stress_runs/`, força `start_mode=0` e testa malhas 24, 128, 256 e 384; 1, 2 e 4 threads; tempos finais 0.005, 0.025 e 0.05; e `itc_max` de 2, 5 e 10. Em seguida tenta uma escada de malhas de 320 até `--max-grid`, parando no primeiro problema. `summary.csv` registra duração, código de saída, linhas, extremos de temperatura e concentração e maiores magnitudes. `stdout.log` e `stderr.log` ficam na pasta de cada caso. `PASS` exige saída completa e finita. `UNSTABLE` é um alerta conservador para este fixture: temperatura fora de `[0, 2]` ou concentração fora de `[-0.1, 1.1]`. A temperatura inicial é 1 e a concentração inicial está entre 0 e 1. `CRASH`, `BAD_OUTPUT` e `TIMEOUT` são resultados distintos. O limite observado depende da máquina, dos dados e do prazo por caso. `--smoke` roda apenas os dois casos pequenos, e `--max-grid 1024` estende a escada. Este teste não demonstra convergência física da solução.
