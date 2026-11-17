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

## Organização

- `comum.*` e `functions.h`: estado compartilhado e interfaces do solver.
- `cpu_runtime.h`: execução paralela, alocação e temporização na CPU.
- `res*.cpp`, `solve_*.cpp`, `upwind_*.cpp`: resíduos, transporte e integração temporal.
- `bc*.cpp`: condições de contorno.
- `initial.cpp`, `output.cpp`: entrada, reinício e saída.
- `tests/smoke`: configuração mínima para verificar compilação e um passo.

As matrizes ainda usam os nomes `dev_*` herdados do original para facilitar a comparação fórmula a fórmula; são vetores comuns da CPU.
