#ifndef CPU_RUNTIME_H
#define CPU_RUNTIME_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// Geometria de uma varredura. Mantida para preservar os limites dos kernels originais.
struct dim3 {
    int x, y, z;
    dim3(int x_ = 1, int y_ = 1, int z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Executa cada ponto da antiga grade CUDA uma vez; cada kernel termina antes do seguinte.
template <class Function>
void cpu_for_each(dim3 grid, dim3 block, Function function) {
    const int nx = grid.x * block.x;
    const int ny = grid.y * block.y;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            function(x, y);
}

// Vetores são zerados para evitar leitura de memória não inicializada nas bordas.
inline int cpuMalloc(void **pointer, size_t bytes) {
    *pointer = std::calloc(1, bytes);
    if (!*pointer) {
        std::fprintf(stderr, "Falha ao alocar %zu bytes\n", bytes);
        std::exit(EXIT_FAILURE);
    }
    return 0;
}
inline void cpuFree(void *pointer) { std::free(pointer); }
// Copia local usada nas sondas de diagnóstico herdadas do solver.
inline void cpuMemcpy(void *destination, const void *source, size_t bytes) {
    std::memcpy(destination, source, bytes);
}

// Temporizadores de parede substituem eventos da GPU; tempos são retornados em ms.
struct CpuEvent { std::chrono::steady_clock::time_point point; };
inline void cpuEventCreate(CpuEvent *) {}
inline void cpuEventRecord(CpuEvent &event) { event.point = std::chrono::steady_clock::now(); }
inline void cpuEventSynchronize(CpuEvent) {}
inline void cpuEventElapsedTime(float *milliseconds, CpuEvent start, CpuEvent stop) {
    *milliseconds = std::chrono::duration<float, std::milli>(stop.point - start.point).count();
}
inline void cpuEventDestroy(CpuEvent) {}

#endif
