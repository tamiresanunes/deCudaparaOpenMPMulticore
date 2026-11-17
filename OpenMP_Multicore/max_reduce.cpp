#include "comum.h"

// Calcula a norma infinita dos resíduos em memória compartilhada.
// A redução usa máximo absoluto, equivalente à operação CUB da versão CUDA.
double max_reduce(double *values, int imax, int jmax) {
    const int size = imax * jmax;
    double maximum = 0.0;
    #pragma omp parallel for reduction(max:maximum) schedule(static)
    for (int i = 0; i < size; ++i)
        maximum = fmax(maximum, fabs(values[i]));
    return maximum;
}
