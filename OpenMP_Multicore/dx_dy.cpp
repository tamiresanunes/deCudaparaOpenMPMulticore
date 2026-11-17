#include "comum.h"

// Executa a varredura calc dx em paralelo, preservando os índices da malha original.
void calc_dx(dim3 grid, dim3 block, double *dev_dx, double *dev_x, int imax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    if(i <= imax)
        dev_dx[i] = dev_x[i]-dev_x[i-1];

    });
}

// Executa a varredura calc dy em paralelo, preservando os índices da malha original.
void calc_dy(dim3 grid, dim3 block, double *dev_dy, double *dev_y, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int j = tid_x + 2;
    if(j <= jmax)
        dev_dy[j] = dev_y[j]-dev_y[j-1];

    });
}

// Calcula espaçamentos locais da malha nas duas direções.
void calcula_dx_dy(){
    int threads = 256;
    dim3 blocks = grid_1d(jmax-1, threads);
    calc_dy(blocks, threads, dev_dy, dev_y, jmax);

    blocks = grid_1d(imax-1, threads);
    calc_dx(blocks, threads, dev_dx, dev_x, imax);
}