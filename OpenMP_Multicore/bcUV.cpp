#include "comum.h"

// Executa a varredura contorno inferior em paralelo, preservando os índices da malha original.
static void contorno_inferior(dim3 grid, dim3 block, double *dev_um, double *dev_vm, int imax, int jmax, double v_i) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 1;
    if(i > imax) return;
    
    dev_vm[i*(jmax+2)+1] = v_i;
    dev_vm[i*(jmax+2)+2] = v_i;
    
    if(i >= 2){
        dev_um[i*(jmax+1)+1] = 0.0;
    }

    });
}

// Executa a varredura contorno superior em paralelo, preservando os índices da malha original.
static void contorno_superior(dim3 grid, dim3 block, double *dev_um, double *dev_vm, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    if(i > imax) return;
    
    dev_vm[i*(jmax+2)+jmax] = 2.0*dev_vm[i*(jmax+2)+(jmax-2)] - dev_vm[i*(jmax+2)+(jmax-1)];
    dev_vm[i*(jmax+2)+(jmax+1)] = 2.0*dev_vm[i*(jmax+2)+(jmax-1)] - dev_vm[i*(jmax+2)+jmax];
    dev_um[i*(jmax+1)+jmax] = dev_um[i*(jmax+1)+(jmax-1)];

    });
}

// Executa a varredura contorno esquerdo em paralelo, preservando os índices da malha original.
static void contorno_esquerdo(dim3 grid, dim3 block, double *dev_um, double *dev_vm, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int j = tid_x + 1;
    if(j > jmax+1) return;
    
    if(j <= jmax){
        dev_um[2*(jmax+1)+j] = 0.0;
        dev_um[1*(jmax+1)+j] = 0.0;
    }
    
    dev_vm[1*(jmax+2)+j] = dev_vm[2*(jmax+2)+j];

    });
}

// Executa a varredura contorno direito em paralelo, preservando os índices da malha original.
static void contorno_direito(dim3 grid, dim3 block, double *dev_um, double *dev_vm, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int j = tid_x + 1;
    if(j > jmax) return;
    
    dev_um[imax*(jmax+1)+j] = 0.0;
    dev_um[(imax+1)*(jmax+1)+j] = 0.0;
    
    if(j >= 2){
        dev_vm[imax*(jmax+2)+j] = dev_vm[(imax-1)*(jmax+2)+j];
    }

    });
}

// Aplica condições de contorno das duas componentes de velocidade.
void bcUV(double *dev_um, double *dev_vm){
    int threads = 256;
    int blocks_i = grid_1d(imax, threads);
    int blocks_j = grid_1d(jmax+1, threads);
    
    contorno_inferior(blocks_i, threads, dev_um, dev_vm, imax, jmax, v_i);
    contorno_superior(blocks_i, threads, dev_um, dev_vm, imax, jmax);
    contorno_esquerdo(blocks_j, threads, dev_um, dev_vm, imax, jmax);
    contorno_direito(blocks_j, threads, dev_um, dev_vm, imax, jmax);
}