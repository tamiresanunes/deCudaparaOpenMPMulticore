#include "comum.h"

// Executa a varredura pontos medios em paralelo, preservando os índices da malha original.
void pontos_medios(dim3 grid, dim3 block, double *dev_u, double *dev_v, double *dev_um, double *dev_vm, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 1;
    int j = tid_y + 1;

    if(i <= imax && j <= jmax){
        dev_u[i*(jmax+1)+j] = (dev_um[(i+1)*(jmax+1)+j]+dev_um[i*(jmax+1)+j])*0.50;
        dev_v[i*(jmax+1)+j] = (dev_vm[i*(jmax+2)+(j+1)]+dev_vm[i*(jmax+2)+j])*0.50;
    }

    });
}

// Interpola velocidades das faces para os centros das células.
void comp_mean(double *dev_u, double *dev_v, double *dev_um, double *dev_vm){
    dim3 blockDim(16, 16);
    dim3 gridDim((imax-1 + blockDim.x - 1)/blockDim.x, (jmax-1 + blockDim.y - 1)/blockDim.y);
    
    pontos_medios(gridDim, blockDim, dev_u, dev_v, dev_um, dev_vm, imax, jmax);
    
}