#include "comum.h"

// Executa a varredura contorno superior inferior em paralelo, preservando os índices da malha original.
static void contorno_superior_inferior(dim3 grid, dim3 block, double *dev_cn, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    if(i >= imax) return;
    
    dev_cn[i*(jmax+1)+1] = 0.0;
    dev_cn[i*(jmax+1)+jmax] = dev_cn[i*(jmax+1)+(jmax-1)] + (dev_cn[i*(jmax+1)+(jmax-1)]-dev_cn[i*(jmax+1)+(jmax-2)]);

    });
}

// Executa a varredura contorno esquerdo direito em paralelo, preservando os índices da malha original.
static void contorno_esquerdo_direito(dim3 grid, dim3 block, double *dev_cn, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int j = tid_x + 2;
    if(j >= jmax) return;
    
    dev_cn[1*(jmax+1)+j] = dev_cn[2*(jmax+1)+j];
    dev_cn[imax*(jmax+1)+j] = dev_cn[(imax-1)*(jmax+1)+j];

    });
}

// Executa a varredura contorno cantos em paralelo, preservando os índices da malha original.
static void contorno_cantos(dim3 grid, dim3 block, double *dev_cn, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    if(tid_x == 0){
        dev_cn[1*(jmax+1)+1] = dev_cn[2*(jmax+1)+1];
        dev_cn[1*(jmax+1)+jmax] = dev_cn[2*(jmax+1)+jmax];
        dev_cn[imax*(jmax+1)+1] = dev_cn[(imax-1)*(jmax+1)+1];
        dev_cn[imax*(jmax+1)+jmax] = dev_cn[(imax-1)*(jmax+1)+jmax];
    }

    });
}

// Aplica condições de contorno da concentração, deixando os cantos para o final.
void bcC(double *dev_cn){
    int threads = 256;
    int blocks_h = grid_1d(imax-2, threads);
    int blocks_v = grid_1d(jmax-2, threads);
    
    
    
    

    contorno_superior_inferior(blocks_h, threads, dev_cn, imax, jmax);
    contorno_esquerdo_direito(blocks_v, threads, dev_cn, imax, jmax);

    
    
    
    contorno_cantos(1, 1, dev_cn, imax, jmax);
    

    
    
}