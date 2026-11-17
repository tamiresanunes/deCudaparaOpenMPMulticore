#include "comum.h"

// Executa a varredura calc vol u em paralelo, preservando os índices da malha original.
void calc_vol_u(dim3 grid, dim3 block, double *dev_x, double *dev_ym, double *dev_vol_u, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 1;

    if(i <= imax && j <= jmax){
        dev_vol_u[i*(jmax+1)+j] = (dev_x[i]-dev_x[i-1]) * (dev_ym[j+1]-dev_ym[j]); 
    }

    });
}

// Executa a varredura calc vol v em paralelo, preservando os índices da malha original.
void calc_vol_v(dim3 grid, dim3 block, double *dev_xm, double *dev_y, double *dev_vol_v, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 1;
    int j = tid_y + 2;

    if(i <= imax && j <= jmax)
        dev_vol_v[i*(jmax+2)+j] = (dev_y[j]-dev_y[j-1]) * (dev_xm[i+1]-dev_xm[i]); 

    });
}

// Executa a varredura calc vol p em paralelo, preservando os índices da malha original.
void calc_vol_p(dim3 grid, dim3 block, double *dev_xm, double *dev_ym, double *dev_vol_p, int imax, int jmax) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 1;
    int j = tid_y + 2;

    if(i <= imax && j <= jmax)
            dev_vol_p[i*(jmax+1)+j] = (dev_ym[j+1]-dev_ym[j]) * (dev_xm[i+1]-dev_xm[i]); 

    });
}

// Calcula volumes das células de velocidade e pressão.
void calcula_vol(){
    dim3 blockDim(16,16);
    dim3 gridDimVolVP((imax + blockDim.x - 1)/blockDim.x, (jmax-1 + blockDim.y - 1)/blockDim.y);
    dim3 gridDimVolU((imax-1 + blockDim.x - 1)/blockDim.x, (jmax + blockDim.y - 1)/blockDim.y);
    calc_vol_u(gridDimVolVP, blockDim, dev_x, dev_ym, dev_vol_u, imax, jmax);
    calc_vol_p(gridDimVolVP, blockDim, dev_xm, dev_ym, dev_vol_p, imax, jmax);
    calc_vol_v(gridDimVolU, blockDim, dev_xm, dev_y, dev_vol_v, imax, jmax);
}