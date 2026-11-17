#include "comum.h"
#define idx i*(jmax+2)+j

// Calcula o primeiro estágio da integração temporal em paralelo.
void calc_1(dim3 grid, dim3 block, double *dev_res_v, double *dev_vm, double *dev_vm_tau, double *dev_rv, double *dev_vi, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 3;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_v[idx] = ((dev_vm[idx]-dev_vm_tau[idx]) + dev_rv[idx]*dt) * dtau;
        dev_vi[idx] = (dev_vm_tau[idx] + dev_res_v[idx]);
    }

    });
}

// Calcula o segundo estágio da integração temporal em paralelo.
void calc_2(dim3 grid, dim3 block, double *dev_res_v, double *dev_vm, double *dev_vm_tau, double *dev_rv, double *dev_vi, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 3;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_v[idx] = ((dev_vm[idx]-dev_vm_tau[idx]) + dev_rv[idx]*dt) * dtau;
        dev_vi[idx] = ((double)(3.0/4.0) * dev_vm_tau[idx] + (double)(1.0/4.0) * (dev_vi[idx] + dev_res_v[idx]));
    }

    });
}

// Calcula o terceiro estágio da integração temporal em paralelo.
void calc_3(dim3 grid, dim3 block, double *dev_res_v, double *dev_vm, double *dev_vm_tau, double *dev_vm_n_tau, double *dev_rv, double *dev_vi, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 3;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_v[idx] =((dev_vm[idx]-dev_vm_tau[idx]) +  dev_rv[idx]*dt) * dtau;
        dev_vm_n_tau[idx] = (double)(1.0 / 3.0) * dev_vm_tau[idx] + (double)(2.0 / 3.0) * (dev_vi[idx] + dev_res_v[idx]);
    }

    });
}

// Integra a velocidade vertical em três estágios e retorna seu maior resíduo.
double solve_V(double *dev_um, double *dev_vm, double *dev_vm_n, double *dev_um_tau, double *dev_vm_tau, double *dev_vm_n_tau, double *dev_p, double *dev_t, double *dev_vi, double *dev_rv, double *dev_res_v){
    dim3 blockDim(16,16);
    dim3 gridDim((imax-3 + blockDim.x - 1)/blockDim.x, (jmax-4 + blockDim.y - 1)/blockDim.y);
    
    RESV(dev_um_tau, dev_vm_tau, dev_p, dev_t, dev_rv);

    calc_1(gridDim, blockDim, dev_res_v, dev_vm, dev_vm_tau, dev_rv, dev_vi, jmax, imax, dt, dtau);

    bcUV(dev_um_tau, dev_vi);
    RESV(dev_um_tau, dev_vi, dev_p, dev_t, dev_rv);

    calc_2(gridDim, blockDim, dev_res_v, dev_vm, dev_vm_tau, dev_rv, dev_vi, jmax, imax, dt, dtau);

    bcUV(dev_um_tau, dev_vi);
    RESV(dev_um_tau, dev_vi, dev_p, dev_t, dev_rv);

    calc_3(gridDim, blockDim, dev_res_v, dev_vm, dev_vm_tau, dev_vm_n_tau, dev_rv, dev_vi, jmax, imax, dt, dtau);


    bcUV(dev_um_tau, dev_vm_n_tau);

    double residual_v = max_reduce(dev_res_v, imax-1, jmax); 
    return residual_v;
}