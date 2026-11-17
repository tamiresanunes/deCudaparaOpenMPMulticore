#include "comum.h"
#define idx i*(jmax+1)+j

// Calcula o primeiro estágio da integração temporal em paralelo.
static void calc_1(dim3 grid, dim3 block, double *dev_res_z, double *dev_zi, double *dev_z, double *dev_z_tau, double *dev_rz, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	
    if(i <= imax-1 && j <= jmax-1){
        dev_res_z[idx] = ((dev_z[idx]-dev_z_tau[idx]) + dev_rz[idx]*dt) * dtau; 
        dev_zi[idx] = dev_z_tau[idx] + dev_res_z[idx];             
    }

    });
}

// Calcula o segundo estágio da integração temporal em paralelo.
static void calc_2(dim3 grid, dim3 block, double *dev_res_z, double *dev_zi, double *dev_z, double *dev_z_tau, double *dev_rz, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_z[idx] = ((dev_z[idx]-dev_z_tau[idx]) + dev_rz[idx]*dt) * dtau;
        dev_zi[idx] = 0.75 * dev_z_tau[idx] + 0.25 * (dev_zi[idx]+dev_res_z[idx]);
    }

    });
}

// Calcula o terceiro estágio da integração temporal em paralelo.
static void calc_3(dim3 grid, dim3 block, double *dev_res_z, double *dev_zi, double *dev_z, double *dev_z_tau, double *dev_z_n_tau, double *dev_rz, int jmax, int imax, double dt, double dtau) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_z[idx] = ((dev_z[idx]-dev_z_tau[idx]) + dev_rz[idx]*dt) * dtau; 
        dev_z_n_tau[idx] = 1.0 / 3.0 * dev_z_tau[idx] + 2.0 / 3.0 * (dev_zi[idx]+dev_res_z[idx]);
    }

    });
}

//--- Solve Mixture Fraction ---
// Integra a equação de energia em três estágios.
void solve_Z(double *dev_um_n, double *dev_vm_n, double *dev_z, double *dev_z_n_tau, double *dev_z_tau, double *dev_rz){
    dim3 blockDim(16,16);
    dim3 gridDim((imax-3 + blockDim.x - 1)/blockDim.x, (jmax-3 + blockDim.y - 1)/blockDim.y);


    // RALSTON'S METHOD (Second Order Runge-Kutta)
    RESZ(dev_um_n, dev_vm_n, dev_z_tau, dev_rz);

    calc_1(gridDim, blockDim, dev_res_z, dev_zi, dev_z, dev_z_tau, dev_rz, jmax, imax, dt, dtau);

    bcZ(dev_zi);
    RESZ(dev_um_n, dev_vm_n, dev_zi, dev_rz);

    calc_2(gridDim, blockDim, dev_res_z, dev_zi, dev_z, dev_z_tau, dev_rz, jmax, imax, dt, dtau);
    
    bcZ(dev_zi);
    RESZ(dev_um_n, dev_vm_n, dev_zi, dev_rz);

    calc_3(gridDim, blockDim, dev_res_z, dev_zi, dev_z, dev_z_tau, dev_z_n_tau, dev_rz, jmax, imax, dt, dtau);

    bcZ(dev_z_n_tau);
}