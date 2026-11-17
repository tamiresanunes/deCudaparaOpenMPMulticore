#include "comum.h"
#define idx i*(jmax+1)+j

// Calcula o primeiro estágio da integração temporal em paralelo.
static void calc_1(dim3 grid, dim3 block, double *dev_res_c, double *dev_ci, double *dev_c, double *dev_c_tau, double *dev_rc, int jmax, int imax, double dtau, double dt) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	
    if(i <= imax-1 && j <= jmax-1){
        dev_res_c[idx] = ((dev_c[idx]-dev_c_tau[idx]) + dev_rc[idx]*dt) * dtau; 
        dev_ci[idx] = dev_c_tau[idx] + dev_res_c[idx];        
    }

    });
}

// Calcula o segundo estágio da integração temporal em paralelo.
static void calc_2(dim3 grid, dim3 block, double *dev_res_c, double *dev_ci, double *dev_c, double *dev_c_tau, double *dev_rc, int jmax, int imax, double dtau, double dt) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	if(i <= imax-1 && j <= jmax-1){    
        dev_res_c[idx] = ((dev_c[idx]-dev_c_tau[idx]) + dev_rc[idx]*dt) * dtau; 
        dev_ci[idx] = 0.75 * dev_c_tau[idx] + 0.25 * (dev_ci[idx] + dev_res_c[idx]);   
        
    }

    });
}

// Calcula o terceiro estágio da integração temporal em paralelo.
static void calc_3(dim3 grid, dim3 block, double *dev_res_c, double *dev_ci, double *dev_c, double *dev_c_tau, double *dev_c_n_tau, double *dev_rc, int jmax, int imax, double dtau, double dt) {
    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {
    int i = tid_x + 2;
    int j = tid_y + 2;
	if(i <= imax-1 && j <= jmax-1){
        dev_res_c[idx] = ((dev_c[idx]-dev_c_tau[idx]) + dev_rc[idx]*dt) * dtau; 
        dev_c_n_tau[idx] = 1.0 / 3.0 * dev_c_tau[idx] + 2.0 / 3.0 * (dev_ci[idx] + dev_res_c[idx]);            
    } 

    });
}

//--- solve_C - concentration ---
// Integra a equação de concentração em três estágios.
void solve_C(double *dev_um_n, double *dev_vm_n, double *dev_c, double *dev_c_n_tau, double *dev_c_tau, double *dev_rc){
    dim3 blockDim(16,16);
    dim3 gridDim((imax-3 + blockDim.x - 1)/blockDim.x, (jmax-3 + blockDim.y - 1)/blockDim.y);
    
    // RALSTON'S METHOD (Second Order Runge-Kutta)
    RESC(dev_um_n, dev_vm_n, dev_c_tau, dev_rc);

    calc_1(gridDim, blockDim, dev_res_c, dev_ci, dev_c, dev_c_tau, dev_rc, jmax, imax, dtau, dt);

    bcC(dev_ci);
    RESC(dev_um_n, dev_vm_n, dev_ci, dev_rc);
    
    calc_2(gridDim, blockDim, dev_res_c, dev_ci, dev_c, dev_c_tau, dev_rc, jmax, imax, dtau, dt);

    bcC(dev_ci);
    RESC(dev_um_n, dev_vm_n, dev_ci, dev_rc);

    calc_3(gridDim, blockDim, dev_res_c, dev_ci, dev_c, dev_c_tau, dev_c_n_tau, dev_rc, jmax, imax, dtau, dt);

    bcC(dev_c_n_tau);
}