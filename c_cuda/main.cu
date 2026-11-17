//Artificial Compressibility Methods
//Solve Momentum Equation with QUICK Scheme
//BASED ON:
//Versteeg, H. K., and W. Malalasekera. 
//"An introduction to computational Fluid Dynamics, The finite volume control, ed." (1995).
#include "comum.h"
#include <math.h>
#define N_IMAX 51*4
#define N_ITC 800000

__global__ void atualizar_matrizes_linearizadas(double *origem, double *destino, int tamanhoLinha, int tamanhoColuna, int inicio, int coluna){
    int i = blockIdx.x * blockDim.x + threadIdx.x + inicio;
    int j = blockIdx.y * blockDim.y + threadIdx.y + inicio;
    
    if(i <= tamanhoLinha && j <= tamanhoColuna){
        destino[i*(coluna)+j] = origem[i*(coluna)+j];
    }
}

__global__ void atualiza_tc(double *dev_t, double *dev_c, int *dev_flag, double temp_cylinder, double concentracao_inicial, int c_f, int imax, int jmax){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(i > imax || j > jmax) return;

    int idx = i*(jmax+1)+j;
    if(dev_flag[idx] != c_f){
        dev_t[idx] = temp_cylinder;
        dev_c[idx] = concentracao_inicial;
    }
}

int main(int argc, char *argv[]){
    int itc, tr;
    int n_imax, n_itc;
    dim3 blockDim(16, 16);  
    
    if(argc < 3){
        n_imax = N_IMAX;
        n_itc = N_ITC;
    }else{
        n_imax = atoi(argv[1]);
        n_itc = atoi(argv[2]);
    }
    
    calcular(n_imax, n_itc);         // define imax, jmax, dx_c, etc.
    alocar_globais(); 
    
    dim3 gridDim((imax + blockDim.x - 1)/blockDim.x, (jmax + blockDim.y - 1)/blockDim.y);
    dim3 gridDimUm((imax+1 + blockDim.x - 1)/blockDim.x, (jmax + blockDim.y - 1)/blockDim.y);
    dim3 gridDimVm((imax + blockDim.x - 1)/blockDim.x, (jmax+1 + blockDim.y - 1)/blockDim.y);
    double *dev_rz;
    double *dev_vi, *dev_rv, *dev_res_v;
    double *dev_ui, *dev_ru, *dev_res_u;
    double *dev_rc;
    double *dev_um, *dev_vm;
    double *dev_um_n, *dev_vm_n;
    double *dev_um_tau, *dev_vm_tau;
    double *dev_um_n_tau, *dev_vm_n_tau;
    cudaMalloc((void**)&dev_rc, sizeof(double)*(imax+1)*(jmax+1));
    cudaMalloc((void**)&dev_ru, sizeof(double)*(imax+2)*(jmax+1));
    cudaMalloc((void**)&dev_ui, sizeof(double)*(imax+2)*(jmax+1));
    cudaMalloc((void**)&dev_res_u, sizeof(double)*(imax+2)*(jmax+1));
    cudaMallocManaged((void**)&dev_rv, sizeof(double)*(imax+1)*(jmax+2));
    cudaMallocManaged((void**)&dev_vi, sizeof(double)*(imax+1)*(jmax+2));
    cudaMallocManaged((void**)&dev_res_v, sizeof(double)*(imax+1)*(jmax+2));
    cudaMalloc((void**)&dev_rz, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_um, sizeof(double)*(imax+2)*(jmax+1));
    cudaMallocManaged((void**)&dev_vm, sizeof(double)*(imax+1)*(jmax+2));
    cudaMalloc((void**)&dev_um_n, sizeof(double)*(imax+2)*(jmax+1));
    cudaMallocManaged((void**)&dev_um_tau, sizeof(double)*(imax+2)*(jmax+1));
    cudaMallocManaged((void**)&dev_um_n_tau, sizeof(double)*(imax+2)*(jmax+1));
    cudaMalloc((void**)&dev_vm_n, sizeof(double)*(imax+1)*(jmax+2));
    cudaMallocManaged((void**)&dev_vm_tau, sizeof(double)*(imax+1)*(jmax+2));
    cudaMallocManaged((void**)&dev_vm_n_tau, sizeof(double)*(imax+1)*(jmax+2));
    
    double *dev_u, *dev_v, *dev_p, *dev_pn, *dev_h, *dev_t, *dev_z;
    double *dev_c, *dev_t_n_tau, *dev_t_tau, *dev_c_n_tau, *dev_c_tau; 
    int mid_i = imax / 2;
    int mid_j = jmax / 2;
    int mid_idx = mid_i * (jmax + 1) + mid_j;
    cudaMallocManaged((void**)&dev_u, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_v, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_p, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_t, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_c, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_pn, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_h, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_z, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_t_n_tau, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_t_tau, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_c_n_tau, sizeof(double)*(imax+1)*(jmax+1));
    cudaMallocManaged((void**)&dev_c_tau, sizeof(double)*(imax+1)*(jmax+1));

    double residual_p, residual_u, residual_v, error;

    //=================================================================
    // CUDA Events — Instrumentação de tempo (NÃO altera lógica numérica)
    //=================================================================
    cudaEvent_t ev_ts_start, ev_ts_stop;          // timestep total
    cudaEvent_t ev_solveU_start, ev_solveU_stop;  // solve_U
    cudaEvent_t ev_solveV_start, ev_solveV_stop;  // solve_V
    cudaEvent_t ev_solveP_start, ev_solveP_stop;  // solve_P
    cudaEvent_t ev_solveZ_start, ev_solveZ_stop;  // solve_Z
    cudaEvent_t ev_solveC_start, ev_solveC_stop;  // solve_C
    cudaEvent_t ev_h2d_start, ev_h2d_stop;        // H2D (prefetch)
    cudaEvent_t ev_d2h_start, ev_d2h_stop;        // D2H (sync back)

    cudaEventCreate(&ev_ts_start);    cudaEventCreate(&ev_ts_stop);
    cudaEventCreate(&ev_solveU_start); cudaEventCreate(&ev_solveU_stop);
    cudaEventCreate(&ev_solveV_start); cudaEventCreate(&ev_solveV_stop);
    cudaEventCreate(&ev_solveP_start); cudaEventCreate(&ev_solveP_stop);
    cudaEventCreate(&ev_solveZ_start); cudaEventCreate(&ev_solveZ_stop);
    cudaEventCreate(&ev_solveC_start); cudaEventCreate(&ev_solveC_stop);
    cudaEventCreate(&ev_h2d_start);   cudaEventCreate(&ev_h2d_stop);
    cudaEventCreate(&ev_d2h_start);   cudaEventCreate(&ev_d2h_stop);

    float time_solveU, time_solveV, time_solveP, time_solveZ, time_solveC;
    float time_h2d, time_d2h, time_timestep;
    // Acumuladores por timestep (soma das sub-iterações pseudo-tempo)
    float acc_solveU, acc_solveV, acc_solveP, acc_solveZ, acc_solveC;
    float acc_h2d, acc_d2h;

    //duration = omp_get_wtime()
    /*  character(len=128) :: pwd
        REAL ETIME, clockTIME, TARRAY(2)
        clockTIME = ETIME(TARRAY)
        CALL idate(hoje)   ! hoje(1)=day, (2)=month, (3)=year
        CALL itime(agora)     ! agora(1)=hour, (2)=minute, (3)=second
        CALL get_environment_variable('PWD', pwd)

    --- Input data - Fill NAMELIST iterations and ref
    --- Read iterations.dat and reference.dat
    --- Compute Too, Tsub, Tinf 
    */
    init(); 

    too = ref.ts * ref.tn_too;
    tsup = ref.ts / ref.ts;
    tinf = 1.0;  //Temperatura ambiente

    /*--- Log de informações ---
        5 FORMAT ( 'Date ', i2.2, '/', i2.2, '/', i4.4, &
                '; time ',i2.2, ':', i2.2, ':', i2.2 )
        WRITE(*,5)  hoje(2), hoje(1), hoje(3), agora
        WRITE(*,*) 'Current working directory: ',trim(pwd)
        WRITE(*,*) '----------------------'
        WRITE(*,*) 'Tsup =', Tsup
        WRITE(*,*) 'Tinf =', Tinf
        WRITE(*,*) '----------------------'
        WRITE(*,*) 'Mesh =',imax,'x', jmax
        WRITE(*,*) 'imax * jmax =', int(imax*jmax)
        WRITE(*,*) 'eps =',eps
        WRITE(*,*) '----------------------'
    ver se tudo isso é necessário?
    WRITE(*,*) '----------------------'
    WRITE(*,*) 'v_c ='  ,v_c        , '[m/s]'
    WRITE(*,*) 'L_c ='  ,L_c        , '[m]'
    WRITE(*,*) 't_c ='  ,L_c/v_c    , '[s]'
    WRITE(*,*) 'v_i ='  ,v_i
    WRITE(*,*) 'V_idim ='  ,v_i*v_c , '[m/s]'
    WRITE(*,*) '----------------------'
    WRITE(*,*) 'g = '  ,g , '[m/s^2]'
    WRITE(*,*) 'S = '  ,S
    WRITE(*,*) 'q = '  ,q
    WRITE(*,*) 'Pr ='  ,Pr
        WRITE(*,*) 'Re =', Re
        WRITE(*,*) 'Pe =', Pe
        WRITE(*,*) 'Fr =', Fr
        WRITE(*,*) 'InvFr^2 =', InvFr2
        WRITE(*,*) '----------------------'
    */

    //--- Initializations ---
    itc = 1; //Initial iteration
    error = 100.0;
    tr = 1;
    tempo = 0.0;
    residual_p = 0.0;

    //--- Create mesh ---
    mesh();

    double x0=0.0, xN=0.0, y0=0.0, yN=0.0, flag0=0.0, eps0=0.0, lig0=0.0;
    cudaMemcpy(&x0, dev_x + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xN, dev_x + imax, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&y0, dev_y + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&yN, dev_y + jmax, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&flag0, dev_flag + 1, sizeof(int), cudaMemcpyDeviceToHost);
    int flag_mid = 0;
    cudaMemcpy(&flag_mid, dev_flag + mid_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&eps0, dev_epsilon1 + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lig0, dev_liga_poros + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    printf("MESH_CHECK imax=%d jmax=%d x1=%.12g xN=%.12g y1=%.12g yN=%.12g flag1=%g flag_mid=%d eps_mid=%.12g lig_mid=%.12g\n", imax, jmax, x0, xN, y0, yN, flag0, flag_mid, eps0, lig0);
    for(int k=1; k<=5; ++k){
        double xx=0.0, yy=0.0; cudaMemcpy(&xx, dev_x + k, sizeof(double), cudaMemcpyDeviceToHost); cudaMemcpy(&yy, dev_y + k, sizeof(double), cudaMemcpyDeviceToHost); printf("MESH_SAMPLE k=%d x=%.12g y=%.12g\n", k, xx, yy);
    }

    for(int i = 1; i <= imax; ++i){
        for(int j = 1; j <= jmax; ++j){
            int idx = i*(jmax+1)+j;
            if(dev_flag[idx] != c_f){
                dev_t[idx] = temp_cylinder;
                dev_c[idx] = concentracao_inicial;
            }
        }
    }
    cudaDeviceSynchronize();

    //--- Set up initial flow field ---
    if(iterations.start_mode == 0){    
        IC(dev_um, dev_vm, dev_p, dev_t, dev_c, dev_pn);
    }else if(iterations.start_mode == 1){
        restart(dev_um, dev_vm, dev_p, dev_t, dev_c);
    }else if(iterations.start_mode == 2){        
        restart_dom(dev_um, dev_vm, dev_p, dev_t, dev_z, dev_h);
    }

    cudaDeviceSynchronize();
    double p0 = 0.0, t0 = 0.0, c0 = 0.0, um0 = 0.0, vm0 = 0.0;
    cudaMemcpy(&p0, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t0, dev_t, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c0, dev_c, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&um0, dev_um, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vm0, dev_vm, sizeof(double), cudaMemcpyDeviceToHost);
    double p11=0.0, t11=0.0, c11=0.0, um11=0.0, vm11=0.0;
    cudaMemcpy(&p11, dev_p + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t11, dev_t + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c11, dev_c + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&um11, dev_um + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vm11, dev_vm + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    printf("INIT_CHECK p0=%.12g p_mid=%.12g t_mid=%.12g c_mid=%.12g um_mid=%.12g vm_mid=%.12g\n", p0, p11, t11, c11, um11, vm11);

    //--- Pseudo time step ---
    dtau = 5.e-2;
    dt = 0.5e-2;

    for(int i = 1; i <= imax + 1; ++i){
        for(int j = 1; j <= jmax; ++j){
            dev_um_tau[i*(jmax+1)+j] = dev_um[i*(jmax+1)+j];
        }
    }
    for(int i = 1; i <= imax; ++i){
        for(int j = 1; j <= jmax + 1; ++j){
            dev_vm_tau[i*(jmax+2)+j] = dev_vm[i*(jmax+2)+j];
        }
    }
    for(int i = 1; i <= imax; ++i){
        for(int j = 1; j <= jmax; ++j){
            int idx = i*(jmax+1)+j;
            dev_t_tau[idx] = dev_t[idx];
            dev_c_tau[idx] = dev_c[idx];
        }
    }
    cudaDeviceSynchronize();

    /*duration = omp_get_wtime() - duration;
    printf("init %lf/n", duration);
    duration = omp_get_wtime();
    */
    //--- Physical time step ---
    while(tempo < iterations.final_time){
        tempo = tempo + dt;

        // --- CUDA Events: início do timestep ---
        cudaEventRecord(ev_ts_start);
        acc_solveU = 0.0f; acc_solveV = 0.0f; acc_solveP = 0.0f;
        acc_solveZ = 0.0f; acc_solveC = 0.0f;
        acc_h2d = 0.0f; acc_d2h = 0.0f;
        
        //--- Pseudo-time calculation starts ---
        while(itc < iterations.itc_max){
            //--- Solve Momentum Equation with QUICK Scheme ---
            cudaEventRecord(ev_solveU_start);
            residual_u = solve_U(dev_um, dev_vm, dev_um_n, dev_um_tau, dev_vm_tau, dev_um_n_tau, dev_p, dev_ui, dev_ru, dev_res_u);
            cudaEventRecord(ev_solveU_stop);
            cudaEventSynchronize(ev_solveU_stop);
            cudaEventElapsedTime(&time_solveU, ev_solveU_start, ev_solveU_stop);
            acc_solveU += time_solveU;
            if(tr == 1 && itc == 1){
                double um_after=0.0, vm_after=0.0;
                cudaMemcpy(&um_after, dev_um_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&vm_after, dev_vm_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                printf("AFTER_SOLVE_U tr=1 um_mid=%.12g vm_mid=%.12g residual_u=%.12g\n", um_after, vm_after, residual_u);
            }

            cudaEventRecord(ev_solveV_start);
            residual_v = solve_V(dev_um, dev_vm, dev_vm_n, dev_um_tau, dev_vm_tau, dev_vm_n_tau, dev_p, dev_t, dev_vi, dev_rv, dev_res_v);
            cudaEventRecord(ev_solveV_stop);
            cudaEventSynchronize(ev_solveV_stop);
            cudaEventElapsedTime(&time_solveV, ev_solveV_start, ev_solveV_stop);
            acc_solveV += time_solveV;
            if(tr == 1 && itc == 1){
                double um_after=0.0, vm_after=0.0;
                cudaMemcpy(&um_after, dev_um_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&vm_after, dev_vm_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                printf("AFTER_SOLVE_V tr=1 um_mid=%.12g vm_mid=%.12g residual_v=%.12g\n", um_after, vm_after, residual_v);
            }
            
            //--- Solve Continuity Equation ---
            cudaEventRecord(ev_solveP_start);
            residual_p = solve_P(dev_p, dev_um_n_tau, dev_vm_n_tau, dev_pn);
            cudaEventRecord(ev_solveP_stop);
            cudaEventSynchronize(ev_solveP_stop);
            cudaEventElapsedTime(&time_solveP, ev_solveP_start, ev_solveP_stop);
            acc_solveP += time_solveP;
            if(tr == 1 && itc == 1){
                double p_after=0.0, p_next=0.0, um_after=0.0, vm_after=0.0;
                cudaMemcpy(&p_after, dev_p + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&p_next, dev_pn + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&um_after, dev_um_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&vm_after, dev_vm_n_tau + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
                printf("AFTER_SOLVE_P tr=1 p_mid=%.12g p_n_mid=%.12g um_mid=%.12g vm_mid=%.12g residual_p=%.12g\n", p_after, p_next, um_after, vm_after, residual_p);
            }
            
            //--- Solve Energy Equation ---
            cudaEventRecord(ev_solveZ_start);
            solve_Z(dev_um_n_tau, dev_vm_n_tau, dev_t, dev_t_n_tau, dev_t_tau, dev_rz);
            cudaEventRecord(ev_solveZ_stop);
            cudaEventSynchronize(ev_solveZ_stop);
            cudaEventElapsedTime(&time_solveZ, ev_solveZ_start, ev_solveZ_stop);
            acc_solveZ += time_solveZ;

            cudaEventRecord(ev_solveC_start);
            solve_C(dev_um_n_tau, dev_vm_n_tau, dev_c, dev_c_n_tau, dev_c_tau, dev_rc);
            cudaEventRecord(ev_solveC_stop);
            cudaEventSynchronize(ev_solveC_stop);
            cudaEventElapsedTime(&time_solveC, ev_solveC_start, ev_solveC_stop);
            acc_solveC += time_solveC;
            /*--- check convergence ---
            CALL convergence(itc, error, residual_p, residual_u, residual_v)
            itc = itc+1
            */

            error = fmax(residual_p, residual_u);
            error = fmax(residual_v, error);

            //--- Convergence criteria ---
            if(itc != 1 && error < iterations.eps)
                break;

            //--- Update variables ---
            for(int i = 1; i <= imax + 1; ++i){
                for(int j = 1; j <= jmax; ++j){
                    dev_um_tau[i*(jmax+1)+j] = dev_um_n_tau[i*(jmax+1)+j];
                }
            }
            for(int i = 1; i <= imax; ++i){
                for(int j = 1; j <= jmax + 1; ++j){
                    dev_vm_tau[i*(jmax+2)+j] = dev_vm_n_tau[i*(jmax+2)+j];
                }
            }
            for(int i = 1; i <= imax; ++i){
                for(int j = 1; j <= jmax; ++j){
                    int idx = i*(jmax+1)+j;
                    dev_p[idx] = dev_pn[idx];
                    dev_t_tau[idx] = dev_t_n_tau[idx];
                    dev_c_tau[idx] = dev_c_n_tau[idx];
                }
            }
            cudaDeviceSynchronize();

            itc++;
        }

        //--- End of pseudo-time calculation ---
        for(int i = 1; i <= imax + 1; ++i){
            for(int j = 1; j <= jmax; ++j){
                int idx = i*(jmax+1)+j;
                dev_um[idx] = dev_um_n_tau[idx];
            }
        }
        for(int i = 1; i <= imax; ++i){
            for(int j = 1; j <= jmax + 1; ++j){
                int idx = i*(jmax+2)+j;
                dev_vm[idx] = dev_vm_n_tau[idx];
            }
        }
        for(int i = 1; i <= imax; ++i){
            for(int j = 1; j <= jmax; ++j){
                int idx = i*(jmax+1)+j;
                dev_p[idx] = dev_pn[idx];
                dev_t[idx] = dev_t_n_tau[idx];
                dev_c[idx] = dev_c_n_tau[idx];
            }
        }
        cudaDeviceSynchronize();

        if(tr <= 3){
            double p_chk=0.0, t_chk=0.0, c_chk=0.0, um_chk=0.0, vm_chk=0.0;
            cudaMemcpy(&p_chk, dev_p + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t_chk, dev_t + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&c_chk, dev_c + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&um_chk, dev_um + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&vm_chk, dev_vm + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
            printf("STEP_CHECK tr=%d p=%.12g t=%.12g c=%.12g um=%.12g vm=%.12g\n", tr, p_chk, t_chk, c_chk, um_chk, vm_chk);
        }

        /*--- Logs of time and intermediate results
        !IF (MOD(tr, n_tr) .EQ. 0) THEN
        !    WRITE(*,*) '-----------------------------------------------------------'
        !    WRITE(*,*) 'Max Residual:', error
        !    WRITE(*,*) 'Physical time:', time
        !    WRITE(*,*) '-----------------------------------------------------------'
        !    WRITE(*,*) '         dtau:',dtau   
        !    WRITE(*,*) '         dt:',dt   
        !    WRITE(*,*) '    Residual U:',residual_u
        !    WRITE(*,*) '    Residual V:',residual_v
        !    WRITE(*,*) '    Residual P:',residual_p    
        !    !WRITE(*,*) ' Artificial viscosity:',artMAX    
        !    !WRITE(*,*) ' Art Compressibility Par:',c2    
        !    WRITE(*,*) '-----------------------------------------------------------'

            !--- Output preliminary results ---
        !    CALL comp_mean(u, v, um, vm)
        !    CALL transient(u, v, p, T ,C, tr)

            !--- Output data file ---
        !    CALL output(um, vm, u, v, p, T, C, itc)
        !END IF
        */
        itc = 0;
        error = 100.0;

        // --- CUDA Events: fim do timestep ---
        cudaEventRecord(ev_ts_stop);
        cudaEventSynchronize(ev_ts_stop);
        cudaEventElapsedTime(&time_timestep, ev_ts_start, ev_ts_stop);

        // NOTA: Este código usa Unified Memory (cudaMallocManaged).
        // Não há cudaMemcpy explícito Host→Device ou Device→Host.
        // As transferências são gerenciadas implicitamente pelo driver CUDA
        // via page faults. H2D e D2H medem 0.0 pois não há cópias explícitas.
        // Para medir page migrations, use: nsys profile --trace=cuda,um
        time_h2d = 0.0f;
        time_d2h = 0.0f;

        printf("[ITER %d][CUDA] kernel_solve_U = %.6f ms\n", tr, acc_solveU);
        printf("[ITER %d][CUDA] kernel_solve_V = %.6f ms\n", tr, acc_solveV);
        printf("[ITER %d][CUDA] kernel_solve_P = %.6f ms\n", tr, acc_solveP);
        printf("[ITER %d][CUDA] kernel_solve_Z = %.6f ms\n", tr, acc_solveZ);
        printf("[ITER %d][CUDA] kernel_solve_C = %.6f ms\n", tr, acc_solveC);
        printf("[ITER %d][CUDA] H2D = %.6f ms\n", tr, time_h2d);
        printf("[ITER %d][CUDA] D2H = %.6f ms\n", tr, time_d2h);
        printf("[ITER %d][CUDA] timestep_total = %.6f ms\n", tr, time_timestep);

        tr = tr + 1;
    }
    //--- End of physical calculation ---

    /*duration = omp_get_wtime() - duration;
    printf("loop %lf\n", duration);
    duration = omp_get_wtime();
    */

    //--- Final results ---
    /*!open (550,file='data/time.dat')            
    !write (550,*) time
    !close(550)
    */
    cudaDeviceSynchronize();
    double p_end = 0.0, t_end = 0.0, c_end = 0.0, um_end = 0.0, vm_end = 0.0;
    cudaMemcpy(&p_end, dev_p + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t_end, dev_t + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c_end, dev_c + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&um_end, dev_um + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vm_end, dev_vm + mid_idx, sizeof(double), cudaMemcpyDeviceToHost);
    printf("FINAL_CHECK p_mid=%.12g t_mid=%.12g c_mid=%.12g um_mid=%.12g vm_mid=%.12g\n", p_end, t_end, c_end, um_end, vm_end);

    //--- Compute the velocity of mean points ---
    comp_mean(dev_u, dev_v, dev_um, dev_vm);
    
    #ifdef DEBUG
    transient(dev_u, dev_v, dev_p, dev_t, dev_c, itc);
    #endif
    
    //--- output data file ---
    output(dev_um, dev_vm, dev_u, dev_v, dev_p, dev_t, dev_c, itc);

    /*duration = omp_get_wtime() - duration;
    printf("post %lf\n", duration);
    */

    //=================================================================
    // CUDA Events — Destruição
    //=================================================================
    cudaEventDestroy(ev_ts_start);    cudaEventDestroy(ev_ts_stop);
    cudaEventDestroy(ev_solveU_start); cudaEventDestroy(ev_solveU_stop);
    cudaEventDestroy(ev_solveV_start); cudaEventDestroy(ev_solveV_stop);
    cudaEventDestroy(ev_solveP_start); cudaEventDestroy(ev_solveP_stop);
    cudaEventDestroy(ev_solveZ_start); cudaEventDestroy(ev_solveZ_stop);
    cudaEventDestroy(ev_solveC_start); cudaEventDestroy(ev_solveC_stop);
    cudaEventDestroy(ev_h2d_start);   cudaEventDestroy(ev_h2d_stop);
    cudaEventDestroy(ev_d2h_start);   cudaEventDestroy(ev_d2h_stop);

    //desalocando
    cudaFree(dev_um);
    cudaFree(dev_um_n);
    cudaFree(dev_um_tau);
    cudaFree(dev_um_n_tau);
    cudaFree(dev_vm);
    cudaFree(dev_vm_n);
    cudaFree(dev_vm_tau);
    cudaFree(dev_vm_n_tau);
    cudaFree(dev_u);
    cudaFree(dev_v);
    cudaFree(dev_rc);
    cudaFree(dev_p);
    cudaFree(dev_pn);
    cudaFree(dev_h);
    cudaFree(dev_t);
    cudaFree(dev_z);
    cudaFree(dev_rz);
    cudaFree(dev_c);
    cudaFree(dev_t_n_tau);
    cudaFree(dev_t_tau);
    cudaFree(dev_c_n_tau);
    cudaFree(dev_rv);
    cudaFree(dev_ru);
    cudaFree(dev_ui);
    cudaFree(dev_res_u);
    cudaFree(dev_vi);
    cudaFree(dev_res_v);
    cudaFree(dev_c_tau);
    desalocar_globais();
}