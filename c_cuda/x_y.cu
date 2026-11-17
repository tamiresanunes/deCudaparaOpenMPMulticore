#include "comum.h"

void calcula_x_y(){
    int i, j;
    int i_start = (int)(imax/2 + 1);
    int j_start = (int)(y_down / dx_c) + 1;

    for(i = 1; i <= imax; ++i){
        dev_x[i] = (((double)(i) - 1.0) * dx_c) - lhori / 2.0;
    }

    for(j = 1; j <= jmax; ++j){
        dev_y[j] = (((double)(j) - 1.0) * dx_c) - y_down;
    }

    for(i = i_start; i <= imax; ++i){
        double etax = (dev_x[i] - dev_x[imax]) / (dev_x[i_start] - dev_x[imax]);
        double sx = px_grid * etax + (1.0 - px_grid) * (1.0 - (tanh(q_grid * (1.0 - etax)) / tanh(q_grid)));
        dev_x[i] = dev_x[imax] - sx * (dev_x[imax] - dev_x[i_start]);
    }

    for(i = 1; i <= i_start; ++i){
        int idx = i_start - i;
        double etax = (dev_x[idx] + dev_x[imax]) / (dev_x[i_start] + dev_x[imax]);
        double sx = px_grid * etax + (1.0 - px_grid) * (1.0 - (tanh(q_grid * (1.0 - etax)) / tanh(q_grid)));
        dev_x[idx] = -dev_x[imax] - sx * (-dev_x[imax] - dev_x[i_start]);
    }

    for(j = j_start; j <= jmax; ++j){
        double etay = (dev_y[j] - dev_y[jmax]) / (dev_y[j_start] - dev_y[jmax]);
        double sy = py_grid * etay + (1.0 - py_grid) * (1.0 - (tanh(q_grid * (1.0 - etay)) / tanh(q_grid)));
        dev_y[j] = dev_y[jmax] - sy * (dev_y[jmax] - dev_y[j_start]);
    }

    for(j = 1; j <= j_start; ++j){
        int idx = j_start - j;
        double etay = (dev_y[idx] + dev_y[jmax]) / (dev_y[j_start] + dev_y[jmax]);
        double sy = py_grid * etay + (1.0 - py_grid) * (1.0 - (tanh(q_grid * (1.0 - etay)) / tanh(q_grid)));
        dev_y[idx] = -dev_y[jmax] - sy * (-dev_y[jmax] - dev_y[j_start]);
    }
}