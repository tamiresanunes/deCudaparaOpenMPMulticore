#include "comum.h"

// Gera as coordenadas deslocadas usadas nas faces dos volumes.
void calcula_xm_ym(){
    int i, j;

    dev_xm[1] = -(dev_x[2] + dev_x[1]) * 0.5;
    dev_xm[imax + 1] = (dev_x[imax] + dev_x[imax - 1]) * 0.5 + (dev_x[imax] - dev_x[imax - 1]);
    for(i = 2; i <= imax; ++i){
        dev_xm[i] = (dev_x[i] + dev_x[i - 1]) * 0.5;
    }

    dev_ym[1] = dev_y[1] - (dev_y[2] - dev_y[1]) * 0.5;
    dev_ym[jmax + 1] = (dev_y[jmax] + dev_y[jmax - 1]) * 0.5 + (dev_y[jmax] - dev_y[jmax - 1]);
    for(j = 2; j <= jmax; ++j){
        dev_ym[j] = (dev_y[j] + dev_y[j - 1]) * 0.5;
    }
}