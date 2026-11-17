// Calcula quantos blocos cobrem uma dimensão da malha.
int grid_1d(int tamanho, int threads){
    return ((tamanho) + threads - 1) / threads;
}