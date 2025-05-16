#include <stdio.h>

__global__ void my_first_kernel() {
    int index_x = blockDim.x*blockIdx.x + threadIdx.x;
    int index_y = blockDim.y*blockIdx.y + threadIdx.y;
    printf("I am thread number (%d, %d)\n", index_x, index_y);
}

int main() {
    dim3 blockNum(4, 4);
    dim3 blockSize(8, 2);
    my_first_kernel<<<blockNum, blockSize>>>();
    cudaDeviceSynchronize();
}