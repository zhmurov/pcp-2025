#include <stdio.h>

__global__ void vec_add_kernel(float* a, float* b, float* c, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1024;
    float* h_a = (float*)calloc(N, sizeof(float));
    float* h_b = (float*)calloc(N, sizeof(float));
    float* h_c = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i*0.1;
        h_b[i] = i*(-0.1);
    }

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int blockNum = N/blockSize + 1;

    vec_add_kernel<<<blockNum, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%f + %f = %f \n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
}