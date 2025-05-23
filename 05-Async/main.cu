#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>

static constexpr int numIterations = 100;
static constexpr int numValuesToPrint = 10;

__global__ void func1_kernel(const float* in, float* out, int numElements)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        float value = in[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value = sinf(value);
        }
        out[i] = value;
    }
}

__global__ void func2_kernel(const float* in, const float* in2, float* out, int numElements)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        float value = in[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value = -sinf(value);
        }
        out[i] = value + in2[i];
    }
}

int main(int argc, char* argv[])
{

    int numElements = (argc > 1) ? atoi(argv[1]) : 1000000;

    printf("Transforming %d values.\n", numElements);

    float* h_data1;
    float* h_data2;
    cudaMallocHost(&h_data1, numElements*sizeof(float));
    cudaMallocHost(&h_data2, numElements*sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_data1[i] = float(rand())/float(RAND_MAX + 1.0);
        h_data2[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    float* d_data1;
    float* d_data2;
    cudaMalloc(&d_data1, numElements*sizeof(float));
    cudaMalloc(&d_data2, numElements*sizeof(float));

    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t event;

    cudaEventCreate(&event);
    
    int blockSize = 256;
    int blockNum = numElements/blockSize + 1;
    
    // Timing
    clock_t start = clock();

    cudaMemcpyAsync(d_data1, h_data1, numElements*sizeof(float), cudaMemcpyHostToDevice, stream1);
    func1_kernel<<<blockNum, blockSize, 0, stream1>>>(d_data1, d_data1, numElements);
    cudaEventRecord(event, stream1);
    cudaMemcpyAsync(h_data1, d_data1, numElements*sizeof(float), cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_data2, h_data2, numElements*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaStreamWaitEvent(stream2, event);
    func2_kernel<<<blockNum, blockSize, 0, stream2>>>(d_data2, d_data1, d_data2, numElements);
    cudaMemcpyAsync(h_data2, d_data2, numElements*sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Timing
    clock_t finish = clock();

    printf("The results are:\n");
    for (int i = 0; i < numValuesToPrint; i++)
    {
        printf("%f, %f\n", h_data1[i], h_data2[i]);
    }
    printf("...\n");
    for (int i = numElements - numValuesToPrint; i < numElements; i++)
    {
        printf("%f, %f\n", h_data1[i], h_data2[i]);
    }
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < numElements; i++)
    {
        sum1 += h_data1[i];
        sum2 += h_data2[i];
    }
    printf("The summs are: %f and %f\n", sum1, sum2);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    cudaFreeHost(h_data1);
    cudaFreeHost(h_data2);

    cudaFree(d_data1);
    cudaFree(d_data2);
    
    return 0;
}