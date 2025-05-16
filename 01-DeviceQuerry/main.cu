#include <stdio.h>

int main() {
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    printf("Num devices = %d\n", numDevices);
    for (int i = 0; i < numDevices; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("Device %i is %s and has %d multiprocessors\n", i, props.name, props.multiProcessorCount);
    }
}