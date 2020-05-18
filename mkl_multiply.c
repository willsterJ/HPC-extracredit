#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
//#include <mkl_cblas.h>
#include "mkl.h"

#define N 1000

int main(int argc, char* argv[]){
    float *A, *B, *C;
    A = (float *)mkl_malloc(N*N*sizeof(float), 32);
    B = (float *)mkl_malloc(N*N*sizeof(float), 32);
    C = (float *)mkl_malloc(N*N*sizeof(float), 32);

    int i,j = 0;
    for (i=0; i<N;i++){
        for(j=0; j<N; j++){
            if ((j+i) % 2 == 0){
                *(A + i*N + j) = 0.2;
                *(B + i*N + j) = 0.2;
            }
            else{
                *(A + i*N + j) = 0.6;
                *(B + i*N + j) = 0.6;
            }
            *(C + i*N + j) = 0;
        }
    }

    double time, bandwidth, flops, ai;

    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time = (end.tv_sec - start.tv_sec) * 1e9; // convert seconds elapsed to nanoseconds
    time = (time + (end.tv_nsec - start.tv_nsec)) * 1e-9; // take into account nanoseconds passed, and convert from nanosecond to second
    // Not exactly sure how Intel implements their MKL matrix multiply, so we assume it is the standard version but with hardware optimizations
    // For each element in output C, there are 2 * N memory accesses (row in A and col in B) and 1 mem access in element of C
    bandwidth = (2 * N + 1) * 4 * (double)(N * N) / time * 1e-9;
    // For each element in output C, we have dot product with 2 vectors. Each component in dot-product computes twice: once from multiplying element A and B and another from accumulating result (final result stored in C element)
    flops = 2 * N * (double)(N * N) / time * 1e-9;
    ai = flops / bandwidth;
    printf("MKL matrix results:\n");
    printf("time:%fsecs bandwidth:%fGB/s  flops:%fGFLOP/s  arithmetic_intensity:%fFLOP/byte\n", time, bandwidth, flops, ai);


    
}