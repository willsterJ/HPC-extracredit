// #define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1000

// dot product unroll
float dpunroll(long n, float *pA, float *pB){
  float R = 0.0;
  int j;
  for (j = 0; j < n; j += 4)
    R += (pA[j] * pB[j]) + (pA[j + 1] * pB[j + 1]) + (pA[j + 2] * pB[j + 2]) + (pA[j + 3] * pB[j + 3]);
  return R;
}

// transpose the matrix. Parameter variable is changed
float** transpose(float **A){
  // swap values on either side of the first diagonal
  for (int i = 1; i < N; i++){
    // stop the inner loop when b == a
    for (int j = 0; j < i; j++){
      float tmp = A[i][j];
      A[i][j] = A[j][i];
      A[j][i] = tmp;
    }
  }
  return A;
}

// matrix multiplication procedure
float **matrix_multiply(float **A, float **B_t, float **C){
  float *row, *col;
  for (int i = 0; i < N; i++){
    row = A[i];
    for (int j=0; j<N; j++){
      col = B_t[j];
      C[i][j] = dpunroll(N, row, col);
    }
  }
}

int main(int argc, char *argv[]){

  struct timespec start;
  struct timespec end;
  double time;

  // create and init matrix
  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));
  float **C = (float **)malloc(N * sizeof(float *));

  for (int i = 0; i < N; i++){
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    C[i] = (float *)malloc(N * sizeof(float));
    // alternate in a checkerboard-like sequence
    for (int j = 0; j < N; j++){
      if ((j+i) % 2 == 0)
        A[i][j] = B[i][j] = 0.2;
      else
        A[i][j] = B[i][j] = 0.6;
    }
  }

  transpose(B);
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrix_multiply(A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  time = (double)(end.tv_nsec - start.tv_nsec) * 1e-9; // convert from nanosecond to second
  double bandwidth = 3 * 4 * N * N / time * 1e-9; // 4 bytes per element in matrix. 3 NxN matrices reads
  double flops = 7 * 250 * N * N / time * 1e-9;  // 7*250 floating-point ops per element in output C matrix

  printf("time:%fsecs bandwidth:%fGB/s  flops:%fGFLOP/s\n", time, bandwidth, flops);

  free(A); free(B); free(C);
}