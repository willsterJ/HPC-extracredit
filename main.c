#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 1000
#define BLOCK_DIM 250

/*
Specs for Intel Xeon E5630:
L1-cache: 128KB, so submatrix_A = 10x10
L2-cache: 1024KB, so submatrix_A = 250x250
*/

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
float** matrix_multiply(float **A, float **B_t, float **C){
  float *row, *col;
  for (int i = 0; i < N; i++){
    row = A[i];
    for (int j=0; j<N; j++){
      col = B_t[j];
      C[i][j] = dpunroll(N, row, col);
    }
  }
  return C;
}

// naive method where submatrices are not copied. DO NOT USE
float** matrix_multiply_tiling_naive(float **A, float **B_t, float **C){
  int numBlocks = N / BLOCK_DIM;  // number of blocks in the width direction
  int block_idx = 0;
  int stride = BLOCK_DIM;
  float R = 0;  // result stored by dot product

  int row_ind = 0; 
  int col_ind = 0;
  for (int i=row_ind; i<stride && i<N; i++){
    for (int j=col_ind; j<stride && j<N; j++){
      R = 0;
      for (int k=col_ind; k<stride && k<N; k++){  // dot product vect in row_vect_A with col_vect_B within block
        R += A[i][k] * B_t[j][k];
      }
      C[i][j] += R;  // increment output element by partial dot vector result
    }
    block_idx++;  // go to next block

    if (block_idx % numBlocks != 0) // go to next column block
      col_ind += stride;
    else{ // switch to next row set of column blocks
      col_ind = 0;
      row_ind += stride;
    }
  }
  return C;
}

// matrix multiply using tiling
float** matrix_multiply_tiling(float **A, float **B_t, float **C){
  int numBlocks = N / BLOCK_DIM;  // number of blocks in the width direction
  int stride = BLOCK_DIM;

  // create submatrices
  float A_sub[BLOCK_DIM][BLOCK_DIM];
  float B_sub[BLOCK_DIM][BLOCK_DIM];

  int row_ind_start, col_ind_start;
  for (int block=0; block<pow(numBlocks, 2); block++){  // loop through blocks in matrix C
    // get starting indices according to block id
    row_ind_start = (block / numBlocks) * stride;
    col_ind_start = (block % numBlocks) * stride;
    // copy into submatrices
    for (int i=0; i<stride; i++){
      for (int j=0; j<stride; j++){
        A_sub[i][j] = A[i+row_ind_start][j+col_ind_start];
        B_sub[i][j] = B_t[i+row_ind_start][j+col_ind_start];
      }
    }
    // dot product
    for (int i=0; i<stride; i++){
      for (int j=0; j<stride; j++){
        for (int k=0; k<stride; k++){
          C[i+row_ind_start][j+col_ind_start] += A_sub[i][k] * B_sub[j][k];
        }
      }
    }
  }
  return C;
}

int main(int argc, char *argv[]){

  // create and init matrix
  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));
  float **C = (float **)malloc(N * sizeof(float *));

  for (int i = 0; i < N; i++){
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    C[i] = (float *)malloc(N * sizeof(float));
    for (int j = 0; j < N; j++){
      // alternate in a checkerboard-like sequence
      if ((j+i) % 2 == 0)
        A[i][j] = B[i][j] = 0.2;
      else
        A[i][j] = B[i][j] = 0.6;
      C[i][j] = 0;
    }
  }

  // transpose the matrix so that row vector in A and col vector in B are parallel
  transpose(B);

  double time, bandwidth, flops, ai;

  struct timespec start;
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrix_multiply(A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  time = (end.tv_sec - start.tv_sec) * 1e9; // convert seconds elapsed to nanoseconds
  time = (time + (end.tv_nsec - start.tv_nsec)) * 1e-9; // take into account nanoseconds passed, and convert from nanosecond to second
  // 8 memory references (4 bytes each) looped 250 times per dpunroll call
  // 1 memory reference per output C element access for entire NxN matrix
  bandwidth = 8 * 4 * 250 * 1 * (double)(N * N) / time * 1e-9;
  // 8 FLOP looped 250 times per dpunroll call
  // dpunroll is called N*N times
  flops = 8 * 250 * (double)(N * N) / time * 1e-9;  // 8 flops 
  // 7 FLOP and 8 memory accesses looped 250 times
  //ai = (7 * 250) / (double)(8 * 4 * 250);
  ai = flops / bandwidth;
  printf("dpunroll matrix specs:\n");
  printf("time:%fsecs bandwidth:%fGB/s  flops:%fGFLOP/s  arithmetic_intensity:%fFLOP/byte\n", time, bandwidth, flops, ai);


  // reset matrix C to 0
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      C[i][j] = 0;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  matrix_multiply_tiling(A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  time = (end.tv_sec - start.tv_sec) * 1e9; // convert seconds elapsed to nanoseconds
  time = (time + (end.tv_nsec - start.tv_nsec)) * 1e-9; // take into account nanoseconds passed, and convert from nanosecond to second
  // BLOCK_DIM x BLOCK_DIM x 2 memory accesses when copying from matrix portion to submatrix
  // BLOCK_DIM x BLOCK_DIM x BLOCK_DIM x 1 memory accesses for retriving and updating elements in C
  // There are total of (N / BLOCK_DIM)^2 blocks for matrix C (i.e. the entire parent loop)
  bandwidth = ((double)BLOCK_DIM*BLOCK_DIM*2 + BLOCK_DIM*BLOCK_DIM*BLOCK_DIM*1) * pow((N/BLOCK_DIM), 2) / time * 1e-9;
  // BLOCK_DIM x BLOCK_DIM elements in block in C. Each element in block computes partial dot product of size BLOCK_DIM
  // 3 flops per element-wise dot-product of which there are BLOCK_DIM
  // There are total of (N / BLOCK_DIM)^2 blocks for matrix C (i.e. the entire parent loop)
  flops = ((double)BLOCK_DIM*BLOCK_DIM * 2*BLOCK_DIM + 4) * pow((N/BLOCK_DIM), 2) / time * 1e-9;
  // 3 FLOP and 3 memory accesses
  //ai = 3 / (double)(4 * 3);
  ai = flops/bandwidth;
  printf("tiling matrix specs:\n");
  printf("time:%fsecs bandwidth:%fGB/s  flops:%fGFLOP/s  arithmetic_intensity:%fFLOP/byte\n", time, bandwidth, flops, ai);

  // reset matrix C to 0
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      C[i][j] = 0;
    }
  }


  free(A); free(B); free(C);

}