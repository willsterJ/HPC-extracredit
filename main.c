// #define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1000

float dpunroll(long n, float *pA, float *pB) {
  float R = 0.0;
  int j;
  for (j=0;j<n;j+=4)
    R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
  return R;
}

void print_array(float* A){
  printf("Printing array... \n");
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      printf(".2f", A[i][j]);
    }
  }
}

int main(int argc, char *argv[]){
    float A[N][N];
    float B[N][N];

    for (int i=0; i<N; i++){
      for (int j=0; j<N; j++){
        if (j % 2 == 0)
          A[i][j] = B[i][j] = 0.2;
        else
          A[i][j] = B[i][j] = 0.6;
      }
    }
}