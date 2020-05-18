# Instructions

This is the extra credit assignment for HPC. The program is run on crackle1 server in Access CIMS NYU. The CPU of choice is the Intel-Xeon E5630 with the following specs:
- DRAM bandwidth = 25.6GB/s
- CPU peak FLOPS = 81.3GFLOP/s
- L1-cache = 128KB
- L2-cache = 1024KB

Log into CIMS:
```
ssh <username>@access.cims.nyu.edu
ssh crackle1@cims.nyu.edu
```

#### The steps below can be obviated by just running script.sh

To compile and run dpunroll and tiling matrix multiplication:
```
gcc -std=c99 -O3 -Wall -o main.out ./main.c
./main.out
```
The Block dimension can be changed in **#define BLOCK_DIM** in **main.c**

To compile and run dpunroll and tiling matrix multiplication:
```
module load intel/19.0.1
icc -O3 -mkl -o mkl.out mkl_multiply.c
./mkl.out
```