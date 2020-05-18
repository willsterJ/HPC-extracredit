echo "y" | module clear
gcc -std=c99 -O3 -Wall -o main.out ./main.c
./main.out
module load intel/19.0.1
icc -O3 -mkl -o mkl.out mkl_multiply.c
./mkl.out