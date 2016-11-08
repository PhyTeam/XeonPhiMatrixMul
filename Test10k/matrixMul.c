#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int Mat_size;
#define _Pos(i,j) (i * Mat_size + j)
#define REAL float
#define MIC_DEV 0
/***
* Global variable
***/
REAL *Mat_A, *Mat_B, *Mat_C;
int num_threads;
/***
* Simple matrix multiplication using OpenMP on host system
***/
void doMult(REAL* Mat_A, REAL* Mat_B, REAL* Mat_C){
  int i,j ,k;
  // Initialize matrix C
  #pragma omp parallel for private(i) shared(Mat_C)
  for (i = 0; i < Mat_size * Mat_size; i++) {
    Mat_C[i] = 0.0f;
  }
  int p = 0;
  #pragma omp parallel for default(none) private(i, j, k) \
  firstprivate(Mat_A, Mat_B, Mat_C) shared(p, Mat_size) num_threads(num_threads)
  for (i = 0; i < Mat_size; i++) {
    for (j = 0; j < Mat_size; j++) {

      for (k = 0; k < Mat_size; k++) {
        Mat_C[_Pos(i,j)] += Mat_A[_Pos(i, k)] * Mat_B[_Pos(k, j)];
      }

      #pragma omp atomic
            p += 1;
    }
    // Print percent
  #pragma omp critical
    {
      if (p % 100 == 0){
        printf("Percent: %f, %d\n", (float)p / (Mat_size * Mat_size), Mat_size);
      }
    }
  }

}

/***
* Simple matrix multiplication using offload OpenMP on MIC arch
***/
void doMult_offload(REAL* Mat_A, REAL* Mat_B, REAL* Mat_C){
  int size = Mat_size;
  int nthread = 10; // EDIT
  int i,j ,k;
  int p = 0;
// Offload part
#ifdef __INTEL_OFFLOAD
#pragma offload target(mic:MIC_DEV) \
  in(size) \
  in(Mat_A:length(size*size)) \
  in(Mat_B:length(size*size)) \
  out(Mat_C:length(size*size))
  {
/*
    #pragma omp parallel for default(none) private(i) shared(Mat_C,size)
    for(i = 0; i < size * size; ++i)
      Mat_C[i] = 0.0f;
*/
    #pragma omp parallel for default(none) private(i,j,k)\
    shared(Mat_A,Mat_B,Mat_C, Mat_size,size, p) num_threads(nthread)
    for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
        for (k = 0; k < size; k++) {
          Mat_C[_Pos(i,j)] += Mat_A[_Pos(i, k)] * Mat_B[_Pos(k, j)];
        }
        #pragma omp atomic
              p += 1;
      }
      // Print percent
    #pragma omp critical
    {
        if (p % (100) == 0){
          printf("Percent: %f, %d\n", (float)p / (size *size), size);
        }
      }
    }
  }
#else
printf("May deo co Xeon Phi ma doi offload\n");
#endif
}

/***
* Print matrix for testing
***/
void print_matrix(REAL* mat) {
  int i, j;
  for (i = 0; i < Mat_size; i++) {
      for (j = 0; j < Mat_size; j++) {
        printf("%10.3f ", mat[_Pos(i,j)]);
      }
      printf("\n");
  }
}


int main(int argc, char *argv[]){
  int num_threads = 0;
  int mat_len = 0;
  if (argc >= 3){
    num_threads = atoi(argv[1]);
    mat_len = atoi(argv[2]);
  } else {
    num_threads = omp_get_max_threads();
    mat_len = 3;
  }
  printf("Run with number of threads is :%d \n", num_threads);
  printf("Max thread : %d\n", omp_get_max_threads());
  printf("Run with matrix size is (%d, %d)\n", mat_len, mat_len);

  int num_devices = 0;
  printf("Checking for Intel(R) Xeon Phi(TM) (Target CPU) devices...\n\n");
  #ifdef __INTEL_OFFLOAD
     num_devices = _Offload_number_of_devices();
  #endif
   printf("Number of Target devices installed: %d\n\n",num_devices);
  Mat_size = mat_len;
  Mat_A = (REAL*)malloc(sizeof(REAL) * Mat_size * Mat_size);
  Mat_B = (REAL*)malloc(sizeof(REAL) * Mat_size * Mat_size);
  Mat_C = (REAL*)calloc(sizeof(REAL), Mat_size * Mat_size);
  if (Mat_A == NULL || Mat_B == NULL || Mat_C == NULL){
    printf("Error: Can not allocation memory.\n");
    return -1;
  }
  // Initialize data
  int i, j, k;
  for (i = 0; i < Mat_size; i++) {
      for (j = 0; j < Mat_size; j++) {
        Mat_A[_Pos(i,j)] = i * j;
        Mat_B[_Pos(i,j)] = i + j;
      }
  }
  // Load data from file
  // Print matrix
#ifdef PRINT_MATRIX
  printf("Matrix A: \n");
  print_matrix(Mat_A);
  printf("Matrix B: \n");
  print_matrix(Mat_B);
#endif
  // Calculate time
  double stime = omp_get_wtime();
  // Call parallel matrix multiplication
  doMult(Mat_A, Mat_B, Mat_C);
  double etime = omp_get_wtime();
  printf("Escaped time : %lf\n", etime - stime);
#ifdef PRINT_MATRIX
  printf("Matrix C: \n");
  print_matrix(Mat_C);
#endif
  return 0;
}
