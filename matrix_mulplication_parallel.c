/// Include
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <omp.h>
#include <stdbool.h>

/// Global declare
#ifdef __INTEL_COMPILER
////////////////////////////////////////////////////////////////////
// INTEL COMPILER
////////////////////////////////////////////////////////////////////
#define ALLOC_MATRIX(size, type) _mm_malloc(sizeof(type)*size*size,64)
#define DEALLOC_MATRIX(ptr) _mm_free(ptr)
typedef float * __attribute__((align_value(64))) REAL_PTR;
// Declare mic method
__declspec(target(mic)) void strassen_matrix_multiply_recursive(REAL_PTR A, REAL_PTR B, REAL_PTR out, int size);
__declspec(target(mic)) void _add(REAL_PTR, REAL_PTR, REAL_PTR, int);
__declspec(target(mic)) void _sub(REAL_PTR, REAL_PTR, REAL_PTR, int);
#else
////////////////////////////////////////////////////////////////////
// GCC COMPILER
////////////////////////////////////////////////////////////////////
typedef float* REAL_PTR;
#define DEALLOC_MATRIX(ptr) free(ptr)
#define ALLOC_MATRIX(size, type) malloc(sizeof(type) * size * size)
/// Declare local function
float* strassen_matrix_multiply_recursive(REAL_PTR A, REAL_PTR B, REAL_PTR out, int size);
void _add(READ_PTR, READ_PTR, REAL_PTR, int);
#endif

#define REAL float
#define CPY_32FMATRIX(a, b, size) memcpy(a, b, sizeof(float) * size * size)

/// Variable declare
//__declspec(target(mic))
REAL_PTR matrix_A;
//__declspec(target(mic))
REAL_PTR matrix_B;
//__declspec(target(mic))
REAL_PTR matrix_C;
// Offload this to devices
__declspec(target(mic)) REAL_PTR zm_a;
__declspec(target(mic)) REAL_PTR zm_b;
__declspec(target(mic)) REAL_PTR zm_c;


inline uint64_t mortonEncode_for(unsigned int x, unsigned int y){
    uint64_t answer = 0;
    //uint64_t i;
    //for (i = 0; i < (sizeof(uint64_t)* CHAR_BIT)/3; ++i) {
    //    answer |= ((x & ((uint64_t)1 << i)) << i) | ((y & ((uint64_t)1 << i)) << (i + 1));
    //}

    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;

    y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
    y = (y | (y << 2)) & 0x3333333333333333;
    y = (y | (y << 1)) & 0x5555555555555555;

   answer = y | (x << 1);

    return answer;
}

void _add(REAL_PTR ptr_A, REAL_PTR ptr_B, REAL_PTR ptr_out, int size){
    int i = 0;
#pragma omp parallel for simd
    for(i = 0; i < size * size; ++i){
    //while (i < size * size){
        //*(ptr_out + i) = (*(ptr_A + i)) + *(ptr_B + i);
        ptr_out[i] = ptr_A[i] + ptr_B[i];
        //++i;
    }

}

void _sub(REAL_PTR ptr_A, REAL_PTR ptr_B, REAL_PTR ptr_out, int size){
    int i = 0;
#pragma omp parallel for simd
    for(i = 0; i < size * size; ++i){
    // while (i < size * size){
        //*(ptr_out + i) = (*(ptr_A + i)) - *(ptr_B + i);
        ptr_out[i] = ptr_A[i] - ptr_B[i];
        //++i;
    }
}

/*
 * Function: parallel_offload_mm
 * ----------------------------
 *   Main function run in host. Returns the square matrix C <- A * B
 *
 *   zm_a : the square matrix
 *   zm_b : the square matrix
 *   zm_c : the output matrix has size the same A and B
 *   size : matrix size
 */
void parallel_offload_mm(REAL_PTR zm_a, REAL_PTR zm_b, REAL_PTR zm_c, int size)
{
// Execuatable code on devices
#pragma offload target(mic:0) \
  in(zm_a:length(size * size)), \
  in(zm_b:length(size * size)), \
  in(size),\
  out(zm_c:length(size * size))
  {
  #pragma omp parallel
  #pragma omp single nowait
    strassen_matrix_multiply_recursive(zm_a, zm_b, zm_c, size);
  }
}

/*
 * Function: strassen_matrix_multiply_recursive
 * ----------------------------
 *   Returns the square matrix C <- A * B
 *
 *   A : the square matrix
 *   B : the square matrix
 *   out : the output matrix has size the same A and B
 *   size : matrix size
 */
void
strassen_matrix_multiply_recursive(REAL_PTR A, REAL_PTR B, REAL_PTR out, int size)
{
    // allocate memory contain result
    //REAL_PTR ptr_C = ALLOC_MATRIX(size, float);
    //printf("Called strassen: %d \n", size);
    if (size <= MIN_SQUARE_MATRIX) {
        int i, j, k;
        float sum;
        for (i = 0; i < size; ++i){
            for(j = 0; j < size; ++j){
                sum = 0;
                for(k = 0; k < size; ++k){
                    sum = sum + A[i * size + k] * B[k * size + j];
                }
                out[i * size + j] = sum;
            }
        }
        return NULL;
      }
      int i, j;
      REAL_PTR ptr_T = ALLOC_MATRIX(size, float);
      int half_size = size / 2; // half_size <- size / 2
      //printf("Half size : %d\n", half_size);
      REAL_PTR A11;
      REAL_PTR A12; REAL_PTR A21; REAL_PTR A22; REAL_PTR B11; REAL_PTR B12;
      REAL_PTR B21; REAL_PTR B22;

      const int submatrix_el = half_size * half_size;
      float *ptr_C = out;
      float* C11 = ptr_C;
      float* C12 = ptr_C + submatrix_el;
      float* C21 = ptr_C + submatrix_el * 2;
      float* C22 = ptr_C + submatrix_el * 3;

      float* T11 = ptr_T;
      float* T12 = ptr_T + submatrix_el;
      float* T21 = ptr_T + submatrix_el * 2;
      float* T22 = ptr_T + submatrix_el * 3;

      A11 = A;
      A12 = A + submatrix_el;
      A21 = A + submatrix_el * 2;
      A22 = A + submatrix_el * 3;

      B11 = B;
      B12 = B + submatrix_el;
      B21 = B + submatrix_el * 2;
      B22 = B + submatrix_el * 3;

      // Generate new variable
      REAL_PTR _T[7];
      REAL_PTR _S[7];
      REAL_PTR _M[7];
      for(i = 0; i < 7; ++i) {
          _T[i] = ALLOC_MATRIX(half_size, float);
          _S[i] = ALLOC_MATRIX(half_size, float);
          _M[i] = ALLOC_MATRIX(half_size, float);
          if (_T[i] == NULL || _M[i] == NULL || _S[i] == NULL){
              printf("Error!\n");
          }
      }

      _add(A11, A22, _S[0], half_size);
      _add(A21, A22, _S[1], half_size);
      CPY_32FMATRIX(_S[2], A11, half_size);
      CPY_32FMATRIX(_S[3], A22, half_size);
     _add(A11, A12, _S[4], half_size);
      _sub(A21, A11, _S[5], half_size);
      _sub(A12, A22, _S[6], half_size);

      _add(B11, B22, _T[0], half_size);
      CPY_32FMATRIX(_T[1], B11, half_size);
      _sub(B12, B22, _T[2], half_size);
      _sub(B21, B11, _T[3], half_size);
      CPY_32FMATRIX(_T[4], B22, half_size);
      _add(B11, B12, _T[5], half_size);
      _add(B21, B22, _T[6], half_size);

      for(i = 0; i < 7; ++i) {
#pragma omp task final(size < 512)
                strassen_matrix_multiply_recursive(_S[i], _T[i], _M[i], half_size);
      }
#pragma omp taskwait

      _add(_M[0], _M[3], C11, half_size);
      _sub(C11, _M[4], C11, half_size);
      _add(C11, _M[6], C11, half_size);
      _add(_M[2], _M[4], C12, half_size);
      _add(_M[1], _M[3], C21, half_size);
      _sub(_M[0], _M[1], C22, half_size);
      _add(C22, _M[2], C22, half_size);
      _add(C22, _M[5], C22, half_size);
      // Free memory
      for(i = 0; i < 7; ++i) {
          DEALLOC_MATRIX(_S[i]);
          DEALLOC_MATRIX(_T[i]);
          DEALLOC_MATRIX(_M[i]);
      }
      DEALLOC_MATRIX(ptr_T);
}

void generate_square_matrix(int size){
  int i, j;
  matrix_A = ALLOC_MATRIX(size, float);
  matrix_B = ALLOC_MATRIX(size, float);
  matrix_C = ALLOC_MATRIX(size, float);


  zm_a = ALLOC_MATRIX(size, float);
  zm_c = ALLOC_MATRIX(size, float);
  zm_b = ALLOC_MATRIX(size, float);

  srand(time(0));
  for(i = 0; i < size; ++i){
      for(j = 0; j < size; ++j){
          uint64_t zidx = mortonEncode_for(i, j);
          int idx = i * size + j;
          matrix_A[idx] = rand() % 10;
          matrix_B[idx] = rand() % 10;

          zm_a[zidx] = matrix_A[idx];
          zm_b[zidx] = matrix_B[idx];
      }
  }
  int k;
  if (MIN_SQUARE_MATRIX > size) MIN_SQUARE_MATRIX=size;
  REAL_PTR t = ALLOC_MATRIX(MIN_SQUARE_MATRIX * MIN_SQUARE_MATRIX, float);
  for(k = 0; k < (size / MIN_SQUARE_MATRIX) * (size / MIN_SQUARE_MATRIX); ++k){
      int num_of_el = MIN_SQUARE_MATRIX *MIN_SQUARE_MATRIX;
      REAL_PTR ptr = zm_a + num_of_el * k;
      REAL_PTR ptr2 = zm_b + num_of_el * k;
      for(i = 0; i < MIN_SQUARE_MATRIX; ++i){
          for(j = 0; j < MIN_SQUARE_MATRIX; ++j)
          {
              int idx = mortonEncode_for(i, j);
              t[i * MIN_SQUARE_MATRIX + j] = ptr[idx];
          }
      }
      memcpy(ptr, t, sizeof(REAL) * num_of_el);
      for(i = 0; i < MIN_SQUARE_MATRIX; ++i){
          for(j = 0; j < MIN_SQUARE_MATRIX; ++j)
          {
              int idx = mortonEncode_for(i, j);
              t[i * MIN_SQUARE_MATRIX + j] = ptr2[idx];
          }
      }
      memcpy(ptr2, t, sizeof(REAL) * num_of_el);
      //printf("Has reformated.\n");
  }
  DEALLOC_MATRIX(t);
}

int main(int argc, char *argv[])
{
  int size;
  if(argc > 1){
      size = atoi(argv[1]);
  } else {
      size = 32; // Just for test
  }

  // Generate matrix
  generate_square_matrix(size);


  return 0;
}
