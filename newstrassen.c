
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
uint64_t mortonEncode_for(unsigned int x, unsigned int y);
/// Global variable

/// Global declare
#ifdef __INTEL_COMPILER
#define ALLOC_MATRIX(size, type) _mm_malloc(sizeof(type)*size*size,32)
#define DEALLOC_MATRIX(ptr) _mm_free(ptr)
typedef float * __attribute__((align_value(32))) REAL_PTR;
#else
typedef float* REAL_PTR;
#define DEALLOC_MATRIX(ptr) free(ptr)
#define ALLOC_MATRIX(size, type) malloc(sizeof(type) * size * size)
#endif
#define REAL float

#define GET_MATRIX_BY_IDX(name, idx, idy) name##idx##idy
#define CPY_32FMATRIX(a, b, size) memcpy(a, b, sizeof(float) * size * size)

void serial_matrix_mul(REAL_PTR matrix_A,
                       REAL_PTR matrix_B,
                       REAL_PTR matrix_C, int size)
{
    int i, j, k;
    float sum;
    for (i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            sum = 0;
            for(k = 0; k < size; ++k){
                sum = sum + matrix_A[i * size + k] * matrix_B[k * size + j];
            }
            matrix_C[i * size + j] = sum;
        }
    }
}


void simple_matrix_mul(REAL_PTR matrix_A,
                       REAL_PTR matrix_B,
                       REAL_PTR matrix_C, int size)
{
    int i, j, k;
    float sum;
#pragma omp parallel for private(i, j, k, sum) firstprivate(matrix_A, matrix_B, size, matrix_C)
    for (i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            sum = 0;
            for(k = 0; k < size; ++k){
                sum = sum + matrix_A[i * size + k] * matrix_B[k * size + j];
            }
            matrix_C[i * size + j] = sum;
        }
    }
}

void simple_matrix_mul_2(const float* matrix_A, const float* matrix_B,
                         float* matrix_C, const int size)
{

}

void simple_mm_using_zmorton(const float* matrix_A,
                             const float* matrix_B,
                             float* matrix_C, const int size){
    unsigned int i, j, k;
    unsigned int sz = size;
    float sum;

    for (i = 0; i < sz; ++i){
        for(j = 0; j < sz; ++j){
            sum = 0;
            uint64_t idx_c = mortonEncode_for(i,j);
            for(k = 0; k < sz; ++k){
                // Calculate index
                uint64_t idx_a = mortonEncode_for(i, k);
                uint64_t idx_b = mortonEncode_for(k, j);
                sum = sum + matrix_A[idx_a] * matrix_B[idx_b];
            }
            matrix_C[idx_c] = sum;
        }
    }
}

/***
 * Show matrix to terminal
 ***/
void show_matrix(const REAL *matrix, int size, bool isMortonLayout){
    if (size >= 64) {
        return;
    }
    int i, j, idx;
    for (i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            if (isMortonLayout){
                idx = mortonEncode_for(i, j);
            } else
                idx = i * size + j;
            printf("%f ", matrix[idx]);
        }
        printf("\n");
    }
}

/***
 * Write a matrix to a file
 * filename: output file name
 * return: none
 ***/
void write_file_binary(REAL* data, int size, const char* filename)
{
    FILE * pFile;
    pFile = fopen (filename, "wb");
    fwrite (data , sizeof(float), sizeof(float) * size * size, pFile);
    fclose (pFile);
}


/***
 * Write a matrix to a file
 * filename: output file name
 * return: none
 ***/
void write_file(REAL* data, int size, const char* filename)
{
    FILE * pFile;
    pFile = fopen (filename, "w");
    int i, j;
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            fprintf(pFile, "%.2f ", data[i * size + j]);
        }
        fprintf(pFile, "\n");
    }
    fclose (pFile);
}

void write_file_zmorton(REAL* data, int size, const char* filename){
    FILE * pFile;
    pFile = fopen (filename, "w");
    int i, j;
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            uint64_t idx = mortonEncode_for(i, j);
            fprintf(pFile, "%.2f ", data[idx]);
        }
        fprintf(pFile, "\n");
    }
    fclose (pFile);
}

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

inline void add(float* ptr_A, float *ptr_B, int size){
    int i = 0;
    while (i < size * size){
        *(ptr_A + i) = (*(ptr_A + i)) + *(ptr_B + i);
        ++i;
    }
}

void _add(REAL_PTR ptr_A, REAL_PTR ptr_B, REAL_PTR ptr_out, int size){
    int i = 0;

    while (i < size * size){
        *(ptr_out + i) = (*(ptr_A + i)) + *(ptr_B + i);
        ++i;
    }
}

void _sub(REAL_PTR ptr_A, REAL_PTR ptr_B, REAL_PTR ptr_out, int size){
    int i = 0;

    while (i < size * size){
        *(ptr_out + i) = (*(ptr_A + i)) - *(ptr_B + i);
        ++i;
    }
}
#define MIN_SQUARE_MATRIX 32
#define USING_ZMORTON
int type = 0;

float* strassen_matrix_multiply_recursive(REAL_PTR A, REAL_PTR B, REAL_PTR out, int size){
    // allocate memory contain result
    //REAL_PTR ptr_C = ALLOC_MATRIX(size, float);

    if (size <= MIN_SQUARE_MATRIX) {

        int i, j, k;
        float sum;
//#pragma omp parallel for private(i, j, k, sum) firstprivate(A, B, out, size)
        for (i = 0; i < size; ++i){
            for(j = 0; j < size; ++j){
                sum = 0;
                for(k = 0; k < size; ++k){
                    sum = sum + A[i * size + k] * B[k * size + j];
                }
                out[i * size + j] = sum;
            }
        }
        //return;
    } else {
        int i, j;
        REAL_PTR ptr_T = ALLOC_MATRIX(size, float);
        int half_size = size / 2; // half_size <- size / 2
        //printf("Half size : %d\n", half_size);
        REAL_PTR A11;
        REAL_PTR A12; REAL_PTR A21; REAL_PTR A22; REAL_PTR B11; REAL_PTR B12;
        REAL_PTR B21; REAL_PTR B22;

#ifdef USING_ZMORTON
        int submatrix_el = half_size * half_size;
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
#else
        REAL_PTR C11 = ALLOC_MATRIX(half_size, float);
        REAL_PTR C12 = ALLOC_MATRIX(half_size, float);
        REAL_PTR C21 = ALLOC_MATRIX(half_size, float);
        REAL_PTR C22 = ALLOC_MATRIX(half_size, float);

        REAL_PTR T11 = ALLOC_MATRIX(half_size, float);
        REAL_PTR T12 = ALLOC_MATRIX(half_size, float);
        REAL_PTR T21 = ALLOC_MATRIX(half_size, float);
        REAL_PTR T22 = ALLOC_MATRIX(half_size, float);


        A11 = ALLOC_MATRIX(half_size, float);
        A12 = ALLOC_MATRIX(half_size, float);
        A21 = ALLOC_MATRIX(half_size, float);
        A22 = ALLOC_MATRIX(half_size, float);

        B11 = ALLOC_MATRIX(half_size, float);
        B12 = ALLOC_MATRIX(half_size, float);
        B21 = ALLOC_MATRIX(half_size, float);
        B22 = ALLOC_MATRIX(half_size, float);

        for(i = 0; i < half_size;  i++){
            memcpy(A11 + half_size * i, (A + size * i), sizeof(float) * half_size);
            memcpy(A12 + half_size * i, (A + size * i + half_size), sizeof(float) * half_size);
            memcpy(A21 + half_size * i, (A + size * (half_size + i)), sizeof(float) * half_size);
            memcpy(A22 + half_size * i, (A + size * (half_size + i) + half_size), sizeof(float) * half_size);

            memcpy(B11 + half_size * i, (B + size * i), sizeof(float) * half_size);
            memcpy(B12 + half_size * i, (B + size * i + half_size), sizeof(float) * half_size);
            memcpy(B21 + half_size * i, (B + size * (half_size + i)), sizeof(float) * half_size);
            memcpy(B22 + half_size * i, (B + size * (half_size + i) + half_size), sizeof(float) * half_size);
        }
#endif
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
        //#pragma omp taski
        _add(A11, A22, _S[0], half_size);
        //#pragma omp task
        _add(A21, A22, _S[1], half_size);
        //#pragma omp task
        CPY_32FMATRIX(_S[2], A11, half_size);
        //#pragma omp task
        CPY_32FMATRIX(_S[3], A22, half_size);
        //#pragma omp task
       _add(A11, A12, _S[4], half_size);
       //#pragma omp task
        _sub(A21, A11, _S[5], half_size);
        //#pragma omp task
        _sub(A12, A22, _S[6], half_size);


//#pragma omp taskwait

        _add(B11, B22, _T[0], half_size);
        CPY_32FMATRIX(_T[1], B11, half_size);
        _sub(B12, B22, _T[2], half_size);
        _sub(B21, B11, _T[3], half_size);
        CPY_32FMATRIX(_T[4], B22, half_size);
        _add(B11, B12, _T[5], half_size);
        _add(B21, B22, _T[6], half_size);

//#pragma omp parallel for private(i) firstprivate(_S, _T, _M, half_size) schedule(static)

            for(i = 0; i < 7; ++i) {
#pragma omp task final(size < 256)
                  strassen_matrix_multiply_recursive(_S[i], _T[i], _M[i], half_size);
            }
#pragma omp taskwait

//#pragma omp task
        _add(_M[0], _M[3], C11, half_size);
        _sub(C11, _M[4], C11, half_size);
        _add(C11, _M[6], C11, half_size);
//#pragma omp task
        _add(_M[2], _M[4], C12, half_size);
//#pragma omp task
        _add(_M[1], _M[3], C21, half_size);
//#pragma omp task
        _sub(_M[0], _M[1], C22, half_size);
        _add(C22, _M[2], C22, half_size);
        _add(C22, _M[5], C22, half_size);
//#pragma omp taskwait
        for(i = 0; i < 7; ++i) {
            DEALLOC_MATRIX(_S[i]);
            DEALLOC_MATRIX(_T[i]);
            DEALLOC_MATRIX(_M[i]);
        }
#ifdef USING_ZMORTON
        DEALLOC_MATRIX(ptr_T);
#else

        for(i = 0; i < half_size;  i++){
            for(j = 0; j < half_size; j++){
               out[i * size + j] = C11[i * half_size + j];
               out[i * size + j + half_size] = C12[i * half_size + j];
               out[(i + half_size)*size + j] = C21[i * half_size + j];
               out[(i + half_size)*size + j + half_size] = C22[i*half_size + j];
            }
        }
        //printf("\n++++++++++++++++Matrix C++++++++++++++++++++++++\n");
        //show_matrix(ptr_C, size);
        //printf("\n");

        DEALLOC_MATRIX(T11);
        DEALLOC_MATRIX(T12);
        DEALLOC_MATRIX(T21);
        DEALLOC_MATRIX(T22);


        DEALLOC_MATRIX(C11);
        DEALLOC_MATRIX(C12);
        DEALLOC_MATRIX(C21);
        DEALLOC_MATRIX(C22);

        DEALLOC_MATRIX(A11);
        DEALLOC_MATRIX(A12);
        DEALLOC_MATRIX(A21);
        DEALLOC_MATRIX(A22);

        DEALLOC_MATRIX(B11);
        DEALLOC_MATRIX(B12);
        DEALLOC_MATRIX(B21);
        DEALLOC_MATRIX(B22);
#endif
    }
    return NULL;
}


float* square_matrix_multiply_recursive(const float* A, const float* B, int size){
    // allocate memory contain result
    float *ptr_C = ALLOC_MATRIX(size, float);

    if (size <= 1) {
        *ptr_C = (*A) * (*B);
    } else {
        int i, j;
        float *ptr_T = ALLOC_MATRIX(size, float);
        int half_size = size / 2; // half_size <- size / 2
        //printf("Half size : %d\n", half_size);
        float* A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22;

#ifdef USE_ZMORTON
        float* C11 = ptr_C;
        float* C12 = ptr_C + half_size;
        float* C21 = ptr_C + half_size * 2;
        float* C22 = ptr_C + half_size * 3;

        float* T11 = ptr_T;
        float* T12 = ptr_T + half_size;
        float* T21 = ptr_T + half_size * 2;
        float* T22 = ptr_T + half_size * 3;

        A11 = A;
        A12 = A + half_size;
        A21 = A + half_size * 2;
        A22 = A + half_size * 3;

        B11 = B;
        B12 = B + half_size;
        B21 = B + half_size * 2;
        B22 = B + half_size * 3;
#else
        float* C11 = ALLOC_MATRIX(half_size, float);
        float* C12 = ALLOC_MATRIX(half_size, float);
        float* C21 = ALLOC_MATRIX(half_size, float);
        float* C22 = ALLOC_MATRIX(half_size, float);

        float* T11 = ALLOC_MATRIX(half_size, float);
        float* T12 = ALLOC_MATRIX(half_size, float);
        float* T21 = ALLOC_MATRIX(half_size, float);
        float* T22 = ALLOC_MATRIX(half_size, float);


        A11 = ALLOC_MATRIX(half_size, float);
        A12 = ALLOC_MATRIX(half_size, float);
        A21 = ALLOC_MATRIX(half_size, float);
        A22 = ALLOC_MATRIX(half_size, float);

        B11 = ALLOC_MATRIX(half_size, float);
        B12 = ALLOC_MATRIX(half_size, float);
        B21 = ALLOC_MATRIX(half_size, float);
        B22 = ALLOC_MATRIX(half_size, float);

        for(i = 0; i < half_size;  i++){
            memcpy(A11 + half_size * i, (float*)(A + size * i), sizeof(float) * half_size);
            memcpy(A12 + half_size * i, (float*)(A + size * i + half_size), sizeof(float) * half_size);
            memcpy(A21 + half_size * i, (float*)(A + size * (half_size + i)), sizeof(float) * half_size);
            memcpy(A22 + half_size * i, (float*)(A + size * (half_size + i) + half_size), sizeof(float) * half_size);

            memcpy(B11 + half_size * i, (float*)(B + size * i), sizeof(float) * half_size);
            memcpy(B12 + half_size * i, (float*)(B + size * i + half_size), sizeof(float) * half_size);
            memcpy(B21 + half_size * i, (float*)(B + size * (half_size + i)), sizeof(float) * half_size);
            memcpy(B22 + half_size * i, (float*)(B + size * (half_size + i) + half_size), sizeof(float) * half_size);
        }
#endif

        //printf("\nmatrix A12\n");
        //show_matrix(B22, half_size);

        //printf("\nmatrix B11\n");
        //show_matrix(B22, half_size);

        C11 = square_matrix_multiply_recursive(A11, B11, half_size);
        T11 = square_matrix_multiply_recursive(A12, B21, half_size);

        C12 = square_matrix_multiply_recursive(A11, B12, half_size);
        T12 = square_matrix_multiply_recursive(A12, B22, half_size);

        C21 = square_matrix_multiply_recursive(A21, B11, half_size);
        T21 = square_matrix_multiply_recursive(A22, B21, half_size);

        C22 = square_matrix_multiply_recursive(A21, B12, half_size);
        T22 = square_matrix_multiply_recursive(A22, B22, half_size);


        add(C11, T11, half_size);
        add(C12, T12, half_size);
        add(C21, T21, half_size);
        add(C22, T22, half_size);



#ifdef USE_ZMORTON
        DEALLOC_MATRIX(ptr_T);
#else

        for(i = 0; i < half_size;  i++){
            for(j = 0; j < half_size; j++){
               ptr_C[i * size + j] = C11[i * half_size + j];
               ptr_C[i * size + j + half_size] = C12[i * half_size + j];
               ptr_C[(i + half_size)*size + j] = C21[i * half_size + j];
               ptr_C[(i + half_size)*size + j + half_size] = C22[i*half_size + j];
            }
        }
        //printf("\n++++++++++++++++Matrix C++++++++++++++++++++++++\n");
        //show_matrix(ptr_C, size);
        //printf("\n");

        DEALLOC_MATRIX(T11);
        DEALLOC_MATRIX(T12);
        DEALLOC_MATRIX(T21);
        DEALLOC_MATRIX(T22);


        DEALLOC_MATRIX(C11);
        DEALLOC_MATRIX(C12);
        DEALLOC_MATRIX(C21);
        DEALLOC_MATRIX(C22);

        DEALLOC_MATRIX(A11);
        DEALLOC_MATRIX(A12);
        DEALLOC_MATRIX(A21);
        DEALLOC_MATRIX(A22);

        DEALLOC_MATRIX(B11);
        DEALLOC_MATRIX(B12);
        DEALLOC_MATRIX(B21);
        DEALLOC_MATRIX(B22);
#endif
    }
    return ptr_C;
}

void generate_square_matrix(const char* fa, const char* fb, const char* fc, int size){
    int i, j, k;
    REAL *zmatrix_A, *zmatrix_B, *zmatrix_C;
    zmatrix_A = ALLOC_MATRIX(size, REAL);
    zmatrix_B = ALLOC_MATRIX(size, REAL);
    zmatrix_C = ALLOC_MATRIX(size, REAL);

    // Generate matrix mortonEncode
    for(i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            uint64_t idx = mortonEncode_for(i, j);
            //int idx = i * size + j;
            zmatrix_A[idx] = rand() % 100; // Random number has range from 0 --> 99
            zmatrix_B[idx] = rand() % 100;
        }
    }
    printf("Beginning genarate file...\n");
    write_file_binary(zmatrix_A, size, fa);
    printf("Generate file for matrix A completed.\n");
    write_file_binary(zmatrix_B, size, fb);
    printf("Generate file for matrix B completed.\n");
    write_file_binary(zmatrix_C, size, fc);
    printf("Generate file for matrix C completed.\n");


    DEALLOC_MATRIX(zmatrix_A); DEALLOC_MATRIX(zmatrix_B); DEALLOC_MATRIX(zmatrix_C);
}

void load_square_matrix(const char* filename, REAL *matrix, int size, int index)
{
    FILE * pFile;
    pFile = fopen (filename, "r");
    long int s = size * size * sizeof(REAL) * index;
    fseek(pFile, s, SEEK_SET);
    fread(matrix, sizeof(REAL), size * size, pFile);
    fclose (pFile);
}

/// Unit test
void test_generate_and_load_matrix(int size){
    // Generate file
    generate_square_matrix("zma.bin", "zmb.bin", "zmc.bin", size);
    REAL* zmatrix;
    zmatrix = ALLOC_MATRIX(size, REAL);

    load_square_matrix("zma.bin", zmatrix, size, 0);
    printf("______matrix A : full ______\n");
    show_matrix(zmatrix, size, true);
     DEALLOC_MATRIX(zmatrix);
    int i;
    for(i = 0; i < 4; ++i) {

        zmatrix = ALLOC_MATRIX(size / 2, REAL);
        load_square_matrix("zma.bin", zmatrix, size / 2, i);
        printf("_____matrix %d : full ______\n", i);
        show_matrix(zmatrix, size / 2, true);
        DEALLOC_MATRIX(zmatrix);
    }
}

void test_simple_mm_using_zmorton(int size){
    int i, j, k;
    float *zmatrix_A, *zmatrix_B, *zmatrix_C;

    zmatrix_A = ALLOC_MATRIX(size, float);
    zmatrix_B = ALLOC_MATRIX(size, float);
    zmatrix_C = ALLOC_MATRIX(size, float);

    // Generate matrix mortonEncode
    for(i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            uint64_t idx = mortonEncode_for(i, j);
            //int idx = i * size + j;
            zmatrix_A[idx] = rand() % 10;
            zmatrix_B[idx] = rand() % 10;
        }
    }

    printf("\nZ-matrixA\n");
    show_matrix(zmatrix_A, size, true);
    printf("____________________________________________");
    printf("\nZ-matrixB\n");
    show_matrix(zmatrix_B, size, true);
    printf("____________________________________________");

    simple_mm_using_zmorton(zmatrix_A, zmatrix_B, zmatrix_C, size);
    printf("\nZ-matrix RESULT \n");
    show_matrix(zmatrix_C, size, true);
    printf("____________________________________________");
    DEALLOC_MATRIX(zmatrix_A); DEALLOC_MATRIX(zmatrix_B); DEALLOC_MATRIX(zmatrix_C);
}

float cmatrix_a[1000][1000];
float cmatrix_b[1000][1000];
float cmatrix_c[1000][1000];

void simple_matrix_mul_v3(){
    int i, j, k;
    float sum;
    int size = 1000;
#pragma omp parallel for private(i,j,k, sum) shared(cmatrix_a, cmatrix_b, cmatrix_c, size)
    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            sum = 0;
            for(k = 0; k < size; ++k){
                sum += cmatrix_a[i][k] * cmatrix_b[k][j];
            }
            cmatrix_c[i][j] = sum;
        }
    }
}


int main(int argc, char *argv[])
{
    printf("Hello World!\n");
    unsigned int i, j;
    //for(i = 0; i < 8; i++)
    //    for(j = 0; j < 8; j++)
    //   printf("(%u, %u) = (%"PRIu64")\n", i, j, mortonEncode_for(i, j));
    int size;
    if(argc > 1){
        size = atoi(argv[1]);
    } else size = 512;
    bool enable_serial = false;
    if (argc > 2){
        enable_serial = true;
    }

#if defined(_OPENMP)
    int numthreads = 1;
    if (argc > 2){
        numthreads = atoi(argv[2]);
    }
    //omp_set_num_threads(numthreads);

    int pid, nthreads;
#pragma omp parallel private(pid, nthreads)
    {
        pid = omp_get_thread_num();
        if(pid == 0){
            nthreads = omp_get_max_threads();
            printf("Max threads : %d\n", nthreads);
        }
    }

#endif
    float *matrix_A, *matrix_B, *matrix_C;
    float *zm_a, *zm_b, *zm_c;
    //posix_memalign((void**)&matrix_A, 64, sizeof(float) * size * size); // Lam mau khong loi ich gi
    //posix_memalign((void**)&matrix_B, 64, sizeof(float) * size * size); // Lam mau khong loi ich gi
    //posix_memalign((void**)&matrix_C, 64, sizeof(float) * size * size); // Lam mau khong loi ich gi
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
    // Test simple z morton
    //test_simple_mm_using_zmorton(size);
    // Test generate and load zmatrix
    //test_generate_and_load_matrix(size);

    printf("\nMatrix A\n");
    //show_matrix(matrix_A, size);
    printf("\nMatrix B\n");
    //show_matrix(matrix_B, size);
    printf("Begin multiply....\n");

    //show_matrix(C, size);
    double strassen_time = 0.0;
    double stupid_time = 0.0;
    double start, stop, elapsed;
if (enable_serial){
    start = omp_get_wtime();
    // Execuatable code
    //float* C = square_matrix_multiply_recursive(matrix_A, matrix_B, size);
//#pragma omp parallel
//#pragma omp single nowait
    serial_matrix_mul(matrix_A, matrix_B, matrix_C, size);
    stop = omp_get_wtime();
    elapsed = (double)(stop - start);
    strassen_time = elapsed;
    printf("Time elapsed in ms: %f\n", elapsed);
}

    start = omp_get_wtime();
    // Execuatable code
    //float* C = square_matrix_multiply_recursive(matrix_A, matrix_B, size);
#pragma omp parallel
#pragma omp single nowait
    strassen_matrix_multiply_recursive(zm_a, zm_b, zm_c, size);
    stop = omp_get_wtime();
    elapsed = (double)(stop - start);
    strassen_time = elapsed;
    printf("Strassen algorithm. Time elapsed in ms: %f\n", elapsed);
    write_file_zmorton(zm_c, size, "strassen.bin");

    show_matrix(matrix_A, size, false);
    printf("\n");



    printf("\n");
    show_matrix(zm_c, size, true);
    printf("\n");
    start = omp_get_wtime();
        // Execuatable code
        simple_matrix_mul(matrix_A, matrix_B, matrix_C, size);
    stop = omp_get_wtime();
    elapsed = (double)(stop - start) ;
    stupid_time = elapsed;
    show_matrix(matrix_C, size, false);
    write_file(matrix_C, size, "naive.bin");

    int m;
    i = 0; j = 0; // block (0,0)
    //for(i = 0; i < size / MIN_SQUARE_MATRIX; ++i){
    //    for(j = 0; j < size / MIN_SQUARE_MATRIX; ++j){
    REAL_PTR ptr = zm_c + (i * (size / MIN_SQUARE_MATRIX) + j) * MIN_SQUARE_MATRIX * MIN_SQUARE_MATRIX;
    /*
    for(k = 0; k < MIN_SQUARE_MATRIX; ++k){
                for(m = 0; m < MIN_SQUARE_MATRIX; ++m){
                    //printf("%.2f ", ptr[k * MIN_SQUARE_MATRIX + m]);
                    if(ptr[k * MIN_SQUARE_MATRIX + m] - matrix_C[size * k + m] != 0)
                    {
                       printf("Fail...\n");
                       break;
                    }
                }
                //printf("\n");
            }
            //printf("\n");
    */
    printf("3For algorithm. Time elapsed in ms: %f\n", elapsed);
    printf("Speed up %f\n", stupid_time / strassen_time);

    // Dealloc matrix
    DEALLOC_MATRIX(matrix_A); DEALLOC_MATRIX(matrix_B); DEALLOC_MATRIX(matrix_C);
    return 0;
}

