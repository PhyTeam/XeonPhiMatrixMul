
#include <stdio.h>
#include <stdlib.h>

#define MY_VLENGTH 8
#define MAX_CORE_THREADS 61
#define MAX_HARDWARE_THREADS 4

#define A_TILE_ROW m_size
#define A_TILE_COL k_size
#define C_TILE_ROW m_size
#define C_TILE_COL 30

#define B_TILE_VECTOR_COL 8

#define B_TILDE(I,J) b_tilde[(I)*(order)+(J)]

#define A_TILDE(I,J,K) a_tilde[(I)+((J)+(K)*kc)*mr]

#define A(I,J) a[(I)*(order + padding_stride)+(J)]
#define B(I,J) b[(I)*(order + padding_stride)+(J)]
#define C(I,J) c[(I)*(order + padding_stride)+(J)]

#define forder ((double) 1.0*order)

#define VB(I,J,K) b[(I)*(order + padding_stride)+(J):(K)]
#define VC(I,J,K) c[(I)*(order + padding_stride)+(J):(K)]

#define VB_TILDE(I,J,K) b_tilde[((I)*kc+(J))*(K):(K)]

#ifdef __MIC_MMAP__

#include <linux/mman.h>
#include <sys/mman.h>

#define aligned_malloc(size,alignment) \
            mmap(NULL, size, PROT_READ | PROT_WRITE, \
                        MAP_PRIVATE | MAP_HUGETLB | MAP_ANONYMOUS | MAP_POPULATE, 0, 0);
#define aligned_free(addr,size) munmap(addr, size);

#elif __MIC_MM_MALLOC_8192__

#define my_malloc(size) _mm_malloc(size,8192)
#define my_free(addr,size) _mm_free(addr)

#elif __MIC_MM_MALLOC_64__

#define my_malloc(size) _mm_malloc(size,64)
#define my_free(addr,size) _mm_free(addr)

#else
#define my_malloc(size) malloc(size)
#define my_free(addr,size) free(addr)

#endif

#ifdef __MIC__

#if __MIC_MMAP__
#define PADDING 8*64
#define my_malloc(size) aligned_malloc(size+PADDING,8192)
#define my_free(addr,size) aligned_free(addr,size+PADDING)
#endif

#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#if defined(_OPENMP)
  #include <omp.h>
#elif defined(MPI)
  #include "mpi.h"
#else
  #include <sys/time.h>
  #define  USEC_TO_SEC   1.0e-6    /* to convert microsecs to secs */
#define NULL 0
#endif

double wtime() {
  double time_seconds;

#if defined(_OPENMP)
  time_seconds = omp_get_wtime();

#elif defined(MPI)
  time_seconds = MPI_Wtime();

#else
  struct timeval  time_data; /* seconds since 0 GMT */

  gettimeofday(&time_data,NULL);

  time_seconds  = (double) time_data.tv_sec;
  time_seconds += (double) time_data.tv_usec * USEC_TO_SEC;
#endif

  return time_seconds;
}

main(int argc, char **argv) {

    int     i, iter, j, l;
    int     iterations;           // number of times the multiplication is done
    double  dgemm_time,           // timing parameters
    avgtime = 0.0,
    maxtime = 0.0,
    mintime = 366.0*24.0*3600.0;  // set the minimum time to a large value;
                                  // one leap year should be enough
    double  checksum = 0.0,       // checksum of result
            ref_checksum;
    double  epsilon = 1.e-8;      // error tolerance
    int     nthread_input,        // thread parameters
            nthread_input2,
            nthread,
            nthread2;
    int     num_error=0;          // flag that signals that requested and
                                  // obtained numbers of threads are the same
    static
    double  *a, *b, *c;           // input (A,B) and output (C) matrices
    double  *b_tilde;
    int     order;                // number of rows and columns of matrices
    int padding, padding_stride, vlength;
    int k_size, m_size;
    int ic, iic, iiic, ir, iir, jc, jjc, jr, k, kc, m, mc, mr, n, nc, nr, pc, ppc;

    if (argc != 7) {
        printf("Usage: %s <# core threads> <# hardware threads> <# iterations> <matrix tile k-size> <matrix tile m-size> <matrix order>\n",*argv);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the number of core threads to use
    nthread_input = atoi(*++argv);

    if ((nthread_input < 1) || (nthread_input > MAX_CORE_THREADS)) {
        printf("ERROR: Invalid number of core threads: %d\n", nthread_input);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the number of hardware threads to use
    nthread_input2 = atoi(*++argv);

    if ((nthread_input2 < 1) || (nthread_input2 > MAX_HARDWARE_THREADS)) {
        printf("ERROR: Invalid number of hardware threads: %d\n", nthread_input2);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the number of iterations to perform
    iterations = atoi(*++argv);
    if (iterations < 1) {
        printf("ERROR: Iterations must be positive: %d\n", iterations);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the matrix-tile k-size
    k_size = atoi(*++argv);
    if (k_size < 1) {
        printf("ERROR: Matrix tile k-size must be positive: %d\n", k_size);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the matrix-tile m-size
    m_size = atoi(*++argv);
    if (m_size < 1) {
        printf("ERROR: Matrix tile m-size must be positive: %d\n", m_size);
        exit(EXIT_FAILURE);
    }

    // Obtain from the command-line, the matrix size
    order = atoi(*++argv);
    if (order < 1) {
        printf("ERROR: Matrix order must be positive: %d\n", order);
        exit(EXIT_FAILURE);
    }

    padding = 64 - ((order*sizeof(double)) % 64);
    padding_stride = padding / sizeof(double);
	a = my_malloc(sizeof(double)*(order+padding)*order);
 	b = my_malloc(sizeof(double)*(order+padding)*order);
 	c = my_malloc(sizeof(double)*(order+padding)*order);

    printf("Matrix padding = %d\n", padding); fflush(0);
    printf("Matrix padding stride = %d\n", padding_stride); fflush(0);
    if (!a || !b || !c) {
        printf("ERROR: Could not allocate space for global matrices\n");
        exit(EXIT_FAILURE);
    } else {
        printf("a = %ld; b = %ld; c = %ld\n\n", (long) a, (long) b, (long) c); fflush(0);
	}

    ref_checksum = (0.25*forder*forder*forder*(forder-1.0)*(forder-1.0));

    #pragma omp parallel for private(i,j)
    for (i = 0; i < order; i++) {
        #pragma vector aligned
        #pragma simd
        for (j = 0; j < order; j++) {
            A(i,j) = (double) j;
            B(i,j) = (double) j;
            C(i,j) = 0.0;
        }
    }

    printf("OpenMP Dense matrix-matrix multiplication\n"); fflush(0);
    printf("Matrix order          = %d\n", order); fflush(0);
    printf("Matrix tile k-size         = %d\n", k_size); fflush(0);
    printf("Matrix tile m-size         = %d\n", m_size); fflush(0);
    printf("Number of core threads     = %d\n", nthread_input); fflush(0);
    printf("Number of hardware threads     = %d\n", nthread_input2); fflush(0);
    printf("Number of iterations  = %d\n", iterations); fflush(0);

    k = order;
    kc = k_size;
    m = order;
    mc = m_size;
    mr = C_TILE_COL;
    n = order;
    nc = n;
    nr = B_TILE_VECTOR_COL;
	vlength = 8;

 	b_tilde = my_malloc(sizeof(double)*kc*order);

	// Record the start time
	dgemm_time = wtime();

    // Timing loop for improved performance measurement
    for (iter = 0; iter < iterations; iter++) {
      // Offload
#pragma offload target(mic:0)\
    in(a:length((order+padding)*order)), \
    in(b:length((order+padding)*order)), \
    in(b_tilde:length(kc*order)), \
    in(order, padding, padding_stride, vlength, k_size, m_size), \
    inout(c:length(order+padding)*order)
{
		for (jc = 0; jc < n; jc += nc) {
			for (pc = 0; pc < k; pc += kc) {
				#pragma omp parallel
				{
				// Transfer data of size "kc" by "nc" from "B" to buffer "B_TILDE" which is to be shared by all
				// of the threads
				#pragma omp for firstprivate(kc,nc,pc,vlength) private(j,jjc,ppc)
				for (ppc = 0; ppc < kc; ppc++) {
					j = 0;
					for (jjc = 0; jjc < nc; jjc+=8) {
						#pragma vector aligned
						VB_TILDE(j,ppc,MY_VLENGTH) = VB(ppc+pc,jjc,MY_VLENGTH);
						j++;
					}
				}
				__declspec(align(64)) double a_tilde[mc*kc];
				#pragma omp for firstprivate(jc,pc,vlength) private(ic,iic,iiic,l,ppc,jr,ir)
				for (ic = 0; ic < m; ic += mc) {
					// Transfer data of size "mc" by "kc" from "A" to buffer "A_TILDE" which will be used by each
					// core thread
					for (iic = 0,l = 0; iic < mc; iic+=mr,l++) {
						#pragma vector aligned
						#pragma simd
						for (ppc = 0; ppc < kc; ppc++) {
							A_TILDE(0,ppc,l) = A(iic+ic,ppc+pc);
							A_TILDE(1,ppc,l) = A(iic+ic+1,ppc+pc);
							A_TILDE(2,ppc,l) = A(iic+ic+2,ppc+pc);
							A_TILDE(3,ppc,l) = A(iic+ic+3,ppc+pc);
							A_TILDE(4,ppc,l) = A(iic+ic+4,ppc+pc);
							A_TILDE(5,ppc,l) = A(iic+ic+5,ppc+pc);
							A_TILDE(6,ppc,l) = A(iic+ic+6,ppc+pc);
							A_TILDE(7,ppc,l) = A(iic+ic+7,ppc+pc);
							A_TILDE(8,ppc,l) = A(iic+ic+8,ppc+pc);
							A_TILDE(9,ppc,l) = A(iic+ic+9,ppc+pc);
							A_TILDE(10,ppc,l) = A(iic+ic+10,ppc+pc);
							A_TILDE(11,ppc,l) = A(iic+ic+11,ppc+pc);
							A_TILDE(12,ppc,l) = A(iic+ic+12,ppc+pc);
							A_TILDE(13,ppc,l) = A(iic+ic+13,ppc+pc);
							A_TILDE(14,ppc,l) = A(iic+ic+14,ppc+pc);
							A_TILDE(15,ppc,l) = A(iic+ic+15,ppc+pc);
							A_TILDE(16,ppc,l) = A(iic+ic+16,ppc+pc);
							A_TILDE(17,ppc,l) = A(iic+ic+17,ppc+pc);
							A_TILDE(18,ppc,l) = A(iic+ic+18,ppc+pc);
							A_TILDE(19,ppc,l) = A(iic+ic+19,ppc+pc);
							A_TILDE(20,ppc,l) = A(iic+ic+20,ppc+pc);
							A_TILDE(21,ppc,l) = A(iic+ic+21,ppc+pc);
							A_TILDE(22,ppc,l) = A(iic+ic+22,ppc+pc);
							A_TILDE(23,ppc,l) = A(iic+ic+23,ppc+pc);
							A_TILDE(24,ppc,l) = A(iic+ic+24,ppc+pc);
							A_TILDE(25,ppc,l) = A(iic+ic+25,ppc+pc);
							A_TILDE(26,ppc,l) = A(iic+ic+26,ppc+pc);
							A_TILDE(27,ppc,l) = A(iic+ic+27,ppc+pc);
							A_TILDE(28,ppc,l) = A(iic+ic+28,ppc+pc);
							A_TILDE(29,ppc,l) = A(iic+ic+29,ppc+pc);
						}
					}
					#pragma omp parallel for firstprivate(ic,vlength) private(ir,iir,j,jr,l,ppc)
					for (jr = 0,j = 0; jr < nc; jr += nr,j++) {
						__declspec(align(64)) double t0[8], t1[8], t2[8], t3[8], t4[8], t5[8];
						__declspec(align(64)) double t6[8], t7[8], t8[8], t9[8], t10[8], t11[8];
						__declspec(align(64)) double t12[8], t13[8], t14[8], t15[8], t16[8], t17[8];
						__declspec(align(64)) double t18[8], t19[8], t20[8], t21[8], t22[8], t23[8];
						__declspec(align(64)) double t24[8], t25[8], t26[8], t27[8], t28[8], t29[8];
						__declspec(align(64)) double *pa_tilde;
						for (ir = 0, iir = 0, l = 0; ir < mc; ir += mr,l++) {
							// Load thirty vector registers with 8 columns of the "C" matrix
							#pragma vector aligned
							t0[0:MY_VLENGTH] = VC(ic+ir,jr,MY_VLENGTH);
							#pragma vector aligned
							t1[0:MY_VLENGTH] = VC(ic+ir+1,jr,MY_VLENGTH);
							#pragma vector aligned
							t2[0:MY_VLENGTH] = VC(ic+ir+2,jr,MY_VLENGTH);
							#pragma vector aligned
							t3[0:MY_VLENGTH] = VC(ic+ir+3,jr,MY_VLENGTH);
							#pragma vector aligned
							t4[0:MY_VLENGTH] = VC(ic+ir+4,jr,MY_VLENGTH);
							#pragma vector aligned
							t5[0:MY_VLENGTH] = VC(ic+ir+5,jr,MY_VLENGTH);
							#pragma vector aligned
							t6[0:MY_VLENGTH] = VC(ic+ir+6,jr,MY_VLENGTH);
							#pragma vector aligned
							t7[0:MY_VLENGTH] = VC(ic+ir+7,jr,MY_VLENGTH);
							#pragma vector aligned
							t8[0:MY_VLENGTH] = VC(ic+ir+8,jr,MY_VLENGTH);
							#pragma vector aligned
							t9[0:MY_VLENGTH] = VC(ic+ir+9,jr,MY_VLENGTH);
							#pragma vector aligned
							t10[0:MY_VLENGTH] = VC(ic+ir+10,jr,MY_VLENGTH);
							#pragma vector aligned
							t11[0:MY_VLENGTH] = VC(ic+ir+11,jr,MY_VLENGTH);
							#pragma vector aligned
							t12[0:MY_VLENGTH] = VC(ic+ir+12,jr,MY_VLENGTH);
							#pragma vector aligned
							t13[0:MY_VLENGTH] = VC(ic+ir+13,jr,MY_VLENGTH);
							#pragma vector aligned
							t14[0:MY_VLENGTH] = VC(ic+ir+14,jr,MY_VLENGTH);
							#pragma vector aligned
							t15[0:MY_VLENGTH] = VC(ic+ir+15,jr,MY_VLENGTH);
							#pragma vector aligned
							t16[0:MY_VLENGTH] = VC(ic+ir+16,jr,MY_VLENGTH);
							#pragma vector aligned
							t17[0:MY_VLENGTH] = VC(ic+ir+17,jr,MY_VLENGTH);
							#pragma vector aligned
							t18[0:MY_VLENGTH] = VC(ic+ir+18,jr,MY_VLENGTH);
							#pragma vector aligned
							t19[0:MY_VLENGTH] = VC(ic+ir+19,jr,MY_VLENGTH);
							#pragma vector aligned
							t20[0:MY_VLENGTH] = VC(ic+ir+20,jr,MY_VLENGTH);
							#pragma vector aligned
							t21[0:MY_VLENGTH] = VC(ic+ir+21,jr,MY_VLENGTH);
							#pragma vector aligned
							t22[0:MY_VLENGTH] = VC(ic+ir+22,jr,MY_VLENGTH);
							#pragma vector aligned
							t23[0:MY_VLENGTH] = VC(ic+ir+23,jr,MY_VLENGTH);
							#pragma vector aligned
							t24[0:MY_VLENGTH] = VC(ic+ir+24,jr,MY_VLENGTH);
							#pragma vector aligned
							t25[0:MY_VLENGTH] = VC(ic+ir+25,jr,MY_VLENGTH);
							#pragma vector aligned
							t26[0:MY_VLENGTH] = VC(ic+ir+26,jr,MY_VLENGTH);
							#pragma vector aligned
							t27[0:MY_VLENGTH] = VC(ic+ir+27,jr,MY_VLENGTH);
							#pragma vector aligned
							t28[0:MY_VLENGTH] = VC(ic+ir+28,jr,MY_VLENGTH);
							#pragma vector aligned
							t29[0:MY_VLENGTH] = VC(ic+ir+29,jr,MY_VLENGTH);

							for (ppc = 0; ppc < kc; ppc++) {
								pa_tilde = &(A_TILDE(iir,ppc,l));
								#pragma vector aligned
								t0[0:MY_VLENGTH] += pa_tilde[0] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t1[0:MY_VLENGTH] += pa_tilde[1] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t2[0:MY_VLENGTH] += pa_tilde[2] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t3[0:MY_VLENGTH] += pa_tilde[3] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t4[0:MY_VLENGTH] += pa_tilde[4] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t5[0:MY_VLENGTH] += pa_tilde[5] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t6[0:MY_VLENGTH] += pa_tilde[6] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t7[0:MY_VLENGTH] += pa_tilde[7] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t8[0:MY_VLENGTH] += pa_tilde[8] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t9[0:MY_VLENGTH] += pa_tilde[9] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t10[0:MY_VLENGTH] += pa_tilde[10] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t11[0:MY_VLENGTH] += pa_tilde[11] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t12[0:MY_VLENGTH] += pa_tilde[12] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t13[0:MY_VLENGTH] += pa_tilde[13] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t14[0:MY_VLENGTH] += pa_tilde[14] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t15[0:MY_VLENGTH] += pa_tilde[15] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t16[0:MY_VLENGTH] += pa_tilde[16] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t17[0:MY_VLENGTH] += pa_tilde[17] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t18[0:MY_VLENGTH] += pa_tilde[18] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t19[0:MY_VLENGTH] += pa_tilde[19] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t20[0:MY_VLENGTH] += pa_tilde[20] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t21[0:MY_VLENGTH] += pa_tilde[21] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t22[0:MY_VLENGTH] += pa_tilde[22] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t23[0:MY_VLENGTH] += pa_tilde[23] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t24[0:MY_VLENGTH] += pa_tilde[24] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t25[0:MY_VLENGTH] += pa_tilde[25] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t26[0:MY_VLENGTH] += pa_tilde[26] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t27[0:MY_VLENGTH] += pa_tilde[27] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t28[0:MY_VLENGTH] += pa_tilde[28] * VB_TILDE(j,ppc,MY_VLENGTH);
								#pragma vector aligned
								t29[0:MY_VLENGTH] += pa_tilde[29] * VB_TILDE(j,ppc,MY_VLENGTH);
							}

							// Store thirty vector register results back into the "C" matrix
							#pragma vector aligned
							VC(ic+ir,jr,MY_VLENGTH) = t0[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+1,jr,MY_VLENGTH) = t1[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+2,jr,MY_VLENGTH) = t2[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+3,jr,MY_VLENGTH) = t3[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+4,jr,MY_VLENGTH) = t4[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+5,jr,MY_VLENGTH) = t5[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+6,jr,MY_VLENGTH) = t6[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+7,jr,MY_VLENGTH) = t7[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+8,jr,MY_VLENGTH) = t8[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+9,jr,MY_VLENGTH) = t9[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+10,jr,MY_VLENGTH) = t10[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+11,jr,MY_VLENGTH) = t11[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+12,jr,MY_VLENGTH) = t12[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+13,jr,MY_VLENGTH) = t13[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+14,jr,MY_VLENGTH) = t14[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+15,jr,MY_VLENGTH) = t15[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+16,jr,MY_VLENGTH) = t16[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+17,jr,MY_VLENGTH) = t17[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+18,jr,MY_VLENGTH) = t18[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+19,jr,MY_VLENGTH) = t19[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+20,jr,MY_VLENGTH) = t20[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+21,jr,MY_VLENGTH) = t21[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+22,jr,MY_VLENGTH) = t22[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+23,jr,MY_VLENGTH) = t23[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+24,jr,MY_VLENGTH) = t24[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+25,jr,MY_VLENGTH) = t25[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+26,jr,MY_VLENGTH) = t26[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+27,jr,MY_VLENGTH) = t27[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+28,jr,MY_VLENGTH) = t28[0:MY_VLENGTH];
							#pragma vector aligned
							VC(ic+ir+29,jr,MY_VLENGTH) = t29[0:MY_VLENGTH];
						}
					}
				}
				} // End of omp parallel pragma
    		}
    	}
      }// End offload
    } // End of "iter" for-loop

	// Record the completion time
   	dgemm_time = wtime() - dgemm_time;

   	if (iter>0 || iterations==1) { // Skip the first iteration
		avgtime = avgtime + dgemm_time;
		mintime = MIN(mintime, dgemm_time);
		maxtime = MAX(maxtime, dgemm_time);
	}

    for (checksum=0.0,j = 0; j < order; j++)
        for (i = 0; i < order; i++)
            checksum += C(i,j);

    // Verification test
    ref_checksum *= iterations;

    if (ABS((checksum - ref_checksum)/ref_checksum) > epsilon) {
        printf("ERROR: Checksum = %lf, Reference checksum = %lf\n",
        checksum, ref_checksum);
        exit(EXIT_FAILURE);
    } else {
        printf("Solution validates\n");
#ifdef VERBOSE
        printf("Reference checksum = %lf, checksum = %lf\n", ref_checksum, checksum);
#endif
    }

    double nflops = 2.0*forder*forder*forder;
    avgtime = avgtime/(double)(MAX(iterations-1,1));
    printf("Rate (MFlops/s): %lf,  Avg time (s): %lf,  Min time (s): %lf",
         1.0E-06*nflops/mintime, avgtime, mintime);
    printf(", Max time (s): %lf\n", maxtime);

	// Return the allocated storage back to the free-list
	my_free(a,sizeof(double)*(order+padding)*order);
 	my_free(b,sizeof(double)*(order+padding)*order);
 	my_free(c,sizeof(double)*(order+padding)*order);
 	my_free(b_tilde,sizeof(double)*kc*order);

    exit(EXIT_SUCCESS);

}
