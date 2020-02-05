#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 128
#define BLOCK_SIZE_L1 64
#define DIVISOR 12
#endif

#define min(a, b) (((a)<(b))?(a):(b))

// Register blocking
static void inline do_block_register(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int i = 0; i < M; i += 3) {
        for (int j = 0; j < N; j += 12) {
            // Column 0 of C
            register __m256d C_00 = _mm256_loadu_pd(C + (i + 0) * lda + j + 0);
            register __m256d C_10 = _mm256_loadu_pd(C + (i + 0) * lda + j + 4);
            register __m256d C_20 = _mm256_loadu_pd(C + (i + 0) * lda + j + 8);
            // Column 1 of C
            register __m256d C_01 = _mm256_loadu_pd(C + (i + 1) * lda + j + 0);
            register __m256d C_11 = _mm256_loadu_pd(C + (i + 1) * lda + j + 4);
            register __m256d C_21 = _mm256_loadu_pd(C + (i + 1) * lda + j + 8);
            // Column 2 of C
            register __m256d C_02 = _mm256_loadu_pd(C + (i + 2) * lda + j + 0);
            register __m256d C_12 = _mm256_loadu_pd(C + (i + 2) * lda + j + 4);
            register __m256d C_22 = _mm256_loadu_pd(C + (i + 2) * lda + j + 8);

            for (int k = 0; k < K; k++) {
                register __m256d b_00_03 = _mm256_loadu_pd(B + k * lda + j * 12);
                register __m256d b_00_03 = _mm256_loadu_pd(B + k * lda + j * 12);
                register __m256d b_00_03 = _mm256_loadu_pd(B + k * lda + j * 12);
                register __m256d a00 = _mm256_broadcast_sd(A + (4 * i) * lda + k);



            }
        }
    }
}

//static void inline do_block_l1(int lda, int M, int N, int K, double *A, double *B, double *C) {
//    for (int j = 0; j < BLOCK_SIZE_L1; j++) {  // Column iteration
//        for (int i = 0; i < BLOCK_SIZE_L1; i += 4) {  // Row iteration
//            __m256d Cval = _mm256_loadu_pd(C + i + j * lda);
//            for (int k = 0; k < BLOCK_SIZE_L1; ++k) {
//                __m256d Aval = _mm256_loadu_pd(A + i + k * lda);
//                __m256d Bval = _mm256_set1_pd(B[k + j * lda]);
//                Cval = _mm256_fmadd_pd(Aval, Bval, Cval);
//            }
//            _mm256_storeu_pd(C + i + j * lda, Cval);
//        }
//    }
//}

// Level-1 blocking
static void inline do_block_l1(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int j = 0; j < N; j += BLOCK_SIZE_L1) {
        for (int k = 0; k < K; k += BLOCK_SIZE_L1) {
            for (int i = 0; i < M; i += BLOCK_SIZE_L1) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_reg = min (BLOCK_SIZE_L2, M - i);
                int N_reg = min (BLOCK_SIZE_L2, N - j);
                int K_reg = min (BLOCK_SIZE_L2, K - k);
                do_block_register(lda, M_reg, N_reg, K_reg,
                            A + i + k * lda,
                            B + k + j * lda,
                            C + i + j * lda);
            }
        }
    }
}

// Level-2 blocking
static void inline do_block_l2(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int j = 0; j < N; j += BLOCK_SIZE_L1) {
        for (int k = 0; k < K; k += BLOCK_SIZE_L1) {
            for (int i = 0; i < M; i += BLOCK_SIZE_L1) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_L1 = min (BLOCK_SIZE_L2, M - i);
                int N_L1 = min (BLOCK_SIZE_L2, N - j);
                int K_L1 = min (BLOCK_SIZE_L2, K - k);
                do_block_l1(lda, M_L1, N_L1, K_L1,
                            A + i + k * lda,
                            B + k + j * lda,
                            C + i + j * lda);
            }
        }
    }
}

// Perform padding / de-padding
static void inline padding(int N, const double *src, double *tgt, int src_skip, int tgt_skip) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tgt[i * tgt_skip + j] = src[i * src_skip + j];
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    // Determine padding
    int lda_pad = lda;
    int r = lda % DIVISOR;
    if (r != 0) {
        lda_pad += DIVISOR - r;
    }

    double *A_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    padding(lda, A, A_pad, lda, lda_pad);
    double *B_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    padding(lda, B, B_pad, lda, lda_pad);
    double *C_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    padding(lda, C, C_pad, lda, lda_pad);

    for (int j = 0; j < lda_pad; j += BLOCK_SIZE_L2) {
        for (int k = 0; k < lda_pad; k += BLOCK_SIZE_L2) {
            for (int i = 0; i < lda_pad; i += BLOCK_SIZE_L2) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE_L2, lda_pad - i);
                int N = min (BLOCK_SIZE_L2, lda_pad - j);
                int K = min (BLOCK_SIZE_L2, lda_pad - k);
                do_block_l2(lda_pad, M, N, K,
                            A_pad + i + k * lda_pad,
                            B_pad + k + j * lda_pad,
                            C_pad + i + j * lda_pad);
            }
        }
    }

    // Un-pad C
    padding(lda, C_pad, C, lda_pad, lda);
    free(C_pad);
    free(B_pad);
    free(A_pad);
}
