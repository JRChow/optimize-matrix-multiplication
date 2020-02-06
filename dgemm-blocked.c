#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 128
#define BLOCK_SIZE_L1 64
#define DIVISOR 8
#endif

#define min(a, b) (((a)<(b))?(a):(b))


static void inline do_block_naive(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

// Register blocking
static void inline do_block_register(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int j = 0; j < N; j += 4) {
        for (int i = 0; i < M; i += 8) {
            // Column 0 of C
            register __m256d C_00 = _mm256_loadu_pd(C + (i + 0) + (j + 0) * lda);
            register __m256d C_10 = _mm256_loadu_pd(C + (i + 4) + (j + 0) * lda);
            // Column 1 of C
            register __m256d C_01 = _mm256_loadu_pd(C + (i + 0) + (j + 1) * lda);
            register __m256d C_11 = _mm256_loadu_pd(C + (i + 4) + (j + 1) * lda);
            // Column 2 of C
            register __m256d C_02 = _mm256_loadu_pd(C + (i + 0) + (j + 2) * lda);
            register __m256d C_12 = _mm256_loadu_pd(C + (i + 4) + (j + 2) * lda);
            // Column 3 of C
            register __m256d C_03 = _mm256_loadu_pd(C + (i + 0) + (j + 3) * lda);
            register __m256d C_13 = _mm256_loadu_pd(C + (i + 4) + (j + 3) * lda);

            for (int k = 0; k < K; k++) {
                // Load A vectors
                register __m256d A0 = _mm256_loadu_pd(A + (i + 0) + k * lda);
                register __m256d A1 = _mm256_loadu_pd(A + (i + 4) + k * lda);
                // Load B scalars
                register __m256d B0 = _mm256_set1_pd(B[k + (j + 0) * lda]);
                register __m256d B1 = _mm256_set1_pd(B[k + (j + 1) * lda]);
                register __m256d B2 = _mm256_set1_pd(B[k + (j + 2) * lda]);
                register __m256d B3 = _mm256_set1_pd(B[k + (j + 3) * lda]);
                // Compute
                C_00 = _mm256_fmadd_pd(A0, B0, C_00);
                C_10 = _mm256_fmadd_pd(A1, B0, C_10);

                C_01 = _mm256_fmadd_pd(A0, B1, C_01);
                C_11 = _mm256_fmadd_pd(A1, B1, C_11);

                C_02 = _mm256_fmadd_pd(A0, B2, C_02);
                C_12 = _mm256_fmadd_pd(A1, B2, C_12);

                C_03 = _mm256_fmadd_pd(A0, B3, C_03);
                C_13 = _mm256_fmadd_pd(A1, B3, C_13);
            }
            // Write C back to memory
            _mm256_store_pd(C + (i + 0) + (j + 0) * lda, C_00);
            _mm256_store_pd(C + (i + 4) + (j + 0) * lda, C_10);

            _mm256_store_pd(C + (i + 0) + (j + 1) * lda, C_01);
            _mm256_store_pd(C + (i + 4) + (j + 1) * lda, C_11);

            _mm256_store_pd(C + (i + 0) + (j + 2) * lda, C_02);
            _mm256_store_pd(C + (i + 4) + (j + 2) * lda, C_12);

            _mm256_store_pd(C + (i + 0) + (j + 3) * lda, C_03);
            _mm256_store_pd(C + (i + 4) + (j + 3) * lda, C_13);
        }
    }
}

// Level-1 blocking
static void inline do_block_l1(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int j = 0; j < N; j += BLOCK_SIZE_L1) {
        for (int k = 0; k < K; k += BLOCK_SIZE_L1) {
            for (int i = 0; i < M; i += BLOCK_SIZE_L1) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_reg = min(BLOCK_SIZE_L1, M - i);
                int N_reg = min(BLOCK_SIZE_L1, N - j);
                int K_reg = min(BLOCK_SIZE_L1, K - k);
                do_block_register(lda, M_reg, N_reg, K_reg,
                                  A + i + k * lda,
                                  B + k + j * lda,
                                  C + i + j * lda);
//                do_block_naive(lda, M_reg, N_reg, K_reg,
//                                  A + i + k * lda,
//                                  B + k + j * lda,
//                                  C + i + j * lda);
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

//    double *A_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    double *A_pad = (double *) calloc(lda_pad * lda_pad, sizeof(double));
    padding(lda, A, A_pad, lda, lda_pad);
//    double *B_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    double *B_pad = (double *) calloc(lda_pad * lda_pad, sizeof(double));
    padding(lda, B, B_pad, lda, lda_pad);
//    double *C_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    double *C_pad = (double *) malloc(lda_pad * lda_pad * sizeof(double));
    padding(lda, C, C_pad, lda, lda_pad);

    for (int j = 0; j < lda_pad; j += BLOCK_SIZE_L2) {
        for (int k = 0; k < lda_pad; k += BLOCK_SIZE_L2) {
            for (int i = 0; i < lda_pad; i += BLOCK_SIZE_L2) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE_L2, lda_pad - i);
                int N = min(BLOCK_SIZE_L2, lda_pad - j);
                int K = min(BLOCK_SIZE_L2, lda_pad - k);
                do_block_l1(lda_pad, M, N, K,
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
