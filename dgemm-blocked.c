#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 128
#define BLOCK_SIZE_L1 64
#define DIVISOR 8
#endif

#define min(a, b) (((a)<(b))?(a):(b))

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
            }
        }
    }
}

// Perform padding
static void inline pad(int original_N, int padded_N, const double *original, double *padded) {
    for (int j = 0; j < padded_N; j++) {
        for (int i = 0; i < padded_N; i++) {
            if (i < original_N && j < original_N) {
                padded[i + j * padded_N] = original[i + j * original_N];
            } else {
                padded[i + j * padded_N] = 0;
            }
        }
    }
}

// Performs de-padding
static void inline de_pad(int original_N, int padded_N, double *original, const double *padded) {
    for (int j = 0; j < original_N; j++) {
        for (int i = 0; i < original_N; i++) {
            original[i + j * original_N] = padded[i + j * padded_N];
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
    double *A_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    pad(lda, lda_pad, A, A_pad);
//    double *B_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    double *B_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    pad(lda, lda_pad, B, B_pad);
//    double *C_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    double *C_pad = (double *) _mm_malloc(lda_pad * lda_pad * sizeof(double), 32);
    pad(lda, lda_pad, C, C_pad);

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
    de_pad(lda, lda_pad, C, C_pad);
    _mm_free(C_pad);
    _mm_free(B_pad);
    _mm_free(A_pad);
}
