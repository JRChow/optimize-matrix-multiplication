#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif


#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void inline do_block_naive(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i * lda + j];
            for (int k = 0; k < K; ++k) {
                cij += A[i * lda + k] * B[k * lda + j];
            }
            C[i * lda + j] = cij;
        }
    }
}

static void inline do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    for (int i = 0; i < M; ++i) {  // Row iteration
        for (int j = 0; j < N; j += 4) {  // Column iteration
            __m256d Cval = _mm256_loadu_pd(C + i * lda + j);
            for (int k = 0; k < K; ++k) {
                __m256d Aval = _mm256_set1_pd(A[i * lda + k]);
                __m256d Bval = _mm256_loadu_pd(B + k * lda + j);
                Cval = _mm256_fmadd_pd(Aval, Bval, Cval);
            }
            _mm256_storeu_pd(C + i * lda + j, Cval);
        }
    }
}

// Convert column major to row major.
static void inline swap_ordering(int N, double *src, double *tgt) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tgt[i * N + j] = src[i + j * N];
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    double *A_row = (double *) malloc(lda * lda * sizeof(double));
    swap_ordering(lda, A, A_row);
    double *B_row = (double *) malloc(lda * lda * sizeof(double));
    swap_ordering(lda, B, B_row);
    double *C_row = (double *) malloc(lda * lda * sizeof(double));
    swap_ordering(lda, C, C_row);

    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                if ((M == BLOCK_SIZE) &&
                    (N == BLOCK_SIZE) &&
                    (K == BLOCK_SIZE)) {
                    do_block(lda, M, N, K, A_row + i * lda + k, B_row + k * lda + j, C_row + i * lda + j);
                } else {
                    do_block_naive(lda, M, N, K, A_row + i * lda + k, B_row + k * lda + j, C_row + i * lda + j);
                }
            }
        }
    }
    swap_ordering(lda, C_row, C);
    free(C_row);
    free(B_row);
    free(A_row);
}
