#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 12
#endif

static void inline do_block(int lda, double *A, double *B, double *C) {
    for (int j = 0; j < BLOCK_SIZE; j++) {  // Column iteration
        for (int i = 0; i < BLOCK_SIZE; i += 4) {  // Row iteration
            __m256d Cval = _mm256_loadu_pd(C + i + j * lda);
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                __m256d Aval = _mm256_loadu_pd(A + i + k * lda);
                __m256d Bval = _mm256_set1_pd(B[k + j * lda]);
                Cval = _mm256_fmadd_pd(Aval, Bval, Cval);
            }
            _mm256_storeu_pd(C + i + j * lda, Cval);
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

    int rounded = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_row = (double *) calloc(rounded * rounded, sizeof(double));
    padding(lda, A, A_row, lda, rounded);
    double *B_row = (double *) calloc(rounded * rounded, sizeof(double));
    padding(lda, B, B_row, lda, rounded);
    double *C_row = (double *) malloc(rounded * rounded * sizeof(double));
    padding(lda, C, C_row, lda, rounded);

    // For each block-column of B
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
        // Accumulate block dgemms into block of C
        for (int k = 0; k < lda; k += BLOCK_SIZE) {
            // For each block-row of A
            for (int i = 0; i < lda; i += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                do_block(rounded,
                         A_row + i + k * rounded,
                         B_row + k + j * rounded,
                         C_row + i + j * rounded);
            }
        }
    }

    padding(lda, C_row, C, rounded, lda);
    free(C_row);
    free(B_row);
    free(A_row);
}
