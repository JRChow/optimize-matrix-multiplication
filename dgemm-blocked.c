#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))


static void inline do_block(int lda, double *A, double *B, double *C) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {  // Row iteration
        for (int j = 0; j < BLOCK_SIZE; j += 4) {  // Column iteration
            __m256d Cval = _mm256_loadu_pd(C + i * lda + j);
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                __m256d Aval = _mm256_set1_pd(A[i * lda + k]);
                __m256d Bval = _mm256_loadu_pd(B + k * lda + j);
                Cval = _mm256_fmadd_pd(Aval, Bval, Cval);
            }
            _mm256_storeu_pd(C + i * lda + j, Cval);
        }
    }
}

// Convert column major to row major.
static void inline swap_ordering(int N, double *src, double *tgt, 
    int src_skip, int tgt_skip) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tgt[i * tgt_skip + j] = src[i + j * src_skip];
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {

    int rounded = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_row = (double *) calloc(rounded * rounded , sizeof(double));
    swap_ordering(lda, A, A_row, lda, rounded);
    double *B_row = (double *) calloc(rounded * rounded ,sizeof(double));
    swap_ordering(lda, B, B_row, lda, rounded);
    double *C_row = (double *) malloc(rounded * rounded * sizeof(double));
    swap_ordering(lda, C, C_row, lda, rounded);

    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // Accumulate block dgemms into block of C
        for (int k = 0; k < lda; k += BLOCK_SIZE) {
            // For each block-column of B
            for (int j = 0; j < lda; j += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                do_block(rounded, 
                    A_row + i * rounded + k, 
                    B_row + k * rounded + j, 
                    C_row + i * rounded + j);
            }
        }
    }
    swap_ordering(lda, C_row, C, rounded, lda);
    free(C_row);
    free(B_row);
    free(A_row);
}
