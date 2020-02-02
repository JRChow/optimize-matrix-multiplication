#include <immintrin.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
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
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i * lda + k] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

static void inline do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i * lda + k] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

static void inline swap_ordering(int N, double *src, double *tgt){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
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
                if((M == BLOCK_SIZE) && 
                    (N == BLOCK_SIZE) &&
                    (K == BLOCK_SIZE)){
                    do_block(lda, M, N, K, A_row + i * lda + k, B + k + j * lda, C + i + j * lda);
                }else{
                    do_block_naive(lda, M, N, K, A_row + i * lda + k, B + k + j * lda, C + i + j * lda);
                }
            }
        }
    }
    free(A_row);
}
