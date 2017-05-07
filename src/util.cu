#include <iostream>        // includes, standard template & armadillo library
#include <armadillo>
//#include <ctime> // CPU timer
#include "../inst/include/density.h"  
#include "../inst/include/common.h"
#include "../inst/include/reduce.h"

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


/* -------------------------------------------------------------------------  
 KDE operations 
------------------------------------------------------------------------- */
void summary(int *nsim, unsigned int *d_R, float *d_RT, float *out) {
    unsigned int maxThread = 256;
    unsigned int nThread = (*nsim < maxThread) ? nextPow2(*nsim) : maxThread;
    unsigned int nBlk    = ((*nsim) + nThread ) / nThread / 2;

    float *h_n1min_out, *h_n1max_out, *h_sum_out, *h_sqsum_out;
    float *d_n1min_out, *d_n1max_out, *d_sum_out, *d_sqsum_out;
    unsigned int *h_count_out, *h_nsim;
    unsigned int *d_count_out, *d_nsim;
  
    size_t dBlkfSize = nBlk * sizeof(float) * 2;
    size_t blkfSize  = nBlk * sizeof(float);
    size_t dBlkuSize = nBlk * sizeof(unsigned int) * 2;
    size_t uSize     = 1 * sizeof(unsigned int);
  
    h_nsim      = (unsigned int *)malloc(uSize);
    h_n1min_out = (float *)malloc(blkfSize);
    h_n1max_out = (float *)malloc(blkfSize);
    h_sum_out   = (float *)malloc(blkfSize);
    h_sqsum_out = (float *)malloc(dBlkfSize);
    h_count_out = (unsigned int *)malloc(dBlkuSize);
    // must reset h_count_out back to 0
    for(int i=0; i<2*nBlk; i++) { h_count_out[i] = 0; } 
    *h_nsim = (unsigned int)*nsim;

    CHECK(cudaMalloc((void**) &d_nsim,      uSize));
    CHECK(cudaMalloc((void**) &d_n1min_out, blkfSize));
    CHECK(cudaMalloc((void**) &d_n1max_out, blkfSize));
    CHECK(cudaMalloc((void**) &d_sum_out,   blkfSize));
    CHECK(cudaMalloc((void**) &d_sqsum_out, dBlkfSize));
    CHECK(cudaMalloc((void**) &d_count_out, dBlkuSize));
  
    CHECK(cudaMemcpy(d_nsim,      h_nsim,  uSize,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_count_out, h_count_out, dBlkuSize, cudaMemcpyHostToDevice));

    // must be first min and then max
    count_kernel<<<2*nBlk, nThread>>>(d_nsim, d_R, d_count_out); cudaFree(d_R);
    n1min_kernel<<<nBlk, nThread>>>(d_RT, d_n1min_out); 
    n1max_kernel<<<nBlk, nThread>>>(d_RT, d_n1max_out);
    sum_kernel<<<nBlk, nThread>>>(d_RT,   d_sum_out);
    squareSum_kernel<<<2*nBlk, nThread>>>(d_nsim, d_RT, d_sqsum_out);
  
    CHECK(cudaMemcpy(h_n1min_out, d_n1min_out, blkfSize,  cudaMemcpyDeviceToHost)); cudaFree(d_n1min_out);
    CHECK(cudaMemcpy(h_n1max_out, d_n1max_out, blkfSize,  cudaMemcpyDeviceToHost)); cudaFree(d_n1max_out);
    CHECK(cudaMemcpy(h_sum_out,   d_sum_out,   blkfSize,  cudaMemcpyDeviceToHost)); cudaFree(d_sum_out);
    CHECK(cudaMemcpy(h_sqsum_out, d_sqsum_out, dBlkfSize, cudaMemcpyDeviceToHost)); cudaFree(d_sqsum_out);
    CHECK(cudaMemcpy(h_count_out, d_count_out, dBlkuSize, cudaMemcpyDeviceToHost)); cudaFree(d_count_out);

    arma::vec min_tmp(nBlk); arma::vec max_tmp(nBlk);
    float sum = 0, sqsum = 0;
    for (int i=0; i<2*nBlk; i++) {
      sqsum += h_sqsum_out[i];
      if ( i < nBlk ) {
        min_tmp[i] = (double)h_n1min_out[i];
        max_tmp[i] = (double)h_n1max_out[i];
        sum += h_sum_out[i];
      }
    }

    free(h_sqsum_out); free(h_n1min_out); free(h_n1max_out); free(h_sum_out);
    out[0] = min_tmp.min();
    out[1] = max_tmp.max();
    out[2] = std::sqrt( (sqsum - (sum*sum) / h_count_out[0]) / (h_count_out[0] - 1) );
    out[3] = h_count_out[0]; free(h_count_out);

    // printf("RT0 [minimum maximum]: %.2f %.2f\n", min_tmp.min(), max_tmp.max());
    // printf("RT0 [sum sqsum]: %.2f %.2f\n", sum, sqsum);
    // printf("RT0 [nsRT0 sd]: %f %f\n", out[3], out[2]);
    free(h_nsim); cudaFree(d_nsim);
}

void histc(int *nsim, int ngrid, float *h_binedge, float *d_RT, unsigned int *h_hist)
{
    size_t ngrid_plus1fSize = (ngrid + 1) * sizeof(float);
    size_t ngriduSize = ngrid * sizeof(unsigned int);

    float *d_binedge;
    unsigned int *d_hist;
    unsigned int *h_nsim, *d_nsim;
    h_nsim  = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    *h_nsim = (unsigned int)*nsim;
    CHECK(cudaMalloc((void**) &d_nsim, sizeof(unsigned int) * 1));
    CHECK(cudaMemcpy(d_nsim,   h_nsim, sizeof(unsigned int) * 1,  cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**) &d_binedge, ngrid_plus1fSize)); // 1025
    CHECK(cudaMalloc((void**) &d_hist,    ngriduSize));       // 1024
    CHECK(cudaMemcpy(d_binedge, h_binedge, ngrid_plus1fSize, cudaMemcpyHostToDevice)); free(h_binedge);
    CHECK(cudaMemcpy(d_hist,    h_hist,    ngriduSize,       cudaMemcpyHostToDevice));
    histc_kernel<<<*nsim/ngrid, ngrid>>>(d_binedge, d_RT, d_nsim, d_hist);
    cudaFree(d_RT); cudaFree(d_binedge); cudaFree(d_nsim);

    CHECK(cudaMemcpy(h_hist, d_hist, ngriduSize, cudaMemcpyDeviceToHost)); 
    cudaFree(d_hist); free(h_nsim); 
}
