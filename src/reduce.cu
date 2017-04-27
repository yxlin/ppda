#include <unistd.h>
#include <stdio.h>  // C printing
#include "../inst/include/common.h"  
#include <armadillo> // Armadillo vector operations

#define MAX_BLOCK_DIM_SIZE 65535

extern "C" void reduce0_test(double *RT0, int *nRT0, int *nth, double *out);
extern "C" void sum_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void min_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void max_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void minmax_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void sqsum_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void sd_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out);
extern "C" void count_entry(int *n, int *R, int *nth, bool *debug, double *out);

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__global__ void min_kernel(double *g_idata, double *g_odata) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = fmin(g_idata[i], g_idata[i + blockDim.x]);
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) {
             sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void max_kernel(double *g_idata, double *g_odata) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = fmax(g_idata[i], g_idata[i + blockDim.x]);
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) {
             sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void minmax_kernel(double *g_idata, double *g_odata) {
    __shared__ double sdata1[256];
    __shared__ double sdata2[256];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata1[tid] = fmin(g_idata[i], g_idata[i + blockDim.x]);
    sdata2[tid] = fmax(g_idata[i], g_idata[i + blockDim.x]);
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) {
             sdata1[tid] = fmin(sdata1[tid], sdata1[tid + s]);
             sdata2[tid] = fmax(sdata2[tid], sdata2[tid + s]);
      }
      __syncthreads();
    }
    if(tid==0) {
            g_odata[blockIdx.x]             = sdata1[0];
            g_odata[blockIdx.x + gridDim.x] = sdata2[0];
    }
}


__global__ void sum_kernel(double *g_idata, double *g_odata) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) {
            sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void squareSum_kernel(unsigned int* n, double *g_idata, double *g_odata) {
    __shared__ double sdata[256];
    unsigned int numThreads = blockDim.x * gridDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp;
    
    for (unsigned int i=threadID; i<(*n); i += numThreads) {
            // g_tmp[i] = g_idata[i]  * g_idata[i];
            tmp = g_idata[i]  * g_idata[i];
    }
    // sdata[tid] = g_tmp[threadID];
    sdata[tid] = tmp;
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) {
              sdata[tid] += sdata[tid+s];
      }
      __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void count_kernel(unsigned int *n, unsigned int *R, unsigned int *out) {
    __shared__ unsigned int sdata[256];
    sdata[threadIdx.x] = 0;
    __syncthreads();
    
        const int numThreads = blockDim.x * gridDim.x; // total # of threads
        const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int simR;

        for (int i = threadID; i<(*n); i += numThreads) {
          simR = R[i];

          if (simR == 1) {
              atomicAdd(&(sdata[0]), 1);
          } else if (simR == 2) {
              atomicAdd(&(sdata[1]), 1);
          } else {
              atomicAdd(&(sdata[2]), 1);
          }
        }
        __syncthreads();
        // add partial results in each block together.
        atomicAdd( &(out[threadIdx.x]), sdata[threadIdx.x] );
}

void sum_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {
    int *d_nRT0, maxThreads = 256;
    double *h_out, *d_RT0, *d_out, sum = 0;

    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread;
    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    h_out = (double *)malloc(sizeof(double) * (nBlk/2));

    CHECK(cudaMalloc((void**) &d_out,   (nBlk/2) * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_RT0,  *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,     1 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_RT0,  RT0,       *nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nRT0, nRT0, 1 *sizeof(unsigned int), cudaMemcpyHostToDevice));
    sum_kernel<<<nBlk/2, nThread>>>(d_RT0, d_out);
    CHECK(cudaMemcpy(h_out, d_out, (nBlk/2) * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i=0; i<(nBlk/2); i++) { sum += h_out[i]; }

    *out = sum;
    free(h_out);
    cudaFree(d_RT0); cudaFree(d_nRT0); cudaFree(d_out);
}

void sqsum_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {
    int maxThreads = 256;  
    unsigned int *d_nRT0, *h_nRT0;
    double *d_RT0;
    double *d_pssum, *h_pssum, sum2 = 0;

    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread;

    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    // Stage 1 ----------------------------------------------------
    h_pssum = (double *)malloc(sizeof(double) * nBlk);
    h_nRT0  = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    *h_nRT0 = (unsigned int)*nRT0;

    CHECK(cudaMalloc((void**) &d_RT0,  *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,     1 * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**) &d_pssum, nBlk * sizeof(double)));
    CHECK(cudaMemcpy(d_RT0,  RT0, *h_nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nRT0, h_nRT0, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    squareSum_kernel<<<nBlk, nThread>>>(d_nRT0, d_RT0, d_pssum);
    CHECK(cudaMemcpy(h_pssum, d_pssum, nBlk * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i=0; i<nBlk; i++) { sum2 += h_pssum[i]; }
    *out = sum2 ;
  
    free(h_nRT0); free(h_pssum);
    cudaFree(d_RT0);  cudaFree(d_nRT0); cudaFree(d_pssum); 
}




void sd_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {
    int maxThreads = 256;  
    unsigned int *d_nRT0, *h_nRT0;
    double *d_RT0, *h_psum, *d_psum, *h_mean, *d_mean, sum1 = 0;
    double *d_pssum, *h_pssum, sum2 = 0;

    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread;

    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    h_psum  = (double *)malloc(sizeof(double) * (nBlk/2));
    h_mean  = (double *)malloc(sizeof(double) * 1);
    h_nRT0  = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    *h_nRT0 = (unsigned int)*nRT0;
    
    // Stage 1 ----------------------------------------------------
    CHECK(cudaMalloc((void**) &d_psum, (nBlk/2) * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_RT0,  *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,     1 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_RT0,  RT0, *h_nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    sum_kernel<<<nBlk/2, nThread>>>(d_RT0, d_psum);
    CHECK(cudaMemcpy(h_psum, d_psum, (nBlk/2) * sizeof(double), cudaMemcpyDeviceToHost));
        
    for (int i=0; i<(nBlk/2); i++) { sum1 += h_psum[i]; }
    *h_mean = sum1 / (*h_nRT0);

    // Stage 2 ----------------------------------------------------
    h_pssum = (double *)malloc(sizeof(double) * nBlk);

    CHECK(cudaMemcpy(d_nRT0, h_nRT0, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**) &d_pssum, nBlk * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_mean,     1 * sizeof(double)));

    CHECK(cudaMemcpy(d_mean,  h_mean,      1 * sizeof(double), cudaMemcpyHostToDevice));
    squareSum_kernel<<<nBlk, nThread>>>(d_nRT0, d_RT0, d_pssum);
    CHECK(cudaMemcpy(h_pssum, d_pssum, nBlk * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i=0; i<nBlk; i++) { sum2 += h_pssum[i]; }
    *out = std::sqrt( (sum2 - (sum1*sum1) / *h_nRT0) / (*h_nRT0 - 1) );

    free(h_psum); free(h_mean); free(h_nRT0); free(h_pssum); 
    cudaFree(d_psum);  cudaFree(d_RT0);  cudaFree(d_nRT0);
    cudaFree(d_pssum); cudaFree(d_mean);
}


void min_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {
    int *d_nRT0, maxThreads = 256;
    double *h_out, *d_RT0, *d_out;

    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread / 2;
    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    h_out = (double *)malloc(sizeof(double) * nBlk);
    CHECK(cudaMalloc((void**) &d_out,   nBlk * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_RT0,  *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,     1 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_RT0,  RT0,       *nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nRT0, nRT0, 1 *sizeof(unsigned int), cudaMemcpyHostToDevice));
    min_kernel<<<nBlk, nThread>>>(d_RT0, d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBlk * sizeof(double), cudaMemcpyDeviceToHost));

    arma::vec min_tmp(nBlk);
    for (int i=0; i<nBlk; i++) { min_tmp[i] = h_out[i]; }
    out[0] = min_tmp.min();
    free(h_out);
    cudaFree(d_RT0); cudaFree(d_nRT0); cudaFree(d_out);
}


void max_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {
    int *d_nRT0, maxThreads = 256;
    double *h_out, *d_RT0, *d_out;
    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread / 2;
    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    h_out = (double *)malloc(sizeof(double) * nBlk);
    CHECK(cudaMalloc((void**) &d_out,   nBlk * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_RT0,  *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,     1 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_RT0,  RT0,       *nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nRT0, nRT0, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    max_kernel<<<nBlk, nThread>>>(d_RT0, d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBlk * sizeof(double), cudaMemcpyDeviceToHost));

    arma::vec max_tmp(nBlk);
    for (int i=0; i<nBlk; i++) { max_tmp[i] = h_out[i]; }
    out[0] = max_tmp.max();
    free(h_out);
    cudaFree(d_RT0); cudaFree(d_nRT0); cudaFree(d_out);
}

void minmax_entry(double *RT0, int *nRT0, int *nth, bool *debug, double *out) {

    int *d_nRT0, maxThreads = 256;
    double *h_out, *d_RT0, *d_out;

    int nThread = (*nRT0 < maxThreads) ? nextPow2(*nRT0) : maxThreads;
    int nBlk    = ((*nRT0)+nThread - 1) / nThread / 2;
    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nRT0, nBlk, nThread);

    h_out = (double *)malloc(2*nBlk*sizeof(double));
    CHECK(cudaMalloc((void**) &d_out, 2*nBlk*sizeof(double)));
    CHECK(cudaMalloc((void**) &d_RT0,    *nRT0 * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_nRT0,       1 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_RT0,  RT0,        *nRT0 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nRT0, nRT0, 1 *sizeof(unsigned int), cudaMemcpyHostToDevice));
    minmax_kernel<<<nBlk, nThread>>>(d_RT0, d_out);
    CHECK(cudaMemcpy(h_out, d_out, 2*nBlk*sizeof(double), cudaMemcpyDeviceToHost));

    arma::vec min_tmp(nBlk);
    arma::vec max_tmp(nBlk);

    for (int i=0; i<nBlk; i++) { min_tmp[i] = h_out[i]; }
    for (int i=0; i<nBlk; i++) { max_tmp[i] = h_out[i + nBlk]; }
    out[0] = min_tmp.min();
    out[1] = max_tmp.max();

    free(h_out);
    cudaFree(d_RT0); cudaFree(d_nRT0); cudaFree(d_out);
}

void count_entry(int *n, int *R, int *nth, bool *debug, double *out) {
    unsigned int *h_out, *d_out, *h_n, *d_n, *h_R, *d_R, maxThreads = 256;
    h_n  = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    *h_n = (unsigned int)*n;

    h_R  = (unsigned int *)malloc(sizeof(unsigned int) * (*h_n));
    for(unsigned int i=0; i<*h_n; i++) { h_R[i]  = (unsigned int)R[i]; }

    unsigned int nThread = (*h_n < maxThreads) ? nextPow2(*h_n) : maxThreads;
    unsigned int nBlk    = ((*h_n)+nThread - 1) / nThread;
    h_out = (unsigned int *)malloc(sizeof(unsigned int) * nBlk);
    
    for(unsigned int i=0; i<nBlk; i++) { h_out[i] = 0; }

    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *n, nBlk, nThread);

    CHECK(cudaMalloc((void**) &d_out, nBlk * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**) &d_R,   *h_n * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**) &d_n,    1   * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_R,  h_R,       *h_n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_n,  h_n,          1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out,h_out,     nBlk * sizeof(unsigned int), cudaMemcpyHostToDevice));

    count_kernel<<<nBlk, nThread>>>(d_n, d_R, d_out);
    CHECK(cudaMemcpy(h_out, d_out, nBlk * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /*
    printf("\nOutside kernel\n");
    for (unsigned int i=0; i<nBlk; i++) {
        printf("%d\n", h_out[i]);
    }
    */

    out[0] = h_out[0];
    out[1] = h_out[1];

    free(h_out); free(h_R); free(h_n);
    cudaFree(d_R); cudaFree(d_n); cudaFree(d_out);
}
