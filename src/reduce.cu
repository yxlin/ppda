#include <unistd.h>
#include <stdio.h>  // C printing
#include "../inst/include/common.h"
#include "../inst/include/constant.h"
#include "../inst/include/util.h"
#include <armadillo> // Armadillo vector operations

extern "C" void sum_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void min_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void max_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void minmax_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void sqsum_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void sd_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void count_entry(int *nR, int *R, bool *debug, double *out);
extern "C" void n1min_entry(double *RT0, int *nx, bool *debug, double *out);

__global__ void min_kernel(double *input, double *out) {
  __shared__ double cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid]    = fmin(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmin(cache[tid], cache[tid + s]); }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void minunroll_kernel(double *input, double *out) {
  __shared__ double cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid]    = fmin(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>32; s>>=1) {
      if(tid < s) { cache[tid] = fmin(cache[tid], cache[tid + s]); }
      __syncthreads();
  }
  if (tid < 32) {
          cache[tid] = fmin(cache[tid], cache[tid + 32]);
          cache[tid] = fmin(cache[tid], cache[tid + 16]);
          cache[tid] = fmin(cache[tid], cache[tid + 8]);
          cache[tid] = fmin(cache[tid], cache[tid + 4]);
          cache[tid] = fmin(cache[tid], cache[tid + 2]);
          cache[tid] = fmin(cache[tid], cache[tid + 1]);
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void min_kernel(float *input, float *out) {
  __shared__ float cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid] = fmin(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmin(cache[tid], cache[tid + s]); }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void max_kernel(double *input, double *out) {
  __shared__ double cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid] = fmax(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmax(cache[tid], cache[tid + s]); }
       __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void max_kernel(float *input, float *out) {
  __shared__ float cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid] = fmax(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmax(cache[tid], cache[tid + s]); }
       __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void minmax_kernel(double *input, double *out) {
  __shared__ double cache1[256];
  __shared__ double cache2[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache1[tid] = fmin(input[i], input[i + blockDim.x]);
  cache2[tid] = fmax(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
    if(tid < s) {
      cache1[tid] = fmin(cache1[tid], cache1[tid + s]);
      cache2[tid] = fmax(cache2[tid], cache2[tid + s]);
    }
    __syncthreads();
  }
  if(tid==0) {
    out[blockIdx.x]             = cache1[0];
    out[blockIdx.x + gridDim.x] = cache2[0];
  }
}

__global__ void minmax_kernel(float *input, float *out) {
  __shared__ float cache1[256];
  __shared__ float cache2[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache1[tid] = fmin(input[i], input[i + blockDim.x]);
  cache2[tid] = fmax(input[i], input[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
    if(tid < s) {
      cache1[tid] = fmin(cache1[tid], cache1[tid + s]);
      cache2[tid] = fmax(cache2[tid], cache2[tid + s]);
    }
    __syncthreads();
  }
  if(tid==0) {
    out[blockIdx.x]             = cache1[0];
    out[blockIdx.x + gridDim.x] = cache2[0];
  }
}

__global__ void sum_kernel(double *input, double *out) {
  __shared__ double cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid] = input[i] + input[i + blockDim.x];
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] += cache[tid + s]; }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}



__global__ void sum_kernel(float *input, float *out) {
  __shared__ float cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  cache[tid] = input[i] + input[i + blockDim.x];
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] += cache[tid + s]; }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void squareSum_kernel(unsigned int* n, double *input, double *out) {
  __shared__ double cache[256];
  unsigned int numThreads = blockDim.x * gridDim.x;
  unsigned int tid        = threadIdx.x;
  unsigned int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  double tmp = 0;
  
  for (size_t i=threadID; i<(*n); i += numThreads) {
    tmp = input[i]  * input[i];
  }
  cache[tid] = tmp;
  __syncthreads();
  
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] += cache[tid+s]; }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void squareSum_kernel(unsigned int* n, float *input, float *out) {
  __shared__ float cache[256];
  unsigned int numThreads = blockDim.x * gridDim.x;
  unsigned int tid        = threadIdx.x;
  unsigned int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0;
  
  for (size_t i=threadID; i<(*n); i += numThreads) {
    tmp = input[i]  * input[i];
  }
  cache[tid] = tmp;
  __syncthreads();
  
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] += cache[tid+s]; }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void n1min_kernel(float *RT0, float *out) {
  __shared__ float cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  if (RT0[i] == 0) { RT0[i] = CUDART_INF_F; }
  if (RT0[i + blockDim.x] == 0) { RT0[i + blockDim.x] = CUDART_INF_F; }

  cache[tid] = fmin(RT0[i],RT0[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmin(cache[tid], cache[tid + s]); }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}

__global__ void n1max_kernel(float *RT0, float *out) {
  __shared__ float cache[256];
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  if (!isfinite(RT0[i])) { RT0[i] = 0; }
  if (!isfinite(RT0[i + blockDim.x])) { RT0[i + blockDim.x] = 0; }

  cache[tid] = fmax(RT0[i],RT0[i + blockDim.x]);
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
      if(tid < s) { cache[tid] = fmax(cache[tid], cache[tid + s]); }
      __syncthreads();
  }
  if(tid==0) out[blockIdx.x] = cache[0];
}


__global__ void count_kernel(unsigned int *n, unsigned int *R, unsigned int *out) {
  __shared__ unsigned int cache[256];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  unsigned int numThreads = blockDim.x * gridDim.x; // total # of threads
  unsigned int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int simR;
  
  for (size_t i = threadID; i<(*n); i += numThreads) {
    simR = R[i];
    
    if (simR == 1) {
      atomicAdd(&(cache[0]), 1);
    } else if (simR == 2) {
      atomicAdd(&(cache[1]), 1);
    } else {
      atomicAdd(&(cache[2]), 1);
    }
  }
  __syncthreads();
  // add partial results in each block together.
  atomicAdd( &(out[threadIdx.x]), cache[threadIdx.x] );
}

void sum_entry(double *x, int *nx, bool *debug, double *out) {
    unsigned int maxThreads = 256;
    unsigned int nThread   = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
    unsigned int nBlk      = ((*nx) + nThread - 1) / nThread / 2;
    size_t blockSize       = nBlk * sizeof(double);
    size_t nxSize          = *nx * sizeof(double);
    size_t unsignedintSize = 1*sizeof(unsigned int);
    if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

    unsigned int *h_nx, *d_nx;
    double *d_x, *h_out, *d_out;

    h_nx  = (unsigned int *)malloc(unsignedintSize);
    h_out = (double *)malloc(blockSize);
    *h_nx = (unsigned int)*nx;
  
    CHECK(cudaMalloc((void**) &d_out,  blockSize));
    CHECK(cudaMalloc((void**) &d_x,  nxSize));
    CHECK(cudaMalloc((void**) &d_nx, unsignedintSize));
    CHECK(cudaMemcpy(d_x,  x,      nxSize,          cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nx, h_nx,   unsignedintSize, cudaMemcpyHostToDevice));
    sum_kernel<<<nBlk, nThread>>>(d_x, d_out);
    CHECK(cudaMemcpy(h_out, d_out, blockSize,   cudaMemcpyDeviceToHost));

    for (unsigned int i=0; i<nBlk; i++) { out[0] += h_out[i]; }

    cudaFree(d_x); cudaFree(d_nx); cudaFree(d_out);
    free(h_nx);     free(h_out); 
}


void sqsum_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx) + nThread - 1) / nThread;

  size_t blockSize       = nBlk * sizeof(double);
  size_t nxSize          = *nx * sizeof(double);
  size_t unsignedintSize = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *h_nx, *d_nx;
  double *d_x, *h_pssum, *d_pssum;

  h_pssum = (double *)malloc(blockSize);
  h_nx    = (unsigned int *)malloc(unsignedintSize);
  *h_nx   = (unsigned int)*nx;
  
  CHECK(cudaMalloc((void**) &d_x,     nxSize));
  CHECK(cudaMalloc((void**) &d_nx,    unsignedintSize));
  CHECK(cudaMalloc((void**) &d_pssum, blockSize));
  CHECK(cudaMemcpy(d_x,  x, nxSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx, h_nx, unsignedintSize, cudaMemcpyHostToDevice));
  
  squareSum_kernel<<<nBlk, nThread>>>(d_nx, d_x, d_pssum);
  CHECK(cudaMemcpy(h_pssum, d_pssum, blockSize, cudaMemcpyDeviceToHost));
  for (int i=0; i<nBlk; i++) { out[0] += h_pssum[i]; }
  
  free(h_nx); free(h_pssum); 
  cudaFree(d_x);  cudaFree(d_nx); cudaFree(d_pssum); 
}

void sd_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = (*nx + nThread - 1) / nThread;
  size_t blockSize       = nBlk * sizeof(double);
  size_t halfBlockSize   = 0.5 * nBlk * sizeof(double);
  size_t nxSize          = *nx * sizeof(double);
  size_t unsignedintSize = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *d_nx, *h_nx;
  double *d_x, *h_psum, *d_psum, *d_pssum, *h_pssum;
  double sum1 = 0, sum2 = 0;
    
  h_psum  = (double *)malloc(halfBlockSize);
  h_nx    = (unsigned int *)malloc(unsignedintSize);
  *h_nx   = (unsigned int)*nx;
  h_pssum = (double *)malloc(blockSize);
  
  CHECK(cudaMalloc((void**) &d_psum, halfBlockSize));
  CHECK(cudaMalloc((void**) &d_x,  nxSize));
  CHECK(cudaMalloc((void**) &d_nx, unsignedintSize));
  CHECK(cudaMalloc((void**) &d_pssum, blockSize));

  CHECK(cudaMemcpy(d_x,  x,    nxSize,          cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx, h_nx, unsignedintSize, cudaMemcpyHostToDevice));

  sum_kernel<<<nBlk/2, nThread>>>(d_x, d_psum);
  squareSum_kernel<<<nBlk, nThread>>>(d_nx, d_x, d_pssum);

  CHECK(cudaMemcpy(h_psum,  d_psum,  halfBlockSize, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_pssum, d_pssum, blockSize,     cudaMemcpyDeviceToHost));
  
  for (int i=0; i<(nBlk/2); i++) { sum1 += h_psum[i]; }
  for (int i=0; i<nBlk; i++) { sum2 += h_pssum[i]; }
  *out = std::sqrt( (sum2 - (sum1*sum1) / *h_nx) / (*h_nx - 1) );
  
  free(h_psum); free(h_nx); free(h_pssum); 
  cudaFree(d_psum);  cudaFree(d_x);  cudaFree(d_nx);
  cudaFree(d_pssum);
}

void min_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx)+nThread - 1) / nThread / 2;
  size_t blockSize       = nBlk * sizeof(double);
  size_t nxSize          = *nx * sizeof(double);
  size_t unsignedintSize = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *h_nx, *d_nx;
  double *d_x, *d_out, *h_out;

  h_nx  = (unsigned int *)malloc(unsignedintSize);
  h_out = (double *)malloc(blockSize);
  *h_nx = (unsigned int)*nx;
  
  CHECK(cudaMalloc((void**) &d_out, blockSize));
  CHECK(cudaMalloc((void**) &d_x,  nxSize));
  CHECK(cudaMalloc((void**) &d_nx, unsignedintSize));
  CHECK(cudaMemcpy(d_x,  x,      nxSize,          cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx, h_nx,   unsignedintSize, cudaMemcpyHostToDevice));
  min_kernel<<<nBlk, nThread>>>(d_x, d_out);
  CHECK(cudaMemcpy(h_out, d_out, blockSize,       cudaMemcpyDeviceToHost));
  for(int i=0; i<nBlk; i++) { printf("h_out[%d]: %f\n", i, h_out[i]); }

  arma::vec min_tmp(nBlk);
  for (int i=0; i<nBlk; i++) { min_tmp[i] = h_out[i]; }
  out[0] = min_tmp.min();
  free(h_nx); free(h_out); 
  cudaFree(d_x); cudaFree(d_nx); cudaFree(d_out);
}

void max_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx)+nThread - 1) / nThread / 2;
  size_t blockSize       = nBlk * sizeof(double);
  size_t nxSize          = *nx * sizeof(double);
  size_t unsignedintSize = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *h_nx, *d_nx;
  double *d_x, *d_out, *h_out;

  h_nx  = (unsigned int *)malloc(unsignedintSize);
  h_out = (double *)malloc(blockSize);
  *h_nx = (unsigned int)*nx;
  
  CHECK(cudaMalloc((void**) &d_out, blockSize));
  CHECK(cudaMalloc((void**) &d_x,  nxSize));
  CHECK(cudaMalloc((void**) &d_nx, unsignedintSize));
  CHECK(cudaMemcpy(d_x,  x,      nxSize,          cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx, h_nx,   unsignedintSize, cudaMemcpyHostToDevice));
  max_kernel<<<nBlk, nThread>>>(d_x, d_out);
  CHECK(cudaMemcpy(h_out, d_out, blockSize,       cudaMemcpyDeviceToHost));

  arma::vec max_tmp(nBlk);
  for (int i=0; i<nBlk; i++) { max_tmp[i] = h_out[i]; }
  out[0] = max_tmp.max();
  free(h_nx); free(h_out); 
  cudaFree(d_x); cudaFree(d_nx); cudaFree(d_out);
}

void minmax_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx)+nThread - 1) / nThread / 2;
  size_t blockSize       = nBlk * sizeof(double);
  size_t nxSize          = *nx * sizeof(double);
  size_t unsignedintSize = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *h_nx, *d_nx;
  double *d_x, *d_out, *h_out;

  h_nx  = (unsigned int *)malloc(unsignedintSize);
  h_out = (double *)malloc(2*blockSize);
  *h_nx = (unsigned int)*nx;

  CHECK(cudaMalloc((void**) &d_out, 2*blockSize));
  CHECK(cudaMalloc((void**) &d_x,   nxSize));
  CHECK(cudaMalloc((void**) &d_nx,  unsignedintSize));
  CHECK(cudaMemcpy(d_x,  x,  nxSize,          cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx, nx, unsignedintSize, cudaMemcpyHostToDevice));
  minmax_kernel<<<nBlk, nThread>>>(d_x, d_out);
  CHECK(cudaMemcpy(h_out, d_out, 2*blockSize, cudaMemcpyDeviceToHost));
  
  arma::vec min_tmp(nBlk);
  arma::vec max_tmp(nBlk);
  
  for (int i=0; i<nBlk; i++) {
      min_tmp[i] = h_out[i];
      max_tmp[i] = h_out[i + nBlk];
  }
  out[0] = min_tmp.min();
  out[1] = max_tmp.max();
  
  free(h_out);
  cudaFree(d_x); cudaFree(d_nx); cudaFree(d_out);
}

void count_entry(int *nR, int *R, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nR < maxThreads) ? nextPow2(*nR) : maxThreads;
  unsigned int nBlk    = ((*nR) + nThread - 1) / nThread;
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nR, nBlk, nThread);
  size_t unsignedintSize = 1*sizeof(unsigned int);

  unsigned int *h_out, *d_out, *h_n, *d_n, *h_R, *d_R;
  h_n  = (unsigned int *)malloc(unsignedintSize);
  *h_n = (unsigned int)*nR;
  h_R  = (unsigned int *)malloc(unsignedintSize * (*h_n));
  for(int i=0; i<*nR; i++) { h_R[i] = (unsigned int)R[i]; }
  h_out = (unsigned int *)malloc(unsignedintSize * nBlk);
  for(int i=0; i<nBlk; i++) { h_out[i] = 0; }
  
  CHECK(cudaMalloc((void**) &d_out, nBlk * unsignedintSize));
  CHECK(cudaMalloc((void**) &d_R,   *h_n * unsignedintSize));
  CHECK(cudaMalloc((void**) &d_n,    unsignedintSize));
  CHECK(cudaMemcpy(d_R,  h_R,       *h_n * unsignedintSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_n,  h_n,          unsignedintSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_out,h_out,     nBlk * unsignedintSize, cudaMemcpyHostToDevice));
  
  count_kernel<<<nBlk, nThread>>>(d_n, d_R, d_out);
  CHECK(cudaMemcpy(h_out, d_out, nBlk * unsignedintSize, cudaMemcpyDeviceToHost));
  
  out[0] = h_out[0];
  out[1] = h_out[1];
  
  free(h_out); free(h_R); free(h_n);
  cudaFree(d_R); cudaFree(d_n); cudaFree(d_out);
}

void n1min_entry(double *RT0, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx) + nThread - 1) / nThread / 2;
  size_t blockSize     = nBlk * sizeof(float);
  size_t nfSize        = *nx * sizeof(float);
  size_t uSize         = 1*sizeof(unsigned int);
  if (*debug) printf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);

  unsigned int *h_nx, *d_nx;
  float *h_RT0, *d_RT0, *d_out, *h_out;

  h_nx  = (unsigned int *)malloc(uSize);
  h_out = (float *)malloc(blockSize);
  *h_nx = (unsigned int)*nx;
  
  CHECK(cudaHostAlloc((void**)&h_RT0, nfSize, cudaHostAllocDefault));
  for(int i=0; i<*nx; i++) { h_RT0[i] = (float)RT0[i]; }

  CHECK(cudaMalloc((void**) &d_RT0,  nfSize));
  CHECK(cudaMalloc((void**) &d_nx,   uSize));
  CHECK(cudaMalloc((void**) &d_out,  blockSize));
  CHECK(cudaMemcpy(d_RT0, h_RT0, nfSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nx,  h_nx,  uSize,  cudaMemcpyHostToDevice));

  n1min_kernel<<<nBlk, nThread>>>(d_RT0, d_out);
  CHECK(cudaMemcpy(h_out, d_out, blockSize, cudaMemcpyDeviceToHost));

  arma::vec min_tmp(nBlk);
  for (int i=0; i<nBlk; i++) { min_tmp[i] = (double)h_out[i]; }
  out[0] = min_tmp.min();
  
  cudaFreeHost(h_RT0); free(h_nx); free(h_out); 
  cudaFree(d_RT0); cudaFree(d_nx); cudaFree(d_out);
}
