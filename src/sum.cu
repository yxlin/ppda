//#include "../inst/include/common.h"
#include "../inst/include/util.h"
#include <R.h>  // R Rprintf

extern "C" void sumur_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void sqsumur_entry(double *x, int *nx, bool *debug, double *out);
extern "C" void sqsumurd_entry(double *x, int *nx, bool *debug, double *out);

template<class T> struct SharedMemory
{
  __device__ inline operator       T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
  
  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template<> struct SharedMemory<double>
{
  __device__ inline operator       double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
  
  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};


template <class T, unsigned int blockSize>
  __global__ void sumur_kernel(unsigned int *n, T *input, T *out) {
    T *cache = SharedMemory<T>();
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    cache[tid] = 0;
    while (i < *n) { cache[tid] = input[i] + input[i + blockSize]; i += gridSize; }
    __syncthreads();
    
    if ((blockSize >= 512) && (tid < 256)) cache[tid] += cache[tid + 256]; __syncthreads();
    if ((blockSize >= 256) && (tid < 128)) cache[tid] += cache[tid + 128]; __syncthreads();
    if ((blockSize >= 128) && (tid < 64))  cache[tid] += cache[tid + 64];  __syncthreads();
    #if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ) {
      if (blockSize >= 64) cache[tid] += cache[tid + 32];
      for (int offset = warpSize/2; offset > 0; offset /= 2) 
        cache[tid] += __shfl_down(cache[tid], offset);
    }
    #else
    if ((blockSize >= 64) && (tid < 32)) cache[tid] += cache[tid + 32]; __syncthreads();
    if ((blockSize >= 32) && (tid < 16)) cache[tid] += cache[tid + 16]; __syncthreads();
    if ((blockSize >= 16) && (tid <  8)) cache[tid] += cache[tid +  8]; __syncthreads();
    if ((blockSize >=  8) && (tid <  4)) cache[tid] += cache[tid +  4]; __syncthreads();
    if ((blockSize >=  4) && (tid <  2)) cache[tid] += cache[tid +  2]; __syncthreads();
    if ((blockSize >=  2) && (tid <  1)) cache[tid] += cache[tid +  1]; __syncthreads();
    #endif
    if(tid==0) out[blockIdx.x] = cache[0];
  }

template <class T, unsigned int blockSize>
  __global__ void sqsumur_kernel(unsigned int* n, T *in, T *out) {
    T *cache = SharedMemory<T>();
    unsigned int tid      = threadIdx.x;
    unsigned int threadID = blockIdx.x * blockSize + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float sqx = 0;
    for (size_t i=threadID; i<*n; i += gridSize) { sqx = in[i] * in[i]; }
    cache[tid] = sqx;
    __syncthreads();
    
    if ((blockSize >= 512) && (tid < 256)) cache[tid] += cache[tid + 256]; __syncthreads();
    if ((blockSize >= 256) && (tid < 128)) cache[tid] += cache[tid + 128]; __syncthreads();
    if ((blockSize >= 128) && (tid < 64))  cache[tid] += cache[tid + 64];  __syncthreads();
    #if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ) {
      if (blockSize >= 64) cache[tid] += cache[tid + 32];
      for (int offset = warpSize/2; offset > 0; offset /= 2) 
        cache[tid] += __shfl_down(cache[tid], offset);
    }
    #else
    if ((blockSize >= 64) && (tid < 32)) cache[tid] += cache[tid + 32]; __syncthreads();
    if ((blockSize >= 32) && (tid < 16)) cache[tid] += cache[tid + 16]; __syncthreads();
    if ((blockSize >= 16) && (tid <  8)) cache[tid] += cache[tid +  8]; __syncthreads();
    if ((blockSize >=  8) && (tid <  4)) cache[tid] += cache[tid +  4]; __syncthreads();
    if ((blockSize >=  4) && (tid <  2)) cache[tid] += cache[tid +  2]; __syncthreads();
    if ((blockSize >=  2) && (tid <  1)) cache[tid] += cache[tid +  1]; __syncthreads();
    #endif
    
    if(tid==0) out[blockIdx.x] = cache[0];
  }

void sumur_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread   = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk      = ((*nx) + nThread - 1) / nThread / 2;
  int smemSize = (nThread <= 32) ? 2 * nThread * sizeof(double) : nThread * sizeof(double);
  
  size_t blockSize = nBlk * sizeof(double);
  size_t nxSize    = *nx * sizeof(double);
  size_t uSize     = 1*sizeof(unsigned int);

  unsigned int *h_nx, *d_nx;
  double *h_x, *d_x, *h_out, *d_out;
  
  h_nx  = (unsigned int *)malloc(uSize);
  h_out = (double *)malloc(blockSize);
  //h_x   = (double *)malloc(nxSize);
  *h_nx = (unsigned int)*nx;
  //for(size_t i=0; i<*nx; i++) { h_x[i] = (double)x[i]; }
  
  cudaMalloc((void**) &d_out,  blockSize);
  cudaMalloc((void**) &d_x,  nxSize);
  cudaMalloc((void**) &d_nx, uSize);
  cudaMemcpy(d_x,  x,      nxSize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nx, h_nx,     uSize, cudaMemcpyHostToDevice);
  sumur_kernel<double, 256><<< 256, nThread, smemSize >>>(d_nx, d_x, d_out);
  cudaMemcpy(h_out, d_out, blockSize,   cudaMemcpyDeviceToHost);
  
  for (unsigned int i=0; i<nBlk; i++) { out[0] += h_out[i]; }
  cudaFree(d_x); cudaFree(d_nx); cudaFree(d_out);
  free(h_nx);     free(h_out);  free(h_x);
}

void sqsumur_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx) + nThread - 1) / nThread;
  int smemSize = (nThread <= 32) ? 2 * nThread * sizeof(float) : nThread * sizeof(float);
  
  size_t blockSize = nBlk * sizeof(float);
  size_t nxSize    = *nx * sizeof(float);
  size_t uSize     = 1*sizeof(unsigned int);
  if (*debug) Rprintf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);
  
  unsigned int *h_nx, *d_nx;
  float *h_x, *d_x, *h_pssum, *d_pssum;
  
  h_nx    = (unsigned int *)malloc(uSize);
  h_pssum = (float *)malloc(blockSize);
  h_x     = (float *)malloc(nxSize);
  *h_nx   = (unsigned int)*nx;
  for(size_t i=0; i<*nx; i++) { h_x[i] = (float)x[i]; }
  
  cudaMalloc((void**) &d_x,     nxSize);
  cudaMalloc((void**) &d_nx,    uSize);
  cudaMalloc((void**) &d_pssum, blockSize);
  cudaMemcpy(d_x,  h_x, nxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_nx, h_nx, uSize, cudaMemcpyHostToDevice);
  
  sqsumur_kernel<float, 256><<< nBlk, nThread, smemSize >>>(d_nx, d_x, d_pssum);
  
  cudaMemcpy(h_pssum, d_pssum, blockSize, cudaMemcpyDeviceToHost);
  for (int i=0; i<nBlk; i++) { out[0] += (double)h_pssum[i]; }
  free(h_nx); free(h_pssum); free(h_x);
  cudaFree(d_x);  cudaFree(d_nx); cudaFree(d_pssum); 
}

void sqsumurd_entry(double *x, int *nx, bool *debug, double *out) {
  unsigned int maxThreads = 256;
  unsigned int nThread = (*nx < maxThreads) ? nextPow2(*nx) : maxThreads;
  unsigned int nBlk    = ((*nx) + nThread - 1) / nThread;
  
  int smemSize = (nThread <= 32) ? 2 * nThread * sizeof(double) : nThread * sizeof(double);
  
  size_t blockSize = nBlk * sizeof(double);
  size_t nxSize    = *nx * sizeof(double);
  size_t uSize     = 1*sizeof(unsigned int);
  if (*debug) Rprintf("ndata, nblock & nthread: %d %d %d\n", *nx, nBlk, nThread);
  
  unsigned int *h_nx, *d_nx;
  double *h_x, *d_x, *h_pssum, *d_pssum;
  
  h_nx    = (unsigned int *)malloc(uSize);
  h_pssum = (double *)malloc(blockSize);
  h_x     = (double *)malloc(nxSize);
  *h_nx   = (unsigned int)*nx;
  
  cudaMalloc((void**) &d_x,     nxSize);
  cudaMalloc((void**) &d_nx,    uSize);
  cudaMalloc((void**) &d_pssum, blockSize);
  cudaMemcpy(d_x,  x, nxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_nx, h_nx, uSize, cudaMemcpyHostToDevice);
  
  sqsumur_kernel<double, 256><<< nBlk, nThread, smemSize >>>(d_nx, d_x, d_pssum);
  
  cudaMemcpy(h_pssum, d_pssum, blockSize, cudaMemcpyDeviceToHost);
  for (int i=0; i<nBlk; i++) { out[0] += h_pssum[i]; }
  
  free(h_nx); free(h_pssum); free(h_x);
  cudaFree(d_x);  cudaFree(d_nx); cudaFree(d_pssum); 
}
