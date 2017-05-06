#include <unistd.h>
#include <stdio.h>  // C printing
#include <curand_kernel.h> // Device random API
#include "../inst/include/common.h"  
#include "../inst/include/constant.h"  
#include "../inst/include/reduce.h"
#include "../inst/include/random.h"  
#include <armadillo> 

extern "C" void n1PDF(double *x, int *nx, int *nsim, double *b, double *A,
  double *mean_v, int *nmean_v, double *sd_v, double *t0, int *nth, bool *debug,
  double *out);

extern "C" void n1PDF_plba1(double *x, int *nx, int *nsim, double *b, double *A,
                           double *mean_v, int *nmean_v, double *sd_v,
                           double *t0, double *mean_w, double *rD, double *swt,
                           int *nth, bool *debug, double *out);
                           //int *nth, bool *debug, double *RT, int *R, double *out);

extern "C" void n1PDF_plba2(double *x, int *nx, int *nsim, double *b, double *A,
                            double *mean_v, int *nmean_v, double *sd_v,double *sd_w,
                           double *t0, double *mean_w, double *rD, double *swt,
                            //int *nth, bool *debug, double *out);
                           int *nth, bool *debug, double *RT, int *R, double *out);

extern "C" void n1PDF_plba3(double *x, int *nx, int *nsim, double *B, double *A,
                            double *C, double *mean_v, int *nmean_v, double *sd_v,
                            double *sd_w, double *t0, double *mean_w, double *rD,
                            double *tD, double *swt,
                            //int *nth, bool *debug, double *out);
                            int *nth, bool *debug, double *RT, int *R, double *out);

extern "C" void histc_entry(double *binedge, double *rng, int nrng, int ngrid, 
  unsigned int *out);


__global__ void histc_kernel(double *binedge, double *rng, int *nrng,
  unsigned int *out) {
  __shared__ unsigned int cache[1024];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j=0;
  double tmp = 0;
  //double sim = rng[i];
  
  if (rng[i] < binedge[0] || rng[i] > binedge[1024]) {
    // if out-of-range add 0 to the 1st bin, otherwise add 1 to j bin
    atomicAdd(&(cache[0]), 0);
  } else {
    // When 'sim' belongs to 'j' bin, the last line, 'j++' inside 
    // while loop will add one more '1' to j, before leaving the loop.
    // So I use cache[j-1].
    while(tmp==0) {
      tmp = ((rng[i] >= binedge[j]) && (rng[i] < binedge[j+1])); // 0 or 1;
      j++;
    }
    atomicAdd( &(cache[j-1]), 1);
  }
  __syncthreads();
  // add partial histograms in each block together.
  atomicAdd( &(out[threadIdx.x]), cache[threadIdx.x] );
}

__global__ void histc_kernel(float *binedge, float *rng, unsigned int *nrng,
  unsigned int *out) {
  __shared__ unsigned int cache[1024];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j=0;
  float tmp = 0;
  
  if (rng[i] < binedge[0] || rng[i] >= binedge[1024]) {
    atomicAdd(&(cache[0]), 0);
  } else {
    while(tmp==0) {
      tmp = ((rng[i] >= binedge[j]) && (rng[i] < binedge[j+1])); // 0 or 1;
      j++;
      // if (j > 1024) {
      //     printf("RT0[%d] is %f\n", i, rng[i]);
      //     printf("%d j reaches 1024\n", j);
      //     break;
      // }
    }
    atomicAdd( &(cache[j-1]), 1);
  }
  __syncthreads();
  atomicAdd( &(out[threadIdx.x]), cache[threadIdx.x] );
}

void histc_entry(double *binedge, double *rng, int nrng, int ngrid, 
  unsigned int *out) {
  int *h_nrng, *d_nrng;
  double *d_binedge, *d_rng, *h_binedge, *h_rng;
  unsigned int *d_hist, *h_hist;
  h_nrng = (int *)malloc(1 * sizeof(int));
  h_nrng[0] = nrng;
  
  CHECK(cudaHostAlloc((void**)&h_hist, ngrid * sizeof(unsigned int), cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void**)&h_rng,  nrng * sizeof(double), cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void**)&h_binedge,  (ngrid+1) * sizeof(double), cudaHostAllocDefault));
  for(int i=0; i<nrng; i++) { h_rng[i] = rng[i]; }
  for(int i=0; i<(ngrid+1); i++) { h_binedge[i] = binedge[i]; }
  
  CHECK(cudaMalloc((void**) &d_nrng,    1 * sizeof(int)));
  CHECK(cudaMalloc((void**) &d_binedge, (ngrid+1) * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_rng,     nrng * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_hist,    ngrid * sizeof(unsigned int)));
  
  CHECK(cudaMemcpy(d_nrng, h_nrng,       1*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_binedge, h_binedge, (ngrid+1)*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_rng, h_rng,         nrng*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_hist, h_hist,       ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice));
  
  histc_kernel<<<(nrng/1024 + 1), 1024>>>(d_binedge, d_rng, d_nrng, d_hist);
  CHECK(cudaMemcpy(h_hist, d_hist, ngrid * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for(int i=0; i<ngrid; i++) { out[i] = h_hist[i]; }
  
  cudaFreeHost(h_hist); cudaFreeHost(h_rng); cudaFreeHost(h_binedge);
  free(h_nrng); 
  CHECK(cudaFree(d_binedge));
  CHECK(cudaFree(d_rng));
  CHECK(cudaFree(d_nrng));
  CHECK(cudaFree(d_hist));
}

void n1PDF(double *x, int *nx, int *nsim, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, double *t0, int *nth, bool *debug, double *out) 
{
  size_t nsimfSize = *nsim * sizeof(float);
  size_t nsimuSize = *nsim * sizeof(unsigned int);
  float *d_RT0; unsigned int *d_R;
  CHECK(cudaMalloc((void**) &d_RT0, nsimfSize));
  CHECK(cudaMalloc((void**) &d_R,   nsimuSize));
  rn1(nsim, b, A, mean_v, nmean_v, sd_v, t0, nth, d_R, d_RT0); // run kernel
  
  // ------------------------------------------------------------------------80
  unsigned int maxThread = 256;
  unsigned int nThread = (*nsim < maxThread) ? nextPow2(*nsim) : maxThread;
  unsigned int nBlk    = ((*nsim) + nThread - 1) / nThread / 2;

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
  n1min_kernel<<<nBlk, nThread>>>(d_RT0, d_n1min_out); 
  n1max_kernel<<<nBlk, nThread>>>(d_RT0, d_n1max_out);
  sum_kernel<<<nBlk, nThread>>>(d_RT0,   d_sum_out);
  squareSum_kernel<<<2*nBlk, nThread>>>(d_nsim, d_RT0, d_sqsum_out);
  
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
  
  free(h_sqsum_out);
  free(h_n1min_out);
  free(h_n1max_out);
  free(h_sum_out);
  float minRT0 = min_tmp.min();
  float maxRT0 = max_tmp.max();
  int nsRT0    = h_count_out[0]; free(h_count_out);
  float sd     = std::sqrt( (sqsum - (sum*sum)/nsRT0) / (nsRT0 - 1) );
  
  if (*debug) printf("RT0 [minimum maximum sum sqsum nsRT0 sd] values: %.2f %.2f %.2f %.2f %d %.2f\n",
    min_tmp.min(), max_tmp.max(), sum, sqsum, nsRT0, sd);
  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
    cudaFree(d_RT0); cudaFree(d_nsim); free(h_nsim);
    for(size_t i=0; i<*nx; i++) { out[i] = 1e-10; }
  } else {
    float h  = 0.09*sd*std::pow((float)*nsim, -0.2);
    float z0 = minRT0 <= 0 ? minRT0 : minRT0 - 3.0*h; if (z0 < 0) z0 = 0;
    float z1 = maxRT0 > 10.0 ? 10.0 : maxRT0 + 3.0*h;
    int ngrid = 1024;
    int half_ngrid  = 0.5*ngrid;
    size_t ngrid_plus1fSize = (ngrid + 1) * sizeof(float);
    size_t ngriduSize = ngrid * sizeof(unsigned int);
    arma::vec z = arma::linspace<arma::vec>((double)z0, (double)z1, ngrid);
    float dt = z[1] - z[0];
    
    arma::vec filter0(ngrid);
    double z1minusz0 = (double)(z1 - z0);
    double fil0_constant = (double)(-2.0*h*h*M_PI*M_PI) / (z1minusz0*z1minusz0);

    float *h_binedge0, *d_binedge0;
    unsigned int *h_hist0, *d_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    for(size_t i=0; i<ngrid; i++) {
      h_binedge0[i] = z0 + dt*((float)i - 0.5); // binedge
      h_hist0[i]    = 0;  // initialize histogram
      if (i < (1 + half_ngrid)) {       // Get filter (1024)
        filter0[i] = std::exp(fil0_constant * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    
    h_binedge0[ngrid] = (z0 + ((float)(ngrid - 1))*dt);
    
    if (*debug) printf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) printf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) printf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    
    // Get histogram (1024)---------------------------------------------------80
    CHECK(cudaMalloc((void**) &d_binedge0, ngrid_plus1fSize)); // 1025
    CHECK(cudaMalloc((void**) &d_hist0, ngriduSize));          // 1024
    CHECK(cudaMemcpy(d_binedge0, h_binedge0, ngrid_plus1fSize, cudaMemcpyHostToDevice)); free(h_binedge0);
    CHECK(cudaMemcpy(d_hist0,    h_hist0,    ngriduSize,       cudaMemcpyHostToDevice));
    
    histc_kernel<<<*nsim/ngrid, ngrid>>>(d_binedge0, d_RT0, d_nsim, d_hist0);
    cudaFree(d_RT0); cudaFree(d_binedge0); cudaFree(d_nsim);
    
    CHECK(cudaMemcpy(h_hist0, d_hist0, ngriduSize, cudaMemcpyDeviceToHost)); cudaFree(d_hist0); 
    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); free(h_nsim); 

    // FFT: Get simulated PDF ---------------------------
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; 
    arma::vec eDen; // a container for estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(size_t i=0; i<*nx; i++) { 
      out[i] = (eDen[i] < 1e-10 || std::isnan(eDen[i])) ? 1e-10 : eDen[i]; 
    }
  }
}


void n1PDF_plba1(double *x, int *nx, int *nsim, double *b, double *A, double *mean_v,
                int *nmean_v, double *sd_v, double *t0, double *mean_w, double *rD,
                double *swt, int *nth, bool *debug, double *out) {
                //double *swt, int *nth, bool *debug, double *RT, int *R, double *out) {
  size_t nsimfSize = *nsim * sizeof(double);
  size_t nsimuSize = *nsim * sizeof(unsigned int);
  float *d_RT; unsigned int *d_R;
  //float *h_RT; unsigned int *h_R;
  //h_RT  = (float *)malloc(nsimfSize);
  //h_R   = (unsigned int *)malloc(nsimuSize);
  CHECK(cudaMalloc((void**) &d_RT, nsimfSize));
  CHECK(cudaMalloc((void**) &d_R,  nsimuSize));

  double *T0;
  T0       = (double *)malloc(sizeof(double) * 1);
  *T0 = *rD + *swt;

  rplba1_n1(nsim, b, A, mean_v, nmean_v, mean_w, sd_v, t0, T0, nth, d_R, d_RT); 
                    
  //CHECK(cudaMemcpy(h_R,  d_R,  nsimuSize, cudaMemcpyDeviceToHost)); 
  //CHECK(cudaMemcpy(h_RT, d_RT, nsimfSize, cudaMemcpyDeviceToHost));
  // for(int i=0; i<*nsim; i++) {
  //     RT[i] = h_RT[i];
  //     R[i]  = h_R[i];
  // }
  //free(h_R); free(h_RT);

  // ------------------------------------------------------------------------80
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

  free(h_sqsum_out); 
  free(h_n1min_out); free(h_n1max_out); free(h_sum_out);

  float minRT0 = min_tmp.min();
  float maxRT0 = max_tmp.max();
  int nsRT0    = h_count_out[0]; free(h_count_out);
  float sd     = std::sqrt( (sqsum - (sum*sum)/nsRT0) / (nsRT0 - 1) );
  
  if (*debug) printf("RT0 [minimum maximum sum sqsum nsRT0 sd] values: %.2f %.2f %.2f %.2f %d %.2f\n",
    min_tmp.min(), max_tmp.max(), sum, sqsum, nsRT0, sd);

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT); cudaFree(d_nsim); free(h_nsim);
    for(size_t i=0; i<*nx; i++) { out[i] = 1e-10; }
  } else {
    float h  = 0.09*sd*std::pow((float)*nsim, -0.2);
    float z0 = minRT0 <= 0 ? minRT0 : minRT0 - 3.0*h; if (z0 < 0) z0 = 0;
    float z1 = maxRT0 > 10.0 ? 10.0 : maxRT0 + 3.0*h;
    int ngrid = 1024;
    int half_ngrid  = 0.5*ngrid;
    size_t ngrid_plus1fSize = (ngrid + 1) * sizeof(float);
    size_t ngriduSize = ngrid * sizeof(unsigned int);
    arma::vec z = arma::linspace<arma::vec>((double)z0, (double)z1, ngrid);
    float dt = z[1] - z[0];
    
    arma::vec filter0(ngrid);
    double z1minusz0 = (double)(z1 - z0);
    double fil0_constant = (double)(-2.0*h*h*M_PI*M_PI) / (z1minusz0*z1minusz0);

    float *h_binedge0, *d_binedge0;
    unsigned int *h_hist0, *d_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    for(size_t i=0; i<ngrid; i++) {
      h_binedge0[i] = z0 + dt*((float)i - 0.5); // binedge
      h_hist0[i]    = 0;  // initialize histogram
      if (i < (1 + half_ngrid)) {       // Get filter (1024)
        filter0[i] = std::exp(fil0_constant * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    
    h_binedge0[ngrid] = (z0 + ((float)(ngrid - 1))*dt);
    
    if (*debug) printf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) printf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) printf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    
    // Get histogram (1024)---------------------------------------------------80
    CHECK(cudaMalloc((void**) &d_binedge0, ngrid_plus1fSize)); // 1025
    CHECK(cudaMalloc((void**) &d_hist0, ngriduSize));          // 1024
    CHECK(cudaMemcpy(d_binedge0, h_binedge0, ngrid_plus1fSize, cudaMemcpyHostToDevice)); free(h_binedge0);
    CHECK(cudaMemcpy(d_hist0,    h_hist0,    ngriduSize,       cudaMemcpyHostToDevice));
    
    histc_kernel<<<*nsim/ngrid, ngrid>>>(d_binedge0, d_RT, d_nsim, d_hist0);
    cudaFree(d_RT); cudaFree(d_binedge0); cudaFree(d_nsim);
    

    CHECK(cudaMemcpy(h_hist0, d_hist0, ngriduSize, cudaMemcpyDeviceToHost)); cudaFree(d_hist0); 
    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); free(h_nsim); 

    // FFT: Get simulated PDF ---------------------------
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; 
    arma::vec eDen; // a container for estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(size_t i=0; i<*nx; i++) { 
      out[i] = (eDen[i] < 1e-10 || std::isnan(eDen[i])) ? 1e-10 : eDen[i]; 
    }
  }
}


void n1PDF_plba2(double *x, int *nx, int *nsim, double *b, double *A, double *mean_v,
                 int *nmean_v, double *sd_v, double *sd_w, double *t0, double *mean_w, double *rD,
                 //double *swt, int *nth, bool *debug, double *out) {
                double *swt, int *nth, bool *debug, double *RT, int *R, double *out) {
  size_t nsimfSize = *nsim * sizeof(double);
  size_t nsimuSize = *nsim * sizeof(unsigned int);
  float *d_RT; unsigned int *d_R;
  float *h_RT; unsigned int *h_R;
  h_RT  = (float *)malloc(nsimfSize);
  h_R   = (unsigned int *)malloc(nsimuSize);
  CHECK(cudaMalloc((void**) &d_RT, nsimfSize));
  CHECK(cudaMalloc((void**) &d_R,  nsimuSize));

  double *T0;
  T0       = (double *)malloc(sizeof(double) * 1);
  *T0 = *rD + *swt;

  rplba2_n1(nsim, b, A, mean_v, nmean_v, mean_w, sd_v, sd_w, t0, T0, nth, d_R, d_RT); 

  CHECK(cudaMemcpy(h_R,  d_R,  nsimuSize, cudaMemcpyDeviceToHost)); 
  CHECK(cudaMemcpy(h_RT, d_RT, nsimfSize, cudaMemcpyDeviceToHost));
  for(int i=0; i<*nsim; i++) {
       RT[i] = h_RT[i];
       R[i]  = h_R[i];
  }
  free(h_R); free(h_RT);

  // ------------------------------------------------------------------------80
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

  free(h_sqsum_out); 
  free(h_n1min_out); free(h_n1max_out); free(h_sum_out);

  float minRT0 = min_tmp.min();
  float maxRT0 = max_tmp.max();
  int nsRT0    = h_count_out[0]; free(h_count_out);
  float sd     = std::sqrt( (sqsum - (sum*sum)/nsRT0) / (nsRT0 - 1) );
  
  if (*debug) printf("RT0 [minimum maximum sum sqsum nsRT0 sd] values: %.2f %.2f %.2f %.2f %d %.2f\n",
    min_tmp.min(), max_tmp.max(), sum, sqsum, nsRT0, sd);

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT); cudaFree(d_nsim); free(h_nsim);
    for(size_t i=0; i<*nx; i++) { out[i] = 1e-10; }
  } else {
    float h  = 0.09*sd*std::pow((float)*nsim, -0.2);
    float z0 = minRT0 <= 0 ? minRT0 : minRT0 - 3.0*h; if (z0 < 0) z0 = 0;
    float z1 = maxRT0 > 10.0 ? 10.0 : maxRT0 + 3.0*h;
    int ngrid = 1024;
    int half_ngrid  = 0.5*ngrid;
    size_t ngrid_plus1fSize = (ngrid + 1) * sizeof(float);
    size_t ngriduSize = ngrid * sizeof(unsigned int);
    arma::vec z = arma::linspace<arma::vec>((double)z0, (double)z1, ngrid);
    float dt = z[1] - z[0];
    
    arma::vec filter0(ngrid);
    double z1minusz0 = (double)(z1 - z0);
    double fil0_constant = (double)(-2.0*h*h*M_PI*M_PI) / (z1minusz0*z1minusz0);

    float *h_binedge0, *d_binedge0;
    unsigned int *h_hist0, *d_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    for(size_t i=0; i<ngrid; i++) {
      h_binedge0[i] = z0 + dt*((float)i - 0.5); // binedge
      h_hist0[i]    = 0;  // initialize histogram
      if (i < (1 + half_ngrid)) {       // Get filter (1024)
        filter0[i] = std::exp(fil0_constant * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    
    h_binedge0[ngrid] = (z0 + ((float)(ngrid - 1))*dt);
    
    if (*debug) printf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) printf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) printf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    
    // Get histogram (1024)---------------------------------------------------80
    CHECK(cudaMalloc((void**) &d_binedge0, ngrid_plus1fSize)); // 1025
    CHECK(cudaMalloc((void**) &d_hist0, ngriduSize));          // 1024
    CHECK(cudaMemcpy(d_binedge0, h_binedge0, ngrid_plus1fSize, cudaMemcpyHostToDevice)); free(h_binedge0);
    CHECK(cudaMemcpy(d_hist0,    h_hist0,    ngriduSize,       cudaMemcpyHostToDevice));
    
    histc_kernel<<<*nsim/ngrid, ngrid>>>(d_binedge0, d_RT, d_nsim, d_hist0);
    cudaFree(d_RT); cudaFree(d_binedge0); cudaFree(d_nsim);
    

    CHECK(cudaMemcpy(h_hist0, d_hist0, ngriduSize, cudaMemcpyDeviceToHost)); cudaFree(d_hist0); 
    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); free(h_nsim); 

    // FFT: Get simulated PDF ---------------------------
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; 
    arma::vec eDen; // a container for estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(size_t i=0; i<*nx; i++) { 
      out[i] = (eDen[i] < 1e-10 || std::isnan(eDen[i])) ? 1e-10 : eDen[i]; 
    }
  }
}

void n1PDF_plba3(double *x, int *nx, int *nsim, double *B, double *A, double *C, double *mean_v,
                 int *nmean_v, double *sd_v, double *sd_w, double *t0, double *mean_w, double *rD,
                 //double *swt, int *nth, bool *debug, double *out) {
                 double *tD, double *swt, int *nth, bool *debug, double *RT, int *R, double *out) {
  size_t nsimfSize = sizeof(float) * (*nsim);
  size_t nsimuSize = sizeof(unsigned int) * (*nsim);
  size_t fSize     = sizeof(float) * 1;
  size_t vfSize    = sizeof(float) * (*nmean_v);
  float *d_RT; unsigned int *d_R;
  float *h_RT; unsigned int *h_R;
  float *b, *c;
  h_RT = (float *)malloc(nsimfSize);
  h_R  = (unsigned int *)malloc(nsimuSize);
  b    = (float *)malloc(vfSize);
  c    = (float *)malloc(vfSize);
  for(size_t i=0; i<*nmean_v; i++) {
      b[i] = A[i] + B[i];
      c[i] = b[i] + C[i];
  }

  float swt_r = *rD + *swt;
  float swt_b = *tD + *swt;
  float *swt1, *swt2, *swtD;
  bool *a;
  swt1 = (float *)malloc(fSize);
  swt2 = (float *)malloc(fSize);
  swtD = (float *)malloc(fSize);
  a    = (bool  *)malloc(sizeof(bool) * 3);
  a[0] = false;
  a[1] = false;
  a[2] = false;
  if (swt_r == swt_b) {       // condition 0: rate and thresold change co-occur
    a[0] = true;
    *swt1 = swt_r;
    *swt2 = swt_r;
  } else if (swt_b < swt_r) { // condition 1: threshold change occurs early
    a[1] = true;
    *swt1 = swt_b;
    *swt2 = swt_r;
  } else { // (swt_b > swt_r); condition 2: rate change occurs early
    a[2] = true;
    *swt1 = swt_r;
    *swt2 = swt_b;
  }
  *swtD = *swt2 - *swt1;

  CHECK(cudaMalloc((void**) &d_RT, nsimfSize));
  CHECK(cudaMalloc((void**) &d_R,  nsimuSize));

  rplba3_n1(nsim, b, A, c, mean_v, nmean_v, mean_w, sd_v, sd_w, t0, swt1, swt2, swtD, a, nth, d_R, d_RT); 

  CHECK(cudaMemcpy(h_R,  d_R,  nsimuSize, cudaMemcpyDeviceToHost)); 
  CHECK(cudaMemcpy(h_RT, d_RT, nsimfSize, cudaMemcpyDeviceToHost));
  for(int i=0; i<*nsim; i++) {
       RT[i] = h_RT[i];
       R[i]  = h_R[i];
  }
  free(h_R); free(h_RT);
  free(b); free(c); free(swt1); free(swt2); free(swtD); free(a);

  // ------------------------------------------------------------------------80
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

  free(h_sqsum_out);
  free(h_n1min_out); free(h_n1max_out); free(h_sum_out);

  float minRT0 = min_tmp.min();
  float maxRT0 = max_tmp.max();
  int nsRT0    = h_count_out[0];   free(h_count_out);
  float sd     = std::sqrt( (sqsum - (sum*sum)/nsRT0) / (nsRT0 - 1) );
  
  if (*debug) printf("RT0 [minimum maximum sum sqsum nsRT0 sd] values: %.2f %.2f %.2f %.2f %d %.2f\n",
    min_tmp.min(), max_tmp.max(), sum, sqsum, nsRT0, sd);

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT); cudaFree(d_nsim); free(h_nsim);
    for(size_t i=0; i<*nx; i++) { out[i] = 1e-10; }
  } else {
    float h  = 0.09*sd*std::pow((float)*nsim, -0.2);
    float z0 = minRT0 <= 0 ? minRT0 : minRT0 - 3.0*h; if (z0 < 0) z0 = 0;
    float z1 = maxRT0 > 10.0 ? 10.0 : maxRT0 + 3.0*h;
    int ngrid = 1024;
    int half_ngrid  = 0.5*ngrid;
    size_t ngrid_plus1fSize = (ngrid + 1) * sizeof(float);
    size_t ngriduSize = ngrid * sizeof(unsigned int);
    arma::vec z = arma::linspace<arma::vec>((double)z0, (double)z1, ngrid);
    float dt = z[1] - z[0];
    
    arma::vec filter0(ngrid);
    double z1minusz0 = (double)(z1 - z0);
    double fil0_constant = (double)(-2.0*h*h*M_PI*M_PI) / (z1minusz0*z1minusz0);

    float *h_binedge0, *d_binedge0;
    unsigned int *h_hist0, *d_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    for(size_t i=0; i<ngrid; i++) {
      h_binedge0[i] = z0 + dt*((float)i - 0.5); // binedge
      h_hist0[i]    = 0;  // initialize histogram
      if (i < (1 + half_ngrid)) {       // Get filter (1024)
        filter0[i] = std::exp(fil0_constant * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    
    h_binedge0[ngrid] = (z0 + ((float)(ngrid - 1))*dt);
    
    if (*debug) printf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) printf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) printf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    
    // Get histogram (1024)---------------------------------------------------80
    CHECK(cudaMalloc((void**) &d_binedge0, ngrid_plus1fSize)); // 1025
    CHECK(cudaMalloc((void**) &d_hist0, ngriduSize));          // 1024
    CHECK(cudaMemcpy(d_binedge0, h_binedge0, ngrid_plus1fSize, cudaMemcpyHostToDevice)); free(h_binedge0);
    CHECK(cudaMemcpy(d_hist0,    h_hist0,    ngriduSize,       cudaMemcpyHostToDevice));
    
    histc_kernel<<<*nsim/ngrid, ngrid>>>(d_binedge0, d_RT, d_nsim, d_hist0);
    cudaFree(d_RT); cudaFree(d_binedge0); cudaFree(d_nsim);
    

    CHECK(cudaMemcpy(h_hist0, d_hist0, ngriduSize, cudaMemcpyDeviceToHost)); cudaFree(d_hist0); 
    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); free(h_nsim); 

    // FFT: Get simulated PDF ---------------------------
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; 
    arma::vec eDen; // a container for estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(size_t i=0; i<*nx; i++) { 
      out[i] = (eDen[i] < 1e-10 || std::isnan(eDen[i])) ? 1e-10 : eDen[i]; 
    }
  }
}
