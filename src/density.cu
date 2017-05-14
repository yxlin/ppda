//#include <unistd.h>
//#include <stdio.h>  // C printing
#include <R.h>  // R Rprintf
#include <curand_kernel.h> // Device random API
//#include "../inst/include/common.h"  
#include "../inst/include/reduce.h"
#include "../inst/include/random.h"
#include "../inst/include/util.h"  
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
                            int *nth, bool *debug, double *out);
                            //int *nth, bool *debug, double *RT, int *R, double *out);

extern "C" void n1PDF_plba3(double *x, int *nx, int *nsim, double *B, double *A,
                            double *C, double *mean_v, int *nmean_v, double *sd_v,
                            double *sd_w, double *t0, double *mean_w, double *rD,
                            double *tD, double *swt,
                            int *nth, bool *debug, double *out);
                            //int *nth, bool *debug, double *RT, int *R, double *out);

extern "C" void histc_entry(double *binedge, double *rng, int nrng, int ngrid, 
  unsigned int *out);


__global__ void histc_kernel(double *binedge, double *rng, unsigned int *nrng,
  unsigned int *out) {
  __shared__ unsigned int cache[1024];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j=0;
  double tmp = 0;
  
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
      // if (j > 1024) {  // internal check for memory leakage
      //     Rprintf("RT0[%d] is %f\n", i, rng[i]);
      //     Rprintf("%d j reaches 1024\n", j);
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
  unsigned int *h_nrng, *d_nrng;
  double *d_binedge, *d_rng, *h_binedge, *h_rng;
  unsigned int *d_hist, *h_hist;
  h_nrng = (unsigned int *)malloc(1 * sizeof(unsigned int));
  h_nrng[0] = (unsigned int)nrng;
  
  cudaHostAlloc((void**)&h_hist, ngrid * sizeof(unsigned int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_rng,  nrng * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_binedge,  (ngrid+1) * sizeof(double), cudaHostAllocDefault);
  for(int i=0; i<nrng; i++) { h_rng[i] = rng[i]; }
  for(int i=0; i<(ngrid+1); i++) { h_binedge[i] = binedge[i]; }
  
  cudaMalloc((void**) &d_nrng,    1 * sizeof(unsigned int));
  cudaMalloc((void**) &d_binedge, (ngrid+1) * sizeof(double));
  cudaMalloc((void**) &d_rng,     nrng * sizeof(double));
  cudaMalloc((void**) &d_hist,    ngrid * sizeof(unsigned int));
  cudaMemcpy(d_nrng, h_nrng,       1*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_binedge, h_binedge, (ngrid+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rng, h_rng,         nrng*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hist, h_hist,       ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
  
  histc_kernel<<<(nrng/1024), 1024>>>(d_binedge, d_rng, d_nrng, d_hist);
  cudaMemcpy(h_hist, d_hist, ngrid * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for(int i=0; i<ngrid; i++) { out[i] = h_hist[i]; }
  
  cudaFreeHost(h_hist); cudaFreeHost(h_rng); cudaFreeHost(h_binedge);
  free(h_nrng); 
  cudaFree(d_binedge);
  cudaFree(d_rng);
  cudaFree(d_nrng);
  cudaFree(d_hist);
}

void n1PDF(double *x, int *nx, int *nsim, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, double *t0, int *nth, bool *debug, double *out) 
{
  size_t nsimfSize = *nsim * sizeof(float);
  size_t nsimuSize = *nsim * sizeof(unsigned int);
  float *d_RT; unsigned int *d_R;
  cudaMalloc((void**) &d_RT, nsimfSize);
  cudaMalloc((void**) &d_R,   nsimuSize);
  rn1(nsim, b, A, mean_v, nmean_v, sd_v, t0, nth, d_R, d_RT); // run kernel
  
  // ------------------------------------------------------------------------80
  float *KDEStats;
  KDEStats = (float *)malloc(sizeof(float) * 4);
  summary(nsim, d_R, d_RT, KDEStats);
  float minRT0 = KDEStats[0];
  float maxRT0 = KDEStats[1];
  float sd     = KDEStats[2];
  int nsRT0    = (int)KDEStats[3];

  if (*debug) {
      Rprintf("RT0 [minimum maximum]: %.2f %.2f\n", minRT0, maxRT0);
      Rprintf("RT0 [nsRT0 sd]: %d %f\n", nsRT0, sd);
  }
  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
    cudaFree(d_RT); 
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

    float *h_binedge0;
    unsigned int *h_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025) and histogram (1024) -----------------
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
    if (*debug) Rprintf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    histc(nsim, ngrid, h_binedge0, d_RT, h_hist0); // d_RT is free inside histc
    if (*debug) Rprintf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) Rprintf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);

    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); 

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
  cudaMalloc((void**) &d_RT, nsimfSize);
  cudaMalloc((void**) &d_R,  nsimuSize);

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
  float *KDEStats;
  KDEStats = (float *)malloc(sizeof(float) * 4);
  summary(nsim, d_R, d_RT, KDEStats);
  float minRT0 = KDEStats[0];
  float maxRT0 = KDEStats[1];
  float sd     = KDEStats[2];
  int nsRT0    = (int)KDEStats[3];

  if (*debug) {
      Rprintf("RT0 [minimum maximum]: %.2f %.2f\n", minRT0, maxRT0);
      Rprintf("RT0 [nsRT0 sd]: %d %f\n", nsRT0, sd);
  }

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT); 
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

    float *h_binedge0;
    unsigned int *h_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    // Get histogram (1024)---------------------------------------------------80
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
    histc(nsim, ngrid, h_binedge0, d_RT, h_hist0); // d_RT is free inside histc
    
    if (*debug) Rprintf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) Rprintf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) Rprintf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);

    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); 

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
                 double *swt, int *nth, bool *debug, double *out) {
                 //double *swt, int *nth, bool *debug, double *RT, int *R, double *out) {
  size_t nsimfSize = *nsim * sizeof(double);
  size_t nsimuSize = *nsim * sizeof(unsigned int);
  float *d_RT; unsigned int *d_R;
  //float *h_RT; unsigned int *h_R;
  //h_RT  = (float *)malloc(nsimfSize);
  //h_R   = (unsigned int *)malloc(nsimuSize);
  cudaMalloc((void**) &d_RT, nsimfSize);
  cudaMalloc((void**) &d_R,  nsimuSize);

  double *T0;
  T0       = (double *)malloc(sizeof(double) * 1);
  *T0 = *rD + *swt;

  rplba2_n1(nsim, b, A, mean_v, nmean_v, mean_w, sd_v, sd_w, t0, T0, nth, d_R, d_RT); 

  // CHECK(cudaMemcpy(h_R,  d_R,  nsimuSize, cudaMemcpyDeviceToHost)); 
  // CHECK(cudaMemcpy(h_RT, d_RT, nsimfSize, cudaMemcpyDeviceToHost));
  // for(int i=0; i<*nsim; i++) {
  //      RT[i] = h_RT[i];
  //      R[i]  = h_R[i];
  // }
  // free(h_R); free(h_RT);

  // ------------------------------------------------------------------------80
  float *KDEStats;
  KDEStats = (float *)malloc(sizeof(float) * 4);
  summary(nsim, d_R, d_RT, KDEStats);
  float minRT0 = KDEStats[0];
  float maxRT0 = KDEStats[1];
  float sd     = KDEStats[2];
  int nsRT0    = (int)KDEStats[3];

  if (*debug) {
      Rprintf("RT0 [minimum maximum]: %.2f %.2f\n", minRT0, maxRT0);
      Rprintf("RT0 [nsRT0 sd]: %d %f\n", nsRT0, sd);
  }

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT); 
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

    float *h_binedge0;
    unsigned int *h_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);

    // Get binedge (1025)-----------------------------------------------------80
    // Get histogram (1024)---------------------------------------------------80
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
    histc(nsim, ngrid, h_binedge0, d_RT, h_hist0); // d_RT is free inside histc
    if (*debug) Rprintf("min max RT0 : %.3f %.3f\n", minRT0, maxRT0);
    if (*debug) Rprintf("h z0 z1: %.3f %.3f %.3f\n", h, z0, z1);
    if (*debug) Rprintf("binedge[0 & 1024]: %f %f\n", h_binedge0[0], h_binedge0[ngrid]);
    
    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { 
      signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); 
    }
    free(h_hist0); 

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
                 double *tD, double *swt, int *nth, bool *debug, double *out) {
                 //double *tD, double *swt, int *nth, bool *debug, double *RT, int *R, double *out) {
  size_t nsimfSize = sizeof(float) * (*nsim);
  size_t nsimuSize = sizeof(unsigned int) * (*nsim);
  size_t fSize     = sizeof(float) * 1;
  size_t vfSize    = sizeof(float) * (*nmean_v);
  float *d_RT; unsigned int *d_R;
  //float *h_RT; unsigned int *h_R;
  float *b, *c;
  //h_RT = (float *)malloc(nsimfSize);
  //h_R  = (unsigned int *)malloc(nsimuSize);
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
  } else {                    // condition 2: rate change occurs early
    a[2] = true;
    *swt1 = swt_r;
    *swt2 = swt_b;
  }
  *swtD = *swt2 - *swt1;

  cudaMalloc((void**) &d_RT, nsimfSize);
  cudaMalloc((void**) &d_R,  nsimuSize);

  rplba3_n1(nsim, b, A, c, mean_v, nmean_v, mean_w, sd_v, sd_w, t0, swt1, swt2, swtD, a, nth, d_R, d_RT); 

  // CHECK(cudaMemcpy(h_R,  d_R,  nsimuSize, cudaMemcpyDeviceToHost)); 
  // CHECK(cudaMemcpy(h_RT, d_RT, nsimfSize, cudaMemcpyDeviceToHost));
  // for(int i=0; i<*nsim; i++) {
  //      RT[i] = h_RT[i];
  //      R[i]  = h_R[i];
  // }
  // free(h_R); free(h_RT);
  free(b); free(c); free(swt1); free(swt2); free(swtD); free(a);

  // ------------------------------------------------------------------------80
  float *KDEStats;
  KDEStats = (float *)malloc(sizeof(float) * 4);
  summary(nsim, d_R, d_RT, KDEStats);
  float minRT0 = KDEStats[0];
  float maxRT0 = KDEStats[1];
  float sd     = KDEStats[2];
  int nsRT0    = (int)KDEStats[3];

  if (*debug) {
      Rprintf("RT0 [minimum maximum]: %.2f %.2f\n", minRT0, maxRT0);
      Rprintf("RT0 [nsRT0 sd]: %d %f\n", nsRT0, sd);
  }

  // ------------------------------------------------------------------------80
  arma::vec data(*nx);
  for(size_t i=0; i<*nx; i++) { data[i] = x[i]; }
  
  if (nsRT0 < 10 || (double)minRT0 > data.max() || (double)maxRT0 < data.min() || minRT0 < 0) {
      cudaFree(d_RT);
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

    // Get filter, binedge (1025) and histogram (1024)----------------------80
    arma::vec filter0(ngrid);
    double z1minusz0 = (double)(z1 - z0);
    double fil0_constant = (double)(-2.0*h*h*M_PI*M_PI) / (z1minusz0*z1minusz0);
    
    float *h_binedge0;
    unsigned int *h_hist0;
    h_binedge0 = (float *)malloc(ngrid_plus1fSize);
    h_hist0    = (unsigned int *)malloc(ngriduSize);
    
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
    histc(nsim, ngrid, h_binedge0, d_RT, h_hist0); // d_RT is free inside histc    

    arma::vec signal0(ngrid);
    for(size_t i=0; i<ngrid; i++) { signal0[i] = (double)((float)h_hist0[i] / (dt * (float)(*nsim))); }
    free(h_hist0); 

    // FFT: Get simulated PDF ---------------------------
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; 
    arma::vec eDen; // a container for estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(size_t i=0; i<*nx; i++) { 
      out[i] = (eDen[i] < 1e-10 || std::isnan(eDen[i])) ? 1e-10 : eDen[i]; 
    }
  }
}
