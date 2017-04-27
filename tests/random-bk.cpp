#include <unistd.h>
#include <stdio.h>  // C printing
#include <assert.h> // C check
#include <math.h>   // C math
#include <curand.h>        // Host random API
#include <curand_kernel.h> // Device random API
#include "../inst/include/common.h"  
#include "../inst/include/constant.h"  
#include "../inst/include/util.h"
#include "../inst/include/reduce.h"  
#include <armadillo> // Armadillo vector operations

extern "C" void runif_entry(int *n, double *min, double *max, int *nth,
  double *out);
extern "C" void rnorm_entry(int *n, double *mean, double *sd, int *nth,
  double *out);
extern "C" void rtnorm_entry(int *n, double *mean, double *sd, double *l,
  double *u, int *nth, double *out);
extern "C" void rlba_entry(int *n, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, int *nsd_v,
  double *t0, int *nth, double *RT, int *R);


extern "C" void dlba_gpu(double *RT0, double *RT1, int *nRT0, int *nRT1,
  int *nsim,   double *b,  double *A,
  double *v,   int *nv,     double *sv, double *t0,
  int *nThread, double *den1, double *den2);

extern "C" void n1PDF_entry(double *RT0, int *nRT0, int *nsim, double *b, double *A,
  double *mean_v, int *nmean_v, double *sd_v, int *nsd_v,
  double *t0, int *nth, double *den0, bool *debug);

extern "C" void n1PDF_test(double *RT0, int *ndata, int *n, double *b,
  double *A, double *mean_v, int *nmean_v,
  double *sd_v, int *nsd_v, double *t0,
  int *nth, bool *debug, double *out);

/*
extern "C" void n1PDF_test(double *RT0, int *nRT0, int *nsim, double *b,
  double *A, double* mean_v, int *nmean_v,
double *sd_v, int *nsd_v, double *t0, int *nth,
bool *debug, double *out);
*/

__global__ void runif_kernel(int n, double* min, double* max, double* out) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  // 'clock64()' allows each run produces different numbers.
  // '+ threadID' allows each thread generates different numbers.
  // Not place threadID on the 2nd argument speed up for 100 times.
  curandState_t state;
  curand_init((clock64() << 20)+threadID, 0, 0, &state);
  for (int i = threadID; i < n; i += numThreads) {
    out[i] = curand_uniform_double(&state);
    out[i] *= (max[0] - min[0] + 0.0000001);
    out[i] += min[0];
  }
}

__global__ void rnorm_kernel(int n, double* mean, double* sd, double* out) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
  for (int i = threadID; i < n; i += numThreads) {
    out[i] = mean[0] + sd[0]*curand_normal_double(&state);
  }
}

__global__ void rtnorm0_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) {
  // Accept-Reject Algorithm 0;
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int valid;
  double z;
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
  for (int i = threadID; i < n; i += numThreads) {
    valid = 0;
    while (valid == 0) {
      z = curand_normal_double(&state);
      if (z <= *u && z >= *l) {
        out[i] = z*(*sd) + *mean;
        break;
      }
    }
  }
}

__global__ void rtnorm1_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) {
  // Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  int valid;              // alphaStar must be greater than 0
  double z, r, num; // a stands for alphaStar in Robert (1995)
  double a = 0.5 * (sqrt(l[0] * l[0] + 4.0) + l[0]);
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
  
  for (int i = threadID; i < n; i += numThreads) {
    valid = 0;
    while (valid == 0) {
      // rexp via runif. a=0, b=1/alphaStar; see Saucier (2000)
      //   assert( b > 0.);
      //   return a - b * log( uniform(0., 1.)) 
      z   = (-1/a) * log(curand_uniform_double(&state)) + l[0];
      num = curand_uniform_double(&state);
      r   = exp(-0.5 * (z - a) * (z - a));
      if (num <= r && z <= *u) {
        out[i] = z*(*sd) + *mean;
        break;
      }
    }
  }
}

__global__ void rtnorm2_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) {
  // Algorithm 2; 'expu'; use when upper < mean; lower = -INFINITY.
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  int valid;
  double z, r, num; // a stands for alphaStar in Robert (1995)
  double a = 0.5 * (sqrt(u[0] * u[0] + 4.0) - u[0]);
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
  
  for (int i = threadID; i < n; i += numThreads) {
    valid = 0;
    while (valid == 0) {
      z   = (-1/a) * log(curand_uniform_double(&state)) - u[0];
      num = curand_uniform_double(&state);
      r   = exp(-0.5 * (z - a) * (z - a));
      if (num <= r && z <= -l[0]) {
        out[i] = -z*(*sd) + *mean; // note '-' before z
        break;
      }
    }
  }
}

__global__ void rtnorm3_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) {
  // Algorithm 3; u;  page 123. 2.2. Two-sided truncated normal dist.
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  int valid;
  double z, r, num;
  double maxminusmin = u[0] - l[0] + 1e-7;
  double lower2 = l[0] * l[0];
  double upper2 = u[0] * u[0];
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
  
  for (int i = threadID; i < n; i += numThreads) {
    valid = 0;
    while (valid == 0) {
      z = l[0] + maxminusmin*curand_uniform_double(&state);
      if (l[0] > 0) {
        r = exp( 0.5 * (lower2 - z*z) );
      } else if (u[0] < 0) {
        r = exp( 0.5 * (upper2 - z*z) );
      } else { 
        r = exp( -0.5 * z * z );
      }
      
      num = curand_uniform_double(&state);
      if (num <= r) {
        out[i] = z*(*sd) + *mean;
        break;
      }
    }
  }
}

static __device__ inline
  double rtnorm0_device(curandState_t* state, double mean, double sd,
    double l, double u) {
    double z, out; 
    int valid = 0;
    while (valid == 0) {
      z = curand_normal_double(&state[0]);
      if (z <= u && z >= l) {
        out = z*sd + mean;
        break;
      }
    }
    return out;
  }

static __device__ inline
  double rtnorm1_device(curandState_t* state, double a, double mean,
    double sd, double l, double u) {
    double z, r, num, out; // a stands for alphaStar in Robert (1995)
    int valid = 0;
    while (valid == 0) {
      z   = (-1/a) * log(curand_uniform_double(&state[0])) + l;
      num = curand_uniform_double(&state[0]);
      r   = exp(-0.5 * (z - a) * (z - a));
      if (num <= r && z <= u) {
        out = z*sd + mean;
        break;
      }
    }
    return out;
  }


__global__ void rlba_kernel(int* n, double* b, double* A, double* mean_v,
  double* sd_v, double* t0, double* lower,
  double* upper, double* a, bool* c, double* RT, unsigned int* R) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  double v0, v1, dt0, dt1;
  
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (int i = threadID; i < n[0]; i += numThreads)
  {
    v0 = c[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]) :
    rtnorm1_device(&state, a[0], mean_v[0], sd_v[0], lower[0], upper[0]);
    v1 = c[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]) :
      rtnorm1_device(&state, a[1], mean_v[1], sd_v[1], lower[1], upper[1]);
    dt0 = (b[0] - A[0] * curand_uniform_double(&state)) / v0;
    dt1 = (b[0] - A[0] * curand_uniform_double(&state)) / v1;
    RT[i] = (dt0 < dt1) ? (dt0 + t0[0]) : (dt1 + t0[0]);
    R[i]  = (dt0 < dt1) ? 1 : 2;
  }
}

void runif_entry(int *n, double *min, double *max, int *nth, double *out)
{
  double *d_min, *d_max, *d_out, *h_out;  
  CHECK(cudaMalloc((void**) &d_out,   *n * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_min,    1  * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_max,    1  * sizeof(double)));
  CHECK(cudaHostAlloc((void**)&h_out, *n * sizeof(double), cudaHostAllocDefault));
  CHECK(cudaMemcpy(d_min, min, 1*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_max, max, 1*sizeof(double), cudaMemcpyHostToDevice));
  runif_kernel<<<((*n)/(*nth) + 1), *nth>>>((*n), d_min, d_max, d_out);
  CHECK(cudaMemcpy(h_out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost));
  for(int i=0; i<*n; i++) { out[i] = h_out[i]; }
  cudaFreeHost(h_out);   cudaFree(d_out);
  cudaFree(d_min);       cudaFree(d_max);
}


void rnorm_entry(int *n, double *mean, double *sd, int *nth, double *out)
{
  double *d_out, *d_mean, *d_sd, *h_out;
  CHECK(cudaHostAlloc((void**)&h_out, *n * sizeof(double), cudaHostAllocDefault));
  CHECK(cudaMalloc((void**)&d_out,    *n * sizeof(double)));
  CHECK(cudaMalloc((void**)&d_mean,    1 * sizeof(double)));
  CHECK(cudaMalloc((void**)&d_sd,      1 * sizeof(double)));
  CHECK(cudaMemcpy(d_mean, mean,       1 * sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sd, sd,           1 * sizeof(double), cudaMemcpyHostToDevice));
  rnorm_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_out);
  CHECK(cudaMemcpy(h_out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost));
  for(int i=0; i<*n; i++) { out[i] = h_out[i]; }
  cudaFreeHost(h_out); cudaFree(d_out);
  cudaFree(d_mean);    cudaFree(d_sd);
}


void rtnorm_entry(int *n, double *mean, double *sd, double *l, double *u, int *nth, double *out)
{
  double *h_out, *d_out, *d_mean, *d_sd, *d_l, *d_u, *h_stdu, *h_stdl; 
  h_stdl = (double *)malloc(sizeof(double) * (1));
  h_stdu = (double *)malloc(sizeof(double) * (1));
  h_stdl[0] = (*l - *mean) / *sd; // convert to mean=0, sd=1
  h_stdu[0] = (*u - *mean) / *sd;
  
  // Algorithm (a0): Use Naive A-R method
  // Algorithm (a1): Use Proposition 2.3 with only lower truncation. upper==INFINITY
  // rejection sampling with exponential proposal. Use if lower > mean
  // Algorithm (a2): Use -x ~ N_+ (-mu, -mu^+, sigma^2) on page 123. lower==-INFINITY
  // rejection sampling with exponential proposal. Use if upper < mean.
  // Algorithm (a3, else): rejection sampling with uniform proposal. Use if bounds
  // are narrow and central.
  bool a0 = (h_stdl[0] < 0 && h_stdu[0]==INFINITY)  ||
    (h_stdl[0] == -INFINITY && h_stdu[0] > 0) ||
    (std::isfinite(h_stdl[0]) && std::isfinite(h_stdu[0]) && h_stdl[0] < 0 && h_stdu[0] > 0 &&
    (h_stdu[0] - h_stdl[0]) > SQRT_2PI);
  double eq_a1 = h_stdl[0] + (2.0 * std::sqrt(M_E) / (h_stdl[0] + std::sqrt(h_stdl[0] * h_stdl[0] + 4.0))) *
    (std::exp( 0.25 * (2.0 * h_stdl[0] - h_stdl[0] * std::sqrt(h_stdl[0] * h_stdl[0] + 4.0))));
  bool a1 = (h_stdl[0] >= 0) && (h_stdu[0] > eq_a1);
  double eq_a2 = -h_stdu[0] + (2.0 * std::sqrt(M_E) / (-h_stdu[0] + std::sqrt(h_stdu[0]*h_stdu[0] + 4.0))) *
    (std::exp( 0.25 * (2.0 * h_stdu[0] + h_stdu[0] * std::sqrt(h_stdu[0]*h_stdu[0] + 4.0))));
  bool a2 = (h_stdu[0] <= 0) && (-h_stdl[0] > eq_a2);
  
  CHECK(cudaHostAlloc((void**)&h_out, *n * sizeof(double), cudaHostAllocDefault));
  CHECK(cudaMalloc((void**) &d_out,   *n * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_mean,  1  * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_sd,    1  * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_l,     1  * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_u,     1  * sizeof(double)));
  CHECK(cudaMemcpy(d_mean, mean, 1 * sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sd,     sd, 1 * sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_l,  h_stdl, 1 * sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_u,  h_stdu, 1 * sizeof(double), cudaMemcpyHostToDevice));
  
  if (a0) {
    rtnorm0_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
  } else if (a1) {
    rtnorm1_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
  } else if (a2) {
    rtnorm2_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
  } else {
    rtnorm3_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
  }
  
  cudaMemcpy(h_out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost);
  for(int i=0; i<*n; i++) { out[i] = h_out[i]; }
  
  free(h_stdl);  free(h_stdu);
  cudaFreeHost(h_out); cudaFree(d_out);
  cudaFree(d_mean);    cudaFree(d_sd);
  cudaFree(d_l);       cudaFree(d_u);
}


void rlba_entry(int *n, double *b,double *A, double *mean_v, int *nmean_v,
  double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, int *R)
{
  bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_gpu
  int *d_n;
  unsigned int *d_R, *h_R;
  double *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_RT, *d_l, *d_u, *d_a,
  *h_l, *h_u, *h_a, *h_RT; // *h_a and *d_a stands for alphaStart
  
  unsigned int v_dbytes    = nmean_v[0] * sizeof(double);
  unsigned int one_dbytes = 1 * sizeof(double);
  
  h_c = (bool *)malloc(nmean_v[0] * sizeof(bool));
  h_l = (double *)malloc(v_dbytes);
  h_u = (double *)malloc(v_dbytes);
  h_a = (double *)malloc(v_dbytes);
  
  for(int i=0; i<nmean_v[0]; i++) {
    h_l[i] = (0 - mean_v[i]) / sd_v[i]; // convert to mean=0, sd=1
    h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
    h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
    h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
      (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
  }
  
  CHECK(cudaHostAlloc((void**)&h_RT,  *n * sizeof(double), cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void**)&h_R,   *n * sizeof(unsigned int), cudaHostAllocDefault));
  CHECK(cudaMalloc((void**) &d_R,     *n * sizeof(unsigned int)));
  CHECK(cudaMalloc((void**) &d_RT,    *n * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_n,     1  * sizeof(int)));
  CHECK(cudaMalloc((void**) &d_b,     one_dbytes));
  CHECK(cudaMalloc((void**) &d_A,     one_dbytes));
  CHECK(cudaMalloc((void**) &d_t0,    one_dbytes));
  CHECK(cudaMalloc((void**) &d_sd_v,   v_dbytes));
  CHECK(cudaMalloc((void**) &d_mean_v, v_dbytes));
  CHECK(cudaMalloc((void**) &d_l,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_u,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_a,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_c,      nmean_v[0] * sizeof(bool)));
  
  CHECK(cudaMemcpy(d_n,  n,  1*sizeof(int),    cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b,  b,  one_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_A,  A,  one_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_t0, t0, one_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_mean_v, mean_v,   v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sd_v,   sd_v,     v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_l,      h_l,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_u,      h_u,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_a,      h_a,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_c,      h_c,      nmean_v[0]*sizeof(bool),   cudaMemcpyHostToDevice));
  
  rlba_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v, d_t0, d_l,
    d_u, d_a, d_c, d_RT, d_R);
  CHECK(cudaMemcpy(h_RT, d_RT, *n * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_R,  d_R,  *n * sizeof(unsigned int),    cudaMemcpyDeviceToHost));
  for(int i=0; i<*n; i++) {
    RT[i] = h_RT[i];
    R[i]  = h_R[i];
  }
  
  free(h_c); free(h_l); free(h_u); free(h_a);
  cudaFreeHost(h_RT); cudaFreeHost(h_R); 
  CHECK(cudaFree(d_RT));   CHECK(cudaFree(d_R));    CHECK(cudaFree(d_n));
  CHECK(cudaFree(d_b));    CHECK(cudaFree(d_A));    CHECK(cudaFree(d_t0));
  CHECK(cudaFree(d_mean_v)); CHECK(cudaFree(d_sd_v));
  CHECK(cudaFree(d_l)); CHECK(cudaFree(d_u));
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_c));   
}


__global__ void histc_kernel(double *binedge, double *rng, int *nrng, unsigned int *out) {
  __shared__ unsigned int cache[1024];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  const int numThreads = blockDim.x * gridDim.x; // total # of threads
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int i = threadID; i<nrng[0]; i += numThreads) {
    int j=0;
    double tmp=0;
    double sim = rng[i];
    
    if (sim < binedge[0] || sim > binedge[1024]) {
      // if out-of-range add 0 to the 1st bin, otherwise add 1 to j bin
      atomicAdd(&(cache[0]), 0);
    } else {
      // When 'sim' belongs to 'j' bin, the last line, 'j++' inside 
      // while loop will add one more '1' to j, before leaving the loop.
      // So I use cache[j-1].
      while(tmp==0) {
        tmp = ((sim >= binedge[j]) && (sim < binedge[j+1])); // 0 or 1;
        j++;
      }
      atomicAdd( &(cache[j-1]), 1);
    }
  }
  __syncthreads();
  // add partial histograms in each block together.
  atomicAdd( &(out[threadIdx.x]), cache[threadIdx.x] );
}


__global__ void histctest_kernel(double *binedge, double *RT, unsigned int *R, unsigned int *nrng, unsigned int *out) {
  __shared__ unsigned int cache[1024];
  cache[threadIdx.x] = 0;
  __syncthreads();
  
  const int numThreads = blockDim.x * gridDim.x; // total # of threads
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int i = threadID; i<nrng[0]; i += numThreads) {
    int j=0;
    double tmp=0;
    double sim = RT[i];
    double simR = R[i];
    
    if (simR==2 || sim < binedge[0] || sim > binedge[1024]) {
      // if out-of-range add 0 to the 1st bin, otherwise add 1 to j bin
      atomicAdd(&(cache[0]), 0);
    } else {
      // When 'sim' belongs to 'j' bin, the last line, 'j++' inside 
      // while loop will add one more '1' to j, before leaving the loop.
      // So I use cache[j-1].
      while(tmp==0) {
        tmp = ((sim >= binedge[j]) && (sim < binedge[j+1])); // 0 or 1;
        j++;
      }
      atomicAdd( &(cache[j-1]), 1);
    }
  }
  __syncthreads();
  // add partial histograms in each block together.
  atomicAdd( &(out[threadIdx.x]), cache[threadIdx.x] );
}



void n1PDF_test(double *RT0, int *ndata, int *n, double *b, double *A, double *mean_v, int *nmean_v,
  double *sd_v, int *nsd_v, double *t0, int *nth, bool *debug, double *out) {
  arma::vec data(*ndata);
  for(int i=0; i<*ndata; i++) { data[i] = RT0[i];}
  
  bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_gpu
  int *d_n;
  double *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_l, *d_u, *d_a,
  *h_l, *h_u, *h_a, *d_RT; // *h_a and *d_a stands for alphaStart
  unsigned int v_dbytes    = nmean_v[0] * sizeof(double);
  unsigned int one_dbytes = 1 * sizeof(double);
  unsigned int *d_R;
  
  h_c = (bool *)malloc(nmean_v[0] * sizeof(bool));
  h_l = (double *)malloc(v_dbytes);
  h_u = (double *)malloc(v_dbytes);
  h_a = (double *)malloc(v_dbytes);
  
  for(int i=0; i<nmean_v[0]; i++) {
    h_l[i] = (0 - mean_v[i]) / sd_v[i]; // convert to mean=0, sd=1
    h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
    h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
    h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
      (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
  }
  
  CHECK(cudaMalloc((void**) &d_RT,    *n * sizeof(double)));
  CHECK(cudaMalloc((void**) &d_R,     *n * sizeof(unsigned int)))
    
    CHECK(cudaMalloc((void**) &d_n,     1  * sizeof(int)));
  CHECK(cudaMalloc((void**) &d_b,     one_dbytes));
  CHECK(cudaMalloc((void**) &d_A,     one_dbytes));
  CHECK(cudaMalloc((void**) &d_t0,    one_dbytes));
  CHECK(cudaMalloc((void**) &d_sd_v,   v_dbytes));
  CHECK(cudaMalloc((void**) &d_mean_v, v_dbytes));
  CHECK(cudaMalloc((void**) &d_l,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_u,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_a,      v_dbytes));
  CHECK(cudaMalloc((void**) &d_c,      nmean_v[0] * sizeof(bool)));
  
  CHECK(cudaMemcpy(d_n,  n,  1*sizeof(int),    cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b,  b,  1*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_A,  A,  1*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_t0, t0, 1*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_mean_v, mean_v,   v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sd_v,   sd_v,     v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_l,      h_l,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_u,      h_u,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_a,      h_a,      v_dbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_c,      h_c,      nmean_v[0]*sizeof(bool),   cudaMemcpyHostToDevice));
  
  int maxThreads = 256;
  int nThread = (*n < maxThreads) ? nextPow2(*n) : maxThreads;
  int nBlk    = ((*n) + nThread - 1) / nThread / 2;
  
  rlba_kernel<<<nBlk, nThread>>>(d_n, d_b, d_A, d_mean_v, d_sd_v, d_t0, d_l,
    d_u, d_a, d_c, d_RT, d_R);
  free(h_c); free(h_l); free(h_u); free(h_a);
  cudaFree(d_n); cudaFree(d_b);  cudaFree(d_A);  cudaFree(d_t0); cudaFree(d_mean_v); cudaFree(d_sd_v);
  cudaFree(d_l); cudaFree(d_u); cudaFree(d_a); cudaFree(d_c);
  
  // Get information from Monte Carlo simulation
  double *h_minmax_out, *d_minmax_out, *h_sum_out, *d_sum_out, *h_sqsum_out, *d_sqsum_out;
  unsigned int *h_nsim, *d_nsim, *h_count_out, *d_count_out; 
  h_minmax_out = (double *)malloc(2*nBlk*sizeof(double));
  h_sum_out    = (double *)malloc(nBlk*sizeof(double));
  h_sqsum_out  = (double *)malloc(2*nBlk*sizeof(double));
  h_count_out  = (unsigned int *)malloc(2*nBlk*sizeof(unsigned int));
  h_nsim  = (unsigned int *)malloc(1*sizeof(unsigned));
  *h_nsim = (unsigned int)*n;
  
  for(unsigned int i=0; i<2*nBlk; i++) { h_count_out[i] = 0; }
  
  CHECK(cudaMalloc((void**) &d_minmax_out, 2*nBlk*sizeof(double)));
  CHECK(cudaMalloc((void**) &d_sum_out,    nBlk*sizeof(double)));
  CHECK(cudaMalloc((void**) &d_sqsum_out,  2*nBlk*sizeof(double)));
  CHECK(cudaMalloc((void**) &d_count_out,  2*nBlk*sizeof(unsigned int)));
  CHECK(cudaMalloc((void**) &d_nsim,       1*sizeof(unsigned int)));
  
  CHECK(cudaMemcpy(d_count_out, h_count_out, 2*nBlk*sizeof(unsigned int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nsim, h_nsim, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
  
  minmax_kernel<<<nBlk, nThread>>>(d_RT, d_minmax_out);
  sum_kernel<<<nBlk, nThread>>>(d_RT, d_sum_out);
  squareSum_kernel<<<2*nBlk, nThread>>>(d_nsim, d_RT, d_sqsum_out);
  count_kernel<<<2*nBlk, nThread>>>(d_nsim, d_R, d_count_out);
  
  CHECK(cudaMemcpy(h_minmax_out, d_minmax_out, 2*nBlk*sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_sum_out, d_sum_out, nBlk*sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_sqsum_out, d_sqsum_out, 2*nBlk*sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_count_out, d_count_out, 2*nBlk*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  
  arma::vec min_tmp(nBlk);
  arma::vec max_tmp(nBlk);
  arma::vec sum_tmp(nBlk);
  arma::vec sqsum_tmp(2*nBlk);
  
  for (int i=0; i<nBlk; i++) {
    min_tmp[i]  = h_minmax_out[i];
    max_tmp[i]  = h_minmax_out[i + nBlk];
    sum_tmp[i]  = h_sum_out[i];
  }
  for (int i=0; i<(2*nBlk); i++) { sqsum_tmp[i]  = h_sqsum_out[i]; }
  unsigned int nsRT0 = h_count_out[0];
  
  free(h_minmax_out); free(h_sum_out); free(h_sqsum_out); free(h_count_out);
  cudaFree(d_minmax_out); cudaFree(d_sum_out); cudaFree(d_sqsum_out); cudaFree(d_count_out);
  
  double minrt = min_tmp.min();
  double maxrt = max_tmp.max();
  double sumrt = arma::accu(sum_tmp);
  double sqsumrt = arma::accu(sqsum_tmp);
  double sd = std::sqrt( (sqsumrt - (sumrt*sumrt) / *n) / (*n - 1) );
  //printf("In rlba_test2:\n [min max sum sqsum sd]: [%f %f %f %f %f]\n", minrt, maxrt, sumrt, sqsumrt, sd);
  
  if (nsRT0 < 10 || minrt > data.max() || maxrt < data.min()) {
    for(int i=0; i<*ndata; i++) { out[i] = 1e-10; }
  } else {
    double h = 0.8 * sd * std::pow(double(*n), -0.2);
    double z0 = minrt - 3*h;
    double z1 = maxrt > 5.0 ? 5.0 + 3*h : maxrt + 3*h;
    int ngrid = 1024;
    int ngrid_plus1 = ngrid + 1;
    int half_ngrid  = 0.5*ngrid;
    arma::vec z = arma::linspace<arma::vec>(z0, z1, ngrid);
    double dt = z[1] - z[0];
    double *h_binedge;
    h_binedge = (double *)malloc(ngrid_plus1 * sizeof(double));
    // Get binedge (1025)-------------------------------------------------80
    for(int i=0; i<ngrid_plus1; i++) {
      h_binedge[i] = i < ngrid ? z0 + dt*((double)i - 0.5) : (z0 + (double)(i-1)*dt) +  0.5*dt;
    }
    // Get histogram (1024)------------------------------------------------80
    unsigned int *h_hist0, *d_hist0;
    h_hist0 = (unsigned int *)malloc(ngrid * sizeof(unsigned int));
    for(unsigned int i=0; i<ngrid; i++) { h_hist0[i] = 0;  }
    
    double *d_binedge;
    CHECK(cudaMalloc((void**) &d_binedge, (ngrid_plus1) * sizeof(double)));
    CHECK(cudaMalloc((void**) &d_hist0, ngrid * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_binedge, h_binedge, ngrid_plus1*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hist0, h_hist0, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    histctest_kernel<<<*n/ngrid+1, ngrid>>>(d_binedge, d_RT, d_R, d_nsim, d_hist0);
    cudaFree(d_RT); cudaFree(d_R); cudaFree(d_binedge); cudaFree(d_nsim);
    
    CHECK(cudaMemcpy(h_hist0, d_hist0, ngrid * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(d_hist0); free(h_binedge);
    
    // Get filter (1024)--------------------------------------------------80
    arma::vec filter0(ngrid);
    double z1minusz0 = z1 - z0;
    double fil0_constant0 = -2*h*h*M_PI*M_PI / (z1minusz0*z1minusz0);
    for(int i=0; i<ngrid; i++) {
      if (i < (1+half_ngrid)) {
        filter0[i] = std::exp(fil0_constant0 * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    // FFT ---------------------------
    arma::vec signal0(ngrid);
    for(int i=0; i<ngrid; i++) { signal0[i] = h_hist0[i] / (dt*(*n)); }
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; // simulated probability density function
    arma::vec eDen; // estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    
    for(int i=0; i<*ndata; i++) {
      if (eDen[i] < 1e-5 || std::isnan(eDen[i])) {
        out[i] = 1e-5;
      } else {
        out[i] = eDen[i];
      }
    }
  }
}





void histc_entry(double *binedge, double *rng, int nrng, int ngrid, unsigned int *out) {
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


void n1PDF_entry(double *RT0,  int *nRT0, int *nsim, double *b, double *A, double* mean_v,
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *den0, bool *debug) {
  
  arma::vec data(*nRT0);
  for(int i=0; i<*nRT0; i++) { data[i]=RT0[i];}
  
  double *h_RT;
  int *h_R;
  h_R  = (int *)malloc(*nsim * sizeof(int));
  h_RT = (double *)malloc(*nsim * sizeof(double));
  
  if (*debug) {
    printf("[b, A, t0]: [%.3f %.3f %.3f ] \n", *b, *A, *t0);
    for(int i=0; i<*nmean_v; i++) {
      printf("mean_v and sd_v[%d]: [%.3f %.3f]\n", i, mean_v[i], sd_v[i]);
    }
  }
  
  rlba_entry(nsim, b, A, mean_v, nmean_v, sd_v, nsd_v, t0, nth, h_RT, h_R);
  
  if (*debug) {
    printf("RT and R:\n");
    for(int i=0; i<10; i++) {
      printf("[%d]: [%.3f %d]\n", i, h_RT[i], h_R[i]);
    }
  }
  
  arma::vec sRT(*nsim), sR(*nsim), sRT0, tsRT0;
  for(int i=0; i<*nsim; i++) {
    sRT[i] = h_RT[i];
    sR[i]  = h_R[i];
  }
  free(h_RT); free(h_R);
  
  sRT0  = sRT.elem(arma::find(sR == 1));
  if(!sRT0.is_empty()) { tsRT0 = sRT0.elem(arma::find(sRT0 < 10.0));} // truncated, simulation RTs;
  int nsRT0 = sRT0.n_elem;
  
  if (*debug) {
    printf("Found Inf? %d or nan? %d\n", sRT0.has_inf(), sRT0.has_nan());
    printf("Numbers of successful simulations: %d. After truncation: %d\n",
      nsRT0, tsRT0.n_elem);
  }
  
  if (nsRT0 < 10 || tsRT0.min() > data.max() || tsRT0.max() < data.min()) {
    if (*debug) printf("Nubmers of simulation are less than 10!\n");
    for(int i=0; i<*nRT0; i++) { den0[i] = 1e-10; }
  } else {
    double h  = bwNRD0(tsRT0, 0.8);
    double z0 = tsRT0.min() - 3*h;
    double z1 = tsRT0.max() + 3*h;
    if (*debug) { printf("[h: %.2f z0: %.2f z1: %.2f]\n", h, z0, z1);}
    
    arma::uvec nanvec1 = arma::find(data < z0);
    arma::uvec nanvec2 = arma::find(data > z1);
    if (*debug) {
      printf("Lower\n");
      for(int i=0; i<nanvec1.n_elem; i++) { printf("%d ", nanvec1[i]); }
      printf("\nUpper\n");
      for(int i=0; i<nanvec2.n_elem; i++) { printf("%d ", nanvec2[i]); }
      printf("\n");
    }
    
    int ngrid = 1024;
    int ngrid_plus1  = ngrid + 1;
    int half_ngrid   = 0.5*ngrid;
    arma::vec z = arma::linspace<arma::vec>(z0, z1, ngrid);
    double dt = z[1] - z[0];
    double *binedge0;
    binedge0 = (double *)malloc(ngrid_plus1 * sizeof(double));
    // Get binedge (1025)-------------------------------------------------80
    for(int i=0; i<ngrid_plus1; i++) {
      binedge0[i] = i < ngrid ? z0 + dt*((double)i - 0.5) : (z0 + (double)(i-1)*dt) +  0.5*dt;
    }
    
    // if (*debug) {
    //     for(int i=0; i<10; i++) {
    //         printf("binedge0[%d]: %.3f\n", i, binedge0[i]);
    //      }
    // }
    // Get filter (1024)--------------------------------------------------80
    arma::vec filter0(ngrid);
    double z1minusz0 = z1 - z0;
    double fil0_constant0 = -2*h*h*M_PI*M_PI / (z1minusz0*z1minusz0);
    for(int i=0; i<ngrid; i++) {
      if (i < (1+half_ngrid)) {
        filter0[i] = std::exp(fil0_constant0 * (double)(i*i));
      } else { 
        int j = 2*(i - half_ngrid); // flipping
        filter0[i] = filter0[i-j];
      }
    }
    
    // Get histogram (1024)------------------------------------------------80
    // void histc_entry(double *binedge, double *rng, int nrng,
    //          int ngrid, unsigned int *out)
    double *h_tsRT0;
    unsigned int *h_hist0;
    int ntsRT0 = tsRT0.n_elem;
    h_hist0 = (unsigned int *)malloc(ngrid * sizeof(unsigned int));
    h_tsRT0 = (double *)malloc(ntsRT0 * sizeof(double));
    for(int i=0; i<ntsRT0; i++) { h_tsRT0[i] = tsRT0[i]; }
    for(int i=0; i<ngrid; i++) { h_hist0[i] = 0; }
    histc_entry(binedge0, h_tsRT0, ntsRT0, ngrid, h_hist0);
    
    // if (*debug) {
    //     for(int i=0; i<1024; i++) {
    //         printf("%d ", h_hist0[i]);
    //     }
    // }
    
    // FFT ---------------------------
    arma::vec signal0(ngrid);
    
    for(int i=0; i<ngrid; i++) { signal0[i] = h_hist0[i] / (dt*(*nsim)); }
    arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; // simulated probability density function
    
    arma::vec eDen; // estiamted densities
    arma::interp1(z, sPDF, data, eDen);
    for(int i=0; i<*nRT0; i++) {
      if (eDen[i] < 1e-5 || std::isnan(eDen[i])) {
        //if(*debug) printf("%.3f\t", eDen[i]);
        den0[i] = 1e-5;
      } else {
        den0[i] = eDen[i];
      }
    }
    if (*debug) printf("\n");
    free(h_tsRT0); free(h_hist0); free(binedge0); 
  }
}

/*

void n1PDF_test(double *RT0,  int *nRT0, int *nsim, double *b, double *A,
  double* mean_v, int *nmean_v, double *sd_v, int *nsd_v,
double *t0, int *nth, bool *debug, double *out) {

arma::vec data(*nRT0);
for(int i=0; i<*nRT0; i++) { data[i] = RT0[i];}

double *d_RT, *h_pda_parameters;
unsigned int *d_R, *sRTCounts;
CHECK(cudaMalloc((void**) &d_R,     *nsim * sizeof(unsigned int)));
CHECK(cudaMalloc((void**) &d_RT,    *nsim * sizeof(double)));
h_pda_parameters = (double *)malloc(3 * sizeof(double));
sRTCounts = (unsigned int *)malloc(2 * sizeof(unsigned int));

rlba_test2(nsim, b, A, mean_v, nmean_v, sd_v, nsd_v, t0, nth, d_RT, d_R,
  h_pda_parameters, sRTCounts);

double minrt = h_pda_parameters[0];
double maxrt = h_pda_parameters[1];
double sdrt  = h_pda_parameters[2];
unsigned int nsRT0 = sRTCounts[0];

if(nsRT0 < 10 || minrt > data.max() || maxrt < data.min()) {
for(int i=0; i<*nRT0; i++) { out[i] = 1e-10; }
} else {
double h = 0.8 * sdrt * std::pow(double(*nsim), -0.2);
double z0 = minrt - 3*h;
double z1 = maxrt > 10.0 ? 10.0 + 3*h : maxrt + 3*h;
int ngrid = 1024;
int ngrid_plus1 = ngrid + 1;
int half_ngrid  = 0.5*ngrid;
arma::vec z = arma::linspace<arma::vec>(z0, z1, ngrid);
double dt = z[1] - z[0];
double *h_binedge;
h_binedge = (double *)malloc(ngrid_plus1 * sizeof(double));
// Get binedge (1025)-------------------------------------------------80
for(int i=0; i<ngrid_plus1; i++) {
h_binedge[i] = i < ngrid ? z0 + dt*((double)i - 0.5) : (z0 + (double)(i-1)*dt) +  0.5*dt;
}
// Get filter (1024)--------------------------------------------------80
arma::vec filter0(ngrid);
double z1minusz0 = z1 - z0;
double fil0_constant0 = -2*h*h*M_PI*M_PI / (z1minusz0*z1minusz0);
for(int i=0; i<ngrid; i++) {
if (i < (1+half_ngrid)) {
filter0[i] = std::exp(fil0_constant0 * (double)(i*i));
} else { 
int j = 2*(i - half_ngrid); // flipping
filter0[i] = filter0[i-j];
}
}
// Get histogram (1024)------------------------------------------------80
unsigned int *h_hist0, *d_hist0;
unsigned int *h_nsim, *d_nsim;
h_hist0 = (unsigned int *)malloc(ngrid * sizeof(unsigned int));
//h_hist1 = (unsigned int *)malloc(ngrid * sizeof(unsigned int));
h_nsim  = (unsigned int *)malloc(1 * sizeof(unsigned int));
for(unsigned int i=0; i<ngrid; i++) { h_hist0[i] = 0;  }
*h_nsim = (unsigned int)*nsim;

double *d_binedge;
CHECK(cudaMalloc((void**) &d_binedge, (ngrid_plus1) * sizeof(double)));
CHECK(cudaMalloc((void**) &d_hist0, ngrid * sizeof(unsigned int)));
CHECK(cudaMalloc((void**) &d_nsim, 1 * sizeof(unsigned int)));
CHECK(cudaMemcpy(d_binedge, h_binedge, ngrid_plus1*sizeof(double), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_nsim, h_nsim, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_hist0, h_hist0, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice));

histctest_kernel<<<*nsim/ngrid+1, ngrid>>>(d_binedge, d_RT, d_R, d_nsim, d_hist0);
cudaFree(d_RT); cudaFree(d_R);
CHECK(cudaMemcpy(h_hist0, d_hist0, ngrid * sizeof(unsigned int), cudaMemcpyDeviceToHost));

// FFT ---------------------------
arma::vec signal0(ngrid);
for(int i=0; i<ngrid; i++) { signal0[i] = h_hist0[i] / (dt*(*nsim)); }
arma::vec sPDF = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; // simulated probability density function
arma::vec eDen; // estiamted densities
arma::interp1(z, sPDF, data, eDen);

for(int i=0; i<*nRT0; i++) {
if (eDen[i] < 1e-5 || std::isnan(eDen[i])) {
out[i] = 1e-5;
} else {
out[i] = eDen[i];
}
}
free(h_hist0); free(h_nsim);
cudaFree(d_hist0); cudaFree(d_nsim); cudaFree(d_binedge);
}
}

*/